import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from neo4j import GraphDatabase

from main.evaluation.logger import log_antwort

# ----------------------------------------------------------------------------
# 0) Environment
# ----------------------------------------------------------------------------
load_dotenv(find_dotenv())

import os

URI = os.getenv("NEO4J_URI_RAG")
AUTH_USER = os.getenv("NEO4J_USER_RAG")
AUTH_PASSWORD = os.getenv("NEO4J_PASSWORD_RAG")
DATABASE = "rag"

NODE_LABEL = "Chunk"
TEXT_PROPERTY = "text"

# Sparse index (Neo4j fulltext)
FULLTEXT_INDEX_NAME = "chunk_text_ft"

# Dense index (Neo4j vector index)
VECTOR_INDEX_NAME = "rag_chunks"         
EMB_PROPERTY = "embedding"              

# Hybrid parameters
ALPHA = 0.6  
RETRIEVE_K_SPARSE = 30
RETRIEVE_K_DENSE = 30
RERANK_TOP_K = 6

SCRIPT_NAME = "RAG_Advanced_Hybrid"

from pathlib import Path
import os

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT")).expanduser().resolve()

QUESTIONS_PATH = (
    PROJECT_ROOT
    / "main"
    / "evaluation"
    / "evaluation_datasets"
    / "golden_answers_dataset_short.jsonl"
)


driver = GraphDatabase.driver(URI, auth=(AUTH_USER, AUTH_PASSWORD))
driver.verify_connectivity()

# ----------------------------------------------------------------------------
# 1) Models
# ----------------------------------------------------------------------------
llm_router = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
llm_rerank = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
llm_answer = ChatOpenAI(model="gpt-4.1-mini", temperature=0)


embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# ----------------------------------------------------------------------------
# 2) Pre-Retrieval
# ----------------------------------------------------------------------------
def pre_retrieval(query: str) -> Dict[str, Any]:
    rewrite_prompt = f"""
Rewrite the following technical question to be concise, precise,
and unambiguous while preserving its meaning.

Question:
{query}
"""
    rewritten = llm_router.invoke(rewrite_prompt).content.strip()

    expand_prompt = f"""
Given the following technical query:

{rewritten}

List 3–8 important keywords or short phrases (comma-separated, no explanations).
"""
    expansion = llm_router.invoke(expand_prompt).content
    keywords = [k.strip() for k in expansion.split(",") if k.strip()]

    return {
        "original_query": query,
        "rewritten_query": rewritten,
        "keywords": keywords,
    }

# ----------------------------------------------------------------------------
# 3) Sparse Retrieval (Neo4j Fulltext)
# ----------------------------------------------------------------------------
def _build_fulltext_query(pre: Dict[str, Any]) -> str:
    base = pre["rewritten_query"].replace("\n", " ").strip()
    base_terms = [t.strip() for t in base.split(" ") if t.strip()]
    base_terms = [t for t in base_terms if len(t) >= 2]

    base_part = " OR ".join([f'"{t}"' for t in base_terms[:12]])
    kw_part = " OR ".join([f'"{k}"^2' for k in pre["keywords"][:8]])

    if base_part and kw_part:
        return f"({base_part}) OR ({kw_part})"
    if kw_part:
        return f"({kw_part})"
    if base_part:
        return f"({base_part})"
    return '"arduino"'

def retrieve_sparse(pre: Dict[str, Any], k: int) -> List[Dict[str, Any]]:
    q = _build_fulltext_query(pre)

    cypher = """
    CALL db.index.fulltext.queryNodes($index_name, $q)
    YIELD node, score
    RETURN node, score
    ORDER BY score DESC
    LIMIT $k
    """
    rows: List[Dict[str, Any]] = []
    with driver.session(database=DATABASE) as session:
        res = session.run(cypher, index_name=FULLTEXT_INDEX_NAME, q=q, k=k)
        for r in res:
            node = r["node"]
            score = float(r["score"] or 0.0)
            text = node.get(TEXT_PROPERTY, "") or ""
            if not text.strip():
                continue

          
            chunk_id = node.get("chunk_id") or node.get("id") or ""

            rows.append({
                "chunk_id": str(chunk_id),
                "text": text,
                "meta": dict(node),
                "sparse_score": score,
            })
    return rows

# ----------------------------------------------------------------------------
# 4) Dense Retrieval (Neo4j Vector Index)
# ----------------------------------------------------------------------------
def retrieve_dense(query: str, k: int) -> List[Dict[str, Any]]:
    # embed query
    qvec = embeddings.embed_query(query)

    cypher = """
    CALL db.index.vector.queryNodes($index_name, $k, $qvec)
    YIELD node, score
    RETURN node, score
    ORDER BY score DESC
    """
    rows: List[Dict[str, Any]] = []
    with driver.session(database=DATABASE) as session:
        res = session.run(cypher, index_name=VECTOR_INDEX_NAME, k=k, qvec=qvec)
        for r in res:
            node = r["node"]
            score = float(r["score"] or 0.0)
            text = node.get(TEXT_PROPERTY, "") or ""
            if not text.strip():
                continue

            chunk_id = node.get("chunk_id") or node.get("id") or ""

            rows.append({
                "chunk_id": str(chunk_id),
                "text": text,
                "meta": dict(node),
                "dense_score": score,
            })
    return rows

# ----------------------------------------------------------------------------
# 5) Hybrid Fusion (union + normalized score)
# ----------------------------------------------------------------------------
def _minmax_norm(scores: List[float]) -> Dict[float, float]:
    if not scores:
        return {}
    mn = min(scores)
    mx = max(scores)
    if mx == mn:
        return {s: 1.0 for s in scores}
    return {s: (s - mn) / (mx - mn) for s in scores}

def fuse_hybrid(
    sparse_rows: List[Dict[str, Any]],
    dense_rows: List[Dict[str, Any]],
    *,
    alpha: float,
) -> List[Document]:
    # collect score lists for normalization
    sparse_scores = [float(r.get("sparse_score", 0.0) or 0.0) for r in sparse_rows]
    dense_scores = [float(r.get("dense_score", 0.0) or 0.0) for r in dense_rows]
    sparse_norm_map = _minmax_norm(sparse_scores)
    dense_norm_map = _minmax_norm(dense_scores)

    # merge by chunk_id
    merged: Dict[str, Dict[str, Any]] = {}

    def upsert(row: Dict[str, Any], kind: str):
        cid = (row.get("chunk_id") or "").strip()
        if not cid:
          
            cid = f"txt:{hash(row.get('text',''))}"

        if cid not in merged:
            merged[cid] = {
                "chunk_id": cid,
                "text": row.get("text", ""),
                "meta": row.get("meta", {}) or {},
                "sparse_score": None,
                "dense_score": None,
                "sparse_norm": 0.0,
                "dense_norm": 0.0,
            }

        if kind == "sparse":
            s = float(row.get("sparse_score", 0.0) or 0.0)
            merged[cid]["sparse_score"] = s
            merged[cid]["sparse_norm"] = sparse_norm_map.get(s, 0.0)
        else:
            d = float(row.get("dense_score", 0.0) or 0.0)
            merged[cid]["dense_score"] = d
            merged[cid]["dense_norm"] = dense_norm_map.get(d, 0.0)

       
        if isinstance(row.get("meta"), dict) and row["meta"]:
            merged[cid]["meta"].update(row["meta"])

    for r in sparse_rows:
        upsert(r, "sparse")
    for r in dense_rows:
        upsert(r, "dense")

    # compute hybrid score
    docs: List[Document] = []
    for cid, obj in merged.items():
        hybrid_score = alpha * float(obj["dense_norm"]) + (1.0 - alpha) * float(obj["sparse_norm"])
        meta = obj["meta"]
        meta["chunk_id"] = cid
        meta["sparse_score"] = obj["sparse_score"]
        meta["dense_score"] = obj["dense_score"]
        meta["sparse_norm"] = obj["sparse_norm"]
        meta["dense_norm"] = obj["dense_norm"]
        meta["hybrid_score"] = hybrid_score

        docs.append(Document(page_content=obj["text"], metadata=meta))

    docs.sort(key=lambda d: float(d.metadata.get("hybrid_score", 0.0) or 0.0), reverse=True)
    return docs

# ----------------------------------------------------------------------------
# 6) Rerank + Select
# ----------------------------------------------------------------------------
def rerank_and_select(pre: Dict[str, Any], docs: List[Document], top_k: int) -> List[Document]:
    if not docs:
        return []

    # keep the prompt small(ish)
    doc_blocks = []
    for i, d in enumerate(docs[: min(len(docs), 25)]):  
        doc_blocks.append(
            f"[DOC {i} | file={d.metadata.get('file_name','?')} | hybrid={d.metadata.get('hybrid_score','?'):.4f}]\n"
            f"{d.page_content}"
        )

    rerank_prompt = f"""
You are a reranker for technical documentation QA.

User query:
{pre["rewritten_query"]}

Documents:
{chr(10).join(doc_blocks)}

Task:
Return ONLY a comma-separated list of the {top_k} most relevant document indices.
"""
    ranked = llm_rerank.invoke(rerank_prompt).content

    indices: List[int] = []
    for tok in ranked.replace("\n", ",").split(","):
        tok = tok.strip()
        if tok.isdigit():
            idx = int(tok)
            if 0 <= idx < min(len(docs), 25):
                indices.append(idx)

    if not indices:
        indices = list(range(min(top_k, len(docs))))

    return [docs[i] for i in indices[:top_k]]

# ----------------------------------------------------------------------------
# 7) Final Answer (Hybrid RAG)
# ----------------------------------------------------------------------------
def answer_advanced_rag_hybrid(
    question: str,
    *,
    alpha: float = ALPHA,
    k_sparse: int = RETRIEVE_K_SPARSE,
    k_dense: int = RETRIEVE_K_DENSE,
    top_k: int = RERANK_TOP_K,
) -> Tuple[str, List[Dict[str, Any]]]:

    pre = pre_retrieval(question)

    sparse_rows = retrieve_sparse(pre, k=k_sparse)
    dense_rows = retrieve_dense(pre["rewritten_query"], k=k_dense)

    hybrid_docs = fuse_hybrid(sparse_rows, dense_rows, alpha=alpha)
    selected_docs = rerank_and_select(pre, hybrid_docs, top_k=top_k)

    fused_context = "\n\n".join(
        f"From file '{d.metadata.get('file_name','?')}', product '{d.metadata.get('product','?')}', "
        f"chunk_id '{d.metadata.get('chunk_id','')}':\n{d.page_content}"
        for d in selected_docs
    )

    answer_prompt = f"""
You are an assistant answering questions about Arduino product documentation.

Question:
{pre["original_query"]}

Context:
{fused_context}

Rules:
- Answer concisely (max 5 sentences).
- Use ONLY information supported by the context.
- If something is missing, say:
  "The documentation snippet does not contain this information."
"""
    answer = llm_answer.invoke(answer_prompt).content.strip()


    context_items: List[Dict[str, Any]] = []
    for d in selected_docs:
        context_items.append({
            "content": d.page_content,
            "source": d.metadata.get("file_name", ""),
            "product": d.metadata.get("product", ""),
            "id": d.metadata.get("chunk_id", d.metadata.get("id", "")),
            "hybrid_score": d.metadata.get("hybrid_score", None),
            "dense_score": d.metadata.get("dense_score", None),
            "sparse_score": d.metadata.get("sparse_score", None),
        })

    context_items.append({
        "content": fused_context,
        "source": "fused_context",
        "id": "",
    })

    return answer, context_items

# ----------------------------------------------------------------------------
# 8) Logging Helper
# ----------------------------------------------------------------------------
def safe_log(
    script: str,
    question_id: str,
    query_type: str,
    question: str,
    answer: str,
    gold_answer: str,
    context_items: List[Dict[str, Any]],
):
    try:
        log_antwort(
            script,
            question_id,
            query_type,
            question,
            answer,
            gold_answer,
            context_items=context_items,
        )
    except Exception as e:
        print("[WARN] logging failed:", e)
        log_antwort(script, "", "", question, answer, "")

# ----------------------------------------------------------------------------
# 9) Batch Mode
# ----------------------------------------------------------------------------
def run_batch(top_k: int = RERANK_TOP_K):
    if not QUESTIONS_PATH.exists():
        print("[ERROR] Dataset not found.")
        return

    with QUESTIONS_PATH.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue

            obj = json.loads(line)
            question = obj.get("question")
            if not question:
                continue

            question_id = str(obj.get("id", ""))
            query_type = obj.get("query_type", "")
            gold_answer = obj.get("gold_answer", "")

            print(f"[QID {question_id}] [{query_type}] {question}")
            answer, context_items = answer_advanced_rag_hybrid(question, top_k=top_k)
            print(answer, "\n")

            safe_log(
                SCRIPT_NAME,
                question_id,
                query_type,
                question,
                answer,
                gold_answer,
                context_items,
            )

# ----------------------------------------------------------------------------
# 10) Manual Mode
# ----------------------------------------------------------------------------
def manual(top_k: int = RERANK_TOP_K):
    qid = input("Question ID (optional): ").strip()
    question = input("Question: ").strip()
    gold = input("Gold answer (optional): ").strip()

    answer, context_items = answer_advanced_rag_hybrid(question, top_k=top_k)
    print("\nAnswer:\n", answer)

    safe_log(SCRIPT_NAME, qid, "manual", question, answer, gold, context_items)

# ----------------------------------------------------------------------------
# 11) Main
# ----------------------------------------------------------------------------
def main():
    print("ONLY_RAG_HYBRID_Advanced")
    print("y = manual | n = batch | exit")
    print(f"Hybrid settings: alpha={ALPHA}, k_sparse={RETRIEVE_K_SPARSE}, k_dense={RETRIEVE_K_DENSE}, top_k={RERANK_TOP_K}")
    print(f"Indexes: fulltext={FULLTEXT_INDEX_NAME} | vector={VECTOR_INDEX_NAME} (property {EMB_PROPERTY})\n")

    while True:
        cmd = input("> ").strip().lower()
        if cmd in ("exit", "quit", "q"):
            break
        elif cmd in ("y", "yes"):
            manual(top_k=RERANK_TOP_K)
        elif cmd in ("n", "no"):
            run_batch(top_k=RERANK_TOP_K)
        else:
            print("Please enter y/n/exit\n")

if __name__ == "__main__":
    try:
        main()
    finally:
        driver.close()
