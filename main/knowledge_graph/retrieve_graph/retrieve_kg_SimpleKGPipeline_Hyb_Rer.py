# ============================================================================
# SIMPLEKG_HYBRID_RERANK (FULLY ADAPTED TO YOUR SCHEMA)
# - Neo4j FULLTEXT (Lucene) + Neo4j VECTOR
# - Fusion (minmax) + LLM rerank
# - Schema-aware metadata: Product / Category / __Entity__ via :FROM_CHUNK -> Chunk
# - Robust Lucene query building (NO raw question into fulltext)
# - Full logging for your evaluation logger
# ============================================================================

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from dotenv import load_dotenv, find_dotenv
from neo4j import GraphDatabase

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from main.evaluation.logger import log_antwort

# ----------------------------------------------------------------------------
# 0) Config
# ----------------------------------------------------------------------------
load_dotenv(find_dotenv())

URI = "neo4j://127.0.0.1:7687"
AUTH_USER = "neo4j"
AUTH_PASSWORD = "master2025"
DATABASE = "simplekg"

FULLTEXT_INDEX_NAME = "chunkFulltext_simplekg"
VECTOR_INDEX_NAME = "chunkEmbedding_simplekg"

TEXT_PROPERTY = "text"
FILE_PROPERTY_CANDIDATES = ["file_name", "file"]  # if one doesn't exist, we just ignore

QUESTIONS_PATH = Path(
    r"C:\Users\Nasiba\Documents\1 Master Data Science\Master Thesis\VS Code New\master_thesis-rag\main\evaluation\graphrag\golden_answers_dataset_new.jsonl"
)

SCRIPT_NAME = "SimpleKG_Hybrid_KG_Retriever_Rerank"

# Hybrid parameters
ALPHA = 0.6
K_SPARSE = 30
K_DENSE = 30
CANDIDATE_CAP_FOR_RERANK = 25
RERANK_TOP_K = 6

DEBUG = True  # set False if too noisy

driver = GraphDatabase.driver(URI, auth=(AUTH_USER, AUTH_PASSWORD))
driver.verify_connectivity()

# Models
llm_router = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_rerank = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_answer = ChatOpenAI(model="gpt-4o-mini", temperature=0)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# ----------------------------------------------------------------------------
# 1) Helpers
# ----------------------------------------------------------------------------
LUCENE_SPECIAL = r'(\+|\-|\&\&|\|\||\!|\(|\)|\{|\}|\[|\]|\^|"|~|\*|\?|\:|\\|\/)'

def lucene_escape_term(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip())
    s = re.sub(LUCENE_SPECIAL, r"\\\1", s)
    return s

def safe_log(
    script: str,
    question_id: str,
    query_type: str,
    question: str,
    answer: str,
    gold_answer: str,
    context_items: Optional[List[Dict[str, Any]]] = None,
):
    try:
        log_antwort(
            script,
            question_id,
            query_type,
            question,
            answer,
            gold_answer or "",
            context_items=context_items,
        )
    except TypeError:
        log_antwort(script, question_id, query_type, question, answer, gold_answer or "")
    except Exception as e:
        print("[WARN] logging failed:", e)
        try:
            log_antwort(script, question_id, query_type, question, answer, "")
        except Exception:
            log_antwort(script, "", "", question, answer, "")

# ----------------------------------------------------------------------------
# 2) Pre-retrieval: rewrite + keywords (Lucene-safe)
# ----------------------------------------------------------------------------
def pre_retrieval(question: str) -> Dict[str, Any]:
    rewrite_prompt = f"""
Rewrite the following technical question to be concise, precise,
and unambiguous while preserving its meaning.

Question:
{question}
""".strip()
    rewritten = llm_router.invoke(rewrite_prompt).content.strip()

    kw_prompt = f"""
Given the following technical query:

{rewritten}

Return 5–10 important keywords or short phrases.
Rules:
- comma-separated
- no explanations
- prefer technical nouns (e.g., component names, interfaces, protocols)
""".strip()
    kw_out = llm_router.invoke(kw_prompt).content
    keywords = [k.strip() for k in kw_out.split(",") if k.strip()]

    # Lucene-escape each keyword/phrase
    keywords_esc = [lucene_escape_term(k) for k in keywords]

    return {"original": question, "rewritten": rewritten, "keywords": keywords_esc}

def build_fulltext_query(pre: Dict[str, Any]) -> str:
    """
    Build a Lucene query that is safe and effective.
    We DO NOT pass the raw question into Lucene.
    """
    kws = pre["keywords"][:10]
    if not kws:
        return "arduino"

    # phrases get quoted
    parts = []
    for k in kws:
        if " " in k:
            parts.append(f"\"{k}\"")
        else:
            parts.append(k)

    # OR query
    return " OR ".join(parts)

# ----------------------------------------------------------------------------
# 3) Sparse retrieval (Fulltext)
# ----------------------------------------------------------------------------
def retrieve_sparse(pre: Dict[str, Any], k: int) -> List[Dict[str, Any]]:
    q = build_fulltext_query(pre)

    cypher = """
    CALL db.index.fulltext.queryNodes($index, $q, {limit: $k})
    YIELD node, score
    RETURN node, score
    ORDER BY score DESC
    """
    rows: List[Dict[str, Any]] = []
    with driver.session(database=DATABASE) as session:
        res = session.run(cypher, index=FULLTEXT_INDEX_NAME, q=q, k=k)
        for r in res:
            node = r["node"]
            score = float(r["score"] or 0.0)
            text = (node.get(TEXT_PROPERTY) or "").strip()
            if not text:
                continue

            # chunk id is often missing -> use elementId
            chunk_id = node.get("chunk_id") or node.get("id") or node.element_id

            # file name may not exist -> ignore if missing
            file_name = ""
            for fp in FILE_PROPERTY_CANDIDATES:
                if fp in node and node.get(fp):
                    file_name = str(node.get(fp))
                    break

            rows.append({
                "chunk_id": str(chunk_id),
                "text": text,
                "file_name": file_name,
                "sparse_score": score,
            })

    if DEBUG:
        print(f"[DEBUG] sparse_query='{q}' sparse_hits={len(rows)}")

    return rows

# ----------------------------------------------------------------------------
# 4) Dense retrieval (Vector)
# ----------------------------------------------------------------------------
def retrieve_dense(query: str, k: int) -> List[Dict[str, Any]]:
    qvec = embeddings.embed_query(query)

    cypher = """
    CALL db.index.vector.queryNodes($index, $k, $qvec)
    YIELD node, score
    RETURN node, score
    ORDER BY score DESC
    """
    rows: List[Dict[str, Any]] = []
    with driver.session(database=DATABASE) as session:
        res = session.run(cypher, index=VECTOR_INDEX_NAME, k=k, qvec=qvec)
        for r in res:
            node = r["node"]
            score = float(r["score"] or 0.0)
            text = (node.get(TEXT_PROPERTY) or "").strip()
            if not text:
                continue

            chunk_id = node.get("chunk_id") or node.get("id") or node.element_id

            file_name = ""
            for fp in FILE_PROPERTY_CANDIDATES:
                if fp in node and node.get(fp):
                    file_name = str(node.get(fp))
                    break

            rows.append({
                "chunk_id": str(chunk_id),
                "text": text,
                "file_name": file_name,
                "dense_score": score,
            })

    if DEBUG:
        print(f"[DEBUG] dense_hits={len(rows)}")

    return rows

# ----------------------------------------------------------------------------
# 5) Fuse sparse+dense (minmax) + dedupe by chunk_id
# ----------------------------------------------------------------------------
def minmax(scores: List[float]) -> Dict[float, float]:
    if not scores:
        return {}
    mn, mx = min(scores), max(scores)
    if mx == mn:
        return {s: 1.0 for s in scores}
    return {s: (s - mn) / (mx - mn) for s in scores}

def fuse_hybrid(sparse_rows: List[Dict[str, Any]], dense_rows: List[Dict[str, Any]], alpha: float) -> List[Dict[str, Any]]:
    s_scores = [float(r.get("sparse_score", 0.0) or 0.0) for r in sparse_rows]
    d_scores = [float(r.get("dense_score", 0.0) or 0.0) for r in dense_rows]
    s_norm = minmax(s_scores)
    d_norm = minmax(d_scores)

    merged: Dict[str, Dict[str, Any]] = {}

    def upsert(row: Dict[str, Any], kind: str):
        cid = (row.get("chunk_id") or "").strip()
        if not cid:
            return
        if cid not in merged:
            merged[cid] = {
                "chunk_id": cid,
                "text": row.get("text", ""),
                "file_name": row.get("file_name", ""),
                "sparse_score": None,
                "dense_score": None,
                "sparse_norm": 0.0,
                "dense_norm": 0.0,
            }

        if kind == "sparse":
            s = float(row.get("sparse_score", 0.0) or 0.0)
            merged[cid]["sparse_score"] = s
            merged[cid]["sparse_norm"] = s_norm.get(s, 0.0)
        else:
            d = float(row.get("dense_score", 0.0) or 0.0)
            merged[cid]["dense_score"] = d
            merged[cid]["dense_norm"] = d_norm.get(d, 0.0)

        # keep best file_name if available
        if not merged[cid].get("file_name") and row.get("file_name"):
            merged[cid]["file_name"] = row["file_name"]

    for r in sparse_rows:
        upsert(r, "sparse")
    for r in dense_rows:
        upsert(r, "dense")

    out: List[Dict[str, Any]] = []
    for cid, obj in merged.items():
        obj["hybrid_score"] = alpha * float(obj["dense_norm"]) + (1.0 - alpha) * float(obj["sparse_norm"])
        out.append(obj)

    out.sort(key=lambda x: float(x.get("hybrid_score", 0.0) or 0.0), reverse=True)
    return out

# ----------------------------------------------------------------------------
# 6) Schema-aware enrichment (THIS IS YOUR IMPORTANT PART)
#    We fetch Product/category and __Entity__ connected via :FROM_CHUNK -> Chunk
# ----------------------------------------------------------------------------
def enrich_with_schema(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not items:
        return []

    # We query by elementId OR stored id; safest: elementId
    chunk_ids = [it["chunk_id"] for it in items if it.get("chunk_id")]
    if not chunk_ids:
        return items

    cypher = """
    UNWIND $chunk_ids AS cid
    MATCH (c:Chunk)
    WHERE elementId(c) = cid OR c.id = cid OR c.chunk_id = cid

    OPTIONAL MATCH (c)<-[:FROM_CHUNK]-(p:Product)
    OPTIONAL MATCH (c)<-[:FROM_CHUNK]-(e:`__Entity__`)

    RETURN
      (c.id) AS cid_prop,
      elementId(c) AS cid_eid,
      collect(DISTINCT p.name) AS products,
      collect(DISTINCT p.category) AS categories,
      collect(DISTINCT e.name) AS entities,
      collect(DISTINCT e.type) AS entity_types
    """

    lookup: Dict[str, Dict[str, Any]] = {}
    with driver.session(database=DATABASE) as session:
        res = session.run(cypher, chunk_ids=chunk_ids)
        for r in res:
            cid_eid = r["cid_eid"]
            lookup[str(cid_eid)] = {
                "products": [x for x in (r["products"] or []) if x],
                "categories": [x for x in (r["categories"] or []) if x],
                "entities": [x for x in (r["entities"] or []) if x],
                "entity_types": [x for x in (r["entity_types"] or []) if x],
            }

    for it in items:
        meta = lookup.get(it["chunk_id"], {})
        it.update(meta)

    return items

# ----------------------------------------------------------------------------
# 7) LLM Rerank
# ----------------------------------------------------------------------------
def rerank(question: str, items: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    if not items:
        return []

    candidates = items[: min(len(items), CANDIDATE_CAP_FOR_RERANK)]

    blocks = []
    for i, it in enumerate(candidates):
        meta = []
        if it.get("products"):
            meta.append(f"products={it['products']}")
        if it.get("categories"):
            meta.append(f"categories={it['categories']}")
        if it.get("entities"):
            meta.append(f"entities={it['entities'][:10]}")
        meta_str = " | ".join(meta) if meta else "no_meta"

        blocks.append(
            f"[DOC {i} | hybrid={it.get('hybrid_score',0):.3f} | {meta_str}]\n{it['text']}"
        )

    prompt = f"""
You are a reranker for technical documentation QA.

Query:
{question}

Documents:
{chr(10).join(blocks)}

Return ONLY a comma-separated list of the {top_k} most relevant document indices (e.g., "0,3,5").
No explanations.
""".strip()

    out = llm_rerank.invoke(prompt).content.strip()

    idxs: List[int] = []
    for tok in out.replace("\n", ",").split(","):
        tok = tok.strip()
        if tok.isdigit():
            j = int(tok)
            if 0 <= j < len(candidates):
                idxs.append(j)

    if not idxs:
        idxs = list(range(min(top_k, len(candidates))))

    return [candidates[i] for i in idxs[:top_k]]

# ----------------------------------------------------------------------------
# 8) Final answer generation (uses ONLY reranked context)
# ----------------------------------------------------------------------------
def answer_question(question: str) -> Tuple[str, List[Dict[str, Any]]]:
    pre = pre_retrieval(question)

    sparse = retrieve_sparse(pre, k=K_SPARSE)
    dense = retrieve_dense(pre["rewritten"], k=K_DENSE)

    fused = fuse_hybrid(sparse, dense, alpha=ALPHA)
    fused = enrich_with_schema(fused)

    if DEBUG:
        print(f"[DEBUG] fused_total={len(fused)} top5_hybrid={[round(x.get('hybrid_score',0),3) for x in fused[:5]]}")

    if not fused:
        # logging-friendly
        ctx = [{"content": "[NO CANDIDATES FROM RETRIEVAL]", "source": "system", "id": "", "score": ""}]
        return "I don't know.", ctx

    selected = rerank(pre["rewritten"], fused, top_k=RERANK_TOP_K)

    context = "\n\n---\n\n".join(
        f"[CTX {i}] file={it.get('file_name','')} hybrid={it.get('hybrid_score',0):.3f}\n"
        f"products={it.get('products',[])}\n"
        f"categories={it.get('categories',[])}\n"
        f"text:\n{it['text']}"
        for i, it in enumerate(selected, start=1)
    )

    answer_prompt = f"""
You are an assistant answering questions about Arduino documentation.

Question:
{question}

Context:
{context}

Rules:
- Answer concisely (max 5 sentences).
- Use ONLY information supported by the context.
- If the context is insufficient, say: "I don't know."
""".strip()

    answer = llm_answer.invoke(answer_prompt).content.strip()

    # context_items for your logger
    context_items: List[Dict[str, Any]] = []
    for it in selected:
        context_items.append({
            "content": it["text"],
            "source": it.get("file_name", ""),
            "id": it.get("chunk_id", ""),
            "hybrid_score": it.get("hybrid_score", None),
            "dense_score": it.get("dense_score", None),
            "sparse_score": it.get("sparse_score", None),
            "products": it.get("products", []),
            "categories": it.get("categories", []),
            "entities": it.get("entities", []),
        })

    return answer, context_items

# ----------------------------------------------------------------------------
# 9) Batch / Manual
# ----------------------------------------------------------------------------
def run_batch():
    print(f"\n[INFO] Loading dataset from {QUESTIONS_PATH}\n")
    if not QUESTIONS_PATH.exists():
        print("[ERROR] dataset not found.")
        return

    with QUESTIONS_PATH.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                obj = json.loads(line)
            except Exception:
                print(f"[WARN] Invalid JSON at line {line_no}, skipped.")
                continue

            qid = str(obj.get("id") or obj.get("question_id") or obj.get("query_id") or "")
            qtype = str(obj.get("query_type") or "")
            question = str(obj.get("question") or "")
            gold = str(obj.get("gold_answer") or "")

            if not question:
                continue

            print(f"[QID {qid}] [{qtype}] {question}")
            answer, ctx = answer_question(question)

            print("[ANSWER]")
            print(answer)
            print(f"\n[CTX] reranked={len(ctx)}\n")

            safe_log(SCRIPT_NAME, qid, qtype, question, answer, gold, context_items=ctx)

def manual():
    qid = input("Question ID (optional): ").strip() or ""
    qtype = input("Query type (optional): ").strip() or "manual"
    question = input("Question: ").strip()
    gold = input("Gold answer (optional): ").strip() or ""

    answer, ctx = answer_question(question)

    print("\n[ANSWER]\n", answer)
    print(f"\n[CTX] reranked={len(ctx)}\n")

    safe_log(SCRIPT_NAME, qid, qtype, question, answer, gold, context_items=ctx)

def main():
    print("SIMPLEKG_HYBRID_RERANK_FIXED")
    print("y = manual | n = batch | exit\n")
    while True:
        cmd = input("> ").strip().lower()
        if cmd in ("exit", "quit", "q"):
            break
        if cmd in ("y", "yes"):
            manual()
        elif cmd in ("n", "no"):
            run_batch()
        else:
            print("Please enter y/n/exit\n")

if __name__ == "__main__":
    try:
        main()
    finally:
        driver.close()
