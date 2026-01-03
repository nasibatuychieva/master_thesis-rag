import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
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

FULLTEXT_INDEX_NAME = "chunk_text_ft"

SCRIPT_NAME = "RAG_Advanced_Sparse"

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

llm_router = ChatOpenAI(model=os.getenv("OPENAI_MODEL"))
llm_rerank = ChatOpenAI(model=os.getenv("OPENAI_MODEL"))
llm_answer = ChatOpenAI(model=os.getenv("OPENAI_MODEL"))

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
    """
    Build a Lucene-style query string for Neo4j fulltext search.
    Strategy:
      - Put rewritten query as "OR" terms
      - Add extracted keywords as boosts
    Note: Keep it simple and robust for technical docs.
    """
    # Basic tokenization 
    base = pre["rewritten_query"].replace("\n", " ").strip()
    base_terms = [t.strip() for t in base.split(" ") if t.strip()]

   
    base_terms = [t for t in base_terms if len(t) >= 2]

    # OR-combine base terms
    base_part = " OR ".join([f'"{t}"' for t in base_terms[:12]])  # cap

    # Boost keywords (phrases) slightly
    kw_part = " OR ".join([f'"{k}"^2' for k in pre["keywords"][:8]])

    if base_part and kw_part:
        return f"({base_part}) OR ({kw_part})"
    elif kw_part:
        return f"({kw_part})"
    elif base_part:
        return f"({base_part})"
    else:
    
        return '"arduino"'

def retrieve_chunks_sparse(pre: Dict[str, Any], k: int = 12) -> List[Document]:
    """
    Returns List[Document] so the rest of your pipeline stays unchanged.
    Pulls node properties into Document.metadata.
    """
    query_str = _build_fulltext_query(pre)

    cypher = f"""
    CALL db.index.fulltext.queryNodes($index_name, $q)
    YIELD node, score
    RETURN node, score
    ORDER BY score DESC
    LIMIT $k
    """

    docs: List[Document] = []
    with driver.session(database=DATABASE) as session:
        result = session.run(
            cypher,
            index_name=FULLTEXT_INDEX_NAME,
            q=query_str,
            k=k,
        )

        for record in result:
            node = record["node"]
            score = record["score"]

            text = node.get(TEXT_PROPERTY, "")
            if not text:
                continue

            # copy all node properties into metadata 
            meta = dict(node)
            meta["sparse_score"] = float(score)

            docs.append(Document(page_content=text, metadata=meta))

    return docs

# ----------------------------------------------------------------------------
# 4) Rerank + Context Fusion (unchanged)
# ----------------------------------------------------------------------------

def rerank_and_select(
    pre: Dict[str, Any], docs: List[Document], top_k: int = 6
) -> List[Document]:
    if not docs:
        return []

    doc_blocks = []
    for i, d in enumerate(docs):
        doc_blocks.append(
            f"[DOC {i} | file={d.metadata.get('file_name','?')} | score={d.metadata.get('sparse_score','?')}]\n"
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

    indices = []
    for tok in ranked.replace("\n", ",").split(","):
        if tok.strip().isdigit():
            idx = int(tok.strip())
            if 0 <= idx < len(docs):
                indices.append(idx)

    if not indices:
        indices = list(range(min(top_k, len(docs))))

    return [docs[i] for i in indices[:top_k]]

# ----------------------------------------------------------------------------
# 5) Final Answer + CONTEXT ITEMS (unchanged)
# ----------------------------------------------------------------------------

def answer_advanced_rag_sparse(
    question: str,
    *,
    retrieve_k: int = 12,
    top_k: int = 6,
) -> Tuple[str, List[Dict[str, Any]]]:

    pre = pre_retrieval(question)
    retrieved_docs = retrieve_chunks_sparse(pre, k=retrieve_k)
    selected_docs = rerank_and_select(pre, retrieved_docs, top_k=top_k)

    fused_context = "\n\n".join(
        f"From file '{d.metadata.get('file_name','?')}', "
        f"product '{d.metadata.get('product','?')}':\n{d.page_content}"
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
            "sparse_score": d.metadata.get("sparse_score", None),
        })

    context_items.append({
        "content": fused_context,
        "source": "fused_context",
        "id": "",
    })

    return answer, context_items

# ----------------------------------------------------------------------------
# 6) Logging Helper (unchanged)
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
# 7) Batch Mode
# ----------------------------------------------------------------------------

def run_batch():
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

            print(f"[QID {question_id}] {question}")
            answer, context_items = answer_advanced_rag_sparse(question)
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
# 8) Manual Mode
# ----------------------------------------------------------------------------

def manual():
    qid = input("Question ID (optional): ").strip()
    question = input("Question: ").strip()
    gold = input("Gold answer (optional): ").strip()

    answer, context_items = answer_advanced_rag_sparse(question)
    print("\nAnswer:\n", answer)

    safe_log(SCRIPT_NAME, qid, "manual", question, answer, gold, context_items)

# ----------------------------------------------------------------------------
# 9) Main
# ----------------------------------------------------------------------------

def main():
    print("ONLY_RAG_Sparse_Advanced")
    print("y = manual | n = batch | exit")

    while True:
        cmd = input("> ").strip().lower()
        if cmd in ("exit", "quit", "q"):
            break
        elif cmd in ("y", "yes"):
            manual()
        elif cmd in ("n", "no"):
            run_batch()

if __name__ == "__main__":
    main()
