import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from dotenv import load_dotenv, find_dotenv
from neo4j import GraphDatabase

from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.generation import GraphRAG

from main.evaluation.logger import log_antwort

# ---------------------------------------------------------------------------
# 1) Config & Environment
# ---------------------------------------------------------------------------

load_dotenv(find_dotenv())

URI = "neo4j://127.0.0.1:7687"
AUTH_USER = "neo4j"
AUTH_PASSWORD = "master2025"
DATABASE = "llmagraphtrkg"

SCRIPT_NAME = "LLMGraph_Vector_KG_Retriever"

QUESTIONS_PATH = Path(
    r"C:\Users\Nasiba\Documents\1 Master Data Science\Master Thesis\VS Code New\master_thesis-rag\main\evaluation\graphrag\golden_answers_dataset_new.jsonl"
)

# Neo4j driver
driver = GraphDatabase.driver(URI, auth=(AUTH_USER, AUTH_PASSWORD))
driver.verify_connectivity()

# LLM for GraphRAG answer generation
llm = OpenAILLM(
    model_name="gpt-4o-mini",
    model_params={"temperature": 0},
)

# Embedder for vector search
embedder = OpenAIEmbeddings(model="text-embedding-3-small")

# ---------------------------------------------------------------------------
# 2) Retrieval query
# ---------------------------------------------------------------------------

retrieval_query = """
WITH node, score
OPTIONAL MATCH (node)-[:MENTIONS]->(e:Entity)
OPTIONAL MATCH (node)-[:FROM_DOCUMENT]->(d:Document)
WITH node, score,
     collect(DISTINCT e) AS entities,
     collect(DISTINCT d) AS docs
RETURN DISTINCT
  node.id   AS chunk_id,
  node.text AS text,
  score     AS score,
  [en IN entities | en.id] AS entities,
  [d IN docs | d.id]       AS documents
"""

retriever = VectorCypherRetriever(
    driver,
    neo4j_database=DATABASE,
    index_name="chunkEmbedding_llmagraphtrkg",
    embedder=embedder,
    retrieval_query=retrieval_query,
)

rag = GraphRAG(retriever=retriever, llm=llm)

# ---------------------------------------------------------------------------
# 3) Logging
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# 4) LlamaIndex-style context retrieval (DIRECTLY from retriever)
# ---------------------------------------------------------------------------

def retrieve_context_items(question: str, top_k: int) -> List[Dict[str, Any]]:
    """
    Retrieve context directly from VectorCypherRetriever (like LlamaIndex retrieve()).
    This is independent from GraphRAG response schema.
    """
    # Try different APIs depending on neo4j_graphrag version
    results = None

    if hasattr(retriever, "retrieve"):
        # common in some versions
        try:
            results = retriever.retrieve(query_text=question, top_k=top_k)
        except TypeError:
            # sometimes signature is retrieve(question, top_k=?)
            results = retriever.retrieve(question, top_k=top_k)

    elif hasattr(retriever, "search"):
        try:
            results = retriever.search(query_text=question, top_k=top_k)
        except TypeError:
            results = retriever.search(question, top_k=top_k)
    else:
        raise RuntimeError("VectorCypherRetriever has no retrieve/search method in this version.")

    context_items: List[Dict[str, Any]] = []

    # Most often: list[dict] based on retrieval_query RETURN fields
    if isinstance(results, list):
        for r in results:
            if isinstance(r, dict):
                text = str(r.get("text") or "").strip()
                if not text:
                    continue
                docs = r.get("documents", [])
                if not isinstance(docs, list):
                    docs = [str(docs)]
                context_items.append({
                    "content": text,
                    "source": ", ".join([str(x) for x in docs if x is not None]),
                    "id": str(r.get("chunk_id") or ""),
                    "score": r.get("score", ""),
                    "entities": r.get("entities", []),
                    "documents": docs,
                })
            else:
                # fallback: stringify
                s = str(r).strip()
                if s:
                    context_items.append({"content": s, "source": "", "id": "", "score": ""})

    else:
        # Some versions return an object; best-effort stringify
        s = str(results).strip()
        if s:
            context_items.append({"content": s, "source": "retriever_raw", "id": "", "score": ""})

    return context_items

# ---------------------------------------------------------------------------
# 5) Answering: GraphRAG for answer + retriever for context (reliable)
# ---------------------------------------------------------------------------

def answer_with_rag(question: str, top_k: int = 20) -> Tuple[str, List[Dict[str, Any]]]:
    response = rag.search(
        query_text=question,
        retriever_config={"top_k": top_k},
        # return_context=True  # doesn't matter; we don't depend on it anymore
    )
    answer = (getattr(response, "answer", None) or "").strip()

    # 🔥 LlamaIndex-style: get context from retriever directly
    context_items = retrieve_context_items(question, top_k=top_k)

    return answer, context_items

# ---------------------------------------------------------------------------
# 6) Batch / Manual
# ---------------------------------------------------------------------------

def run_batch_from_file(top_k: int = 20):
    print(f"\n[INFO] Loading dataset from {QUESTIONS_PATH}\n")

    if not QUESTIONS_PATH.exists():
        print("[ERROR] golden_answers_dataset_new.jsonl not found.")
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

            question_id = obj.get("id") or obj.get("question_id") or obj.get("query_id") or ""
            query_type  = obj.get("query_type") or ""
            question    = obj.get("question") or ""
            gold_answer = obj.get("gold_answer") or ""

            if not question:
                continue

            print(f"[QID {question_id}] [{query_type}] {question}")

            answer, context_items = answer_with_rag(question, top_k=top_k)

            print(f"[ANSWER]\n{answer}\n")
            print(f"[CTX] n_context={len(context_items)}\n")

            safe_log(
                SCRIPT_NAME,
                str(question_id),
                str(query_type),
                question,
                answer,
                gold_answer,
                context_items=context_items,
            )

    print("\n[INFO] Batch processing completed.\n")


def manual_question(top_k: int = 20):
    qid = input("Question ID (optional): ").strip()
    qtype = input("Query type (optional, e.g., factual/relational/summary): ").strip()
    question = input("Question: ").strip()
    gold_answer = input("Gold Answer (optional): ").strip()

    if not question:
        print("Empty question, skipping.\n")
        return

    answer, context_items = answer_with_rag(question, top_k=top_k)

    print("\nAnswer:\n", answer, "\n")
    print(f"[CTX] n_context={len(context_items)}\n")

    safe_log(
        SCRIPT_NAME,
        qid or "",
        qtype or "manual",
        question,
        answer,
        gold_answer or "",
        context_items=context_items,
    )


def main_loop(top_k: int = 20):
    print("LLMGraphTransformer_VectorCypherRetriever (GraphRAG answer + Retriever context)")
    print("Type 'exit' to quit.\n")

    while True:
        mode = input("Manual question? (y/n, or 'exit'): ").strip().lower()

        if mode in ("exit", "quit", "q"):
            break
        elif mode in ("y", "yes"):
            manual_question(top_k=top_k)
        elif mode in ("n", "no"):
            run_batch_from_file(top_k=top_k)
        else:
            print("Please enter 'y', 'n', or 'exit'.\n")


if __name__ == "__main__":
    try:
        main_loop(top_k=5)
    finally:
        driver.close()
