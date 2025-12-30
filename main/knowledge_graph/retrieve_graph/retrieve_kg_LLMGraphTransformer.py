import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from neo4j_graphrag.generation import RagTemplate
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

import os

URI = os.getenv("NEO4J_URI")
AUTH_USER = os.getenv("NEO4J_USER")
AUTH_PASSWORD = os.getenv("NEO4J_PASSWORD")
DATABASE = "llmagraphtrkg"

SCRIPT_NAME = "LLMGraph_Vector_KG_Retriever"

from pathlib import Path
import os

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT")).expanduser().resolve()

QUESTIONS_PATH = (
    PROJECT_ROOT
    / "main"
    / "evaluation"
    / "graphrag"
    / "golden_answers_dataset.jsonl"
)


# Neo4j driver
driver = GraphDatabase.driver(URI, auth=(AUTH_USER, AUTH_PASSWORD))
driver.verify_connectivity()

# LLM for GraphRAG answer generation
llm = OpenAILLM(
    model_name=os.getenv("OPENAI_MODEL"),
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

prompt_template = RagTemplate(
    template=(
        "You are a technical support assistant.\n"
        "Answer the question using ONLY the provided context.\n"
        "Write a detailed, structured answer.\n"
        "- If the question asks for variants, list all variants.\n"
        "- Include key specs, ranges, and differences.\n"
        "- Use bullet points and short headings.\n"
        "If context is insufficient, say what is missing.\n\n"
        "Examples:\n"
        "{examples}\n\n"
        "Context:\n"
        "{context}\n\n"
        "Question:\n"
        "{query_text}\n\n"
        "Answer:\n"
    )
)


rag = GraphRAG(retriever=retriever, llm=llm,prompt_template=prompt_template)


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
    # Einmal versuchen – und wenn es knallt, soll es sichtbar sein.
    log_antwort(
        script,
        question_id,
        query_type,
        question,
        answer,
        gold_answer or "",
        context_items=context_items,
    )

# ---------------------------------------------------------------------------
# 4) Context retrieval (DIRECTLY from retriever) 
# ---------------------------------------------------------------------------

def retrieve_context_items(question: str, top_k: int) -> List[Dict[str, Any]]:
    """
    Retrieve context directly from VectorCypherRetriever and ensure we log the CHUNK NODE TEXT.
    Robust against differing return schemas and property names.
    """
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

    def _pick_chunk_text(r: Dict[str, Any]) -> str:
        """
        Prefer exactly the chunk node's text.
        Primary key is 'text' because retrieval_query returns: node.text AS text.
        Fallback to other common property names if graph schema differs.
        """
        for key in ("text", "chunk_text", "content", "page_content", "node_text"):
            val = r.get(key)
            if val is not None:
                s = str(val).strip()
                if s:
                    return s
        return ""

    context_items: List[Dict[str, Any]] = []


    if isinstance(results, list):
        for r in results:
            if isinstance(r, dict):
                text = _pick_chunk_text(r)
                if not text:
                
                    continue

                docs = r.get("documents", [])
                if not isinstance(docs, list):
                    docs = [str(docs)]

                context_items.append(
                    {
                        "content": text,  # logger uses this

                        # helpful metadata
                        "node_type": "chunk",
                        "source": ", ".join([str(x) for x in docs if x is not None]),
                        "id": str(r.get("chunk_id") or r.get("id") or ""),
                        "score": r.get("score", ""),
                        "entities": r.get("entities", []),
                        "documents": docs,
                    }
                )
            else:
                # Non-dict result: skip to ensure ONLY chunk text gets logged
                continue
    else:
      
        return []

    return context_items

# ---------------------------------------------------------------------------
# 5) Answering: GraphRAG for answer + retriever for context (reliable)
# ---------------------------------------------------------------------------

def answer_with_rag(question: str, top_k: int = 20) -> Tuple[str, List[Dict[str, Any]]]:
    response = rag.search(
        query_text=question,
        retriever_config={"top_k": top_k},
       
    )
    answer = (getattr(response, "answer", None) or "").strip()


    context_items = retrieve_context_items(question, top_k=top_k)

    return answer, context_items

# ---------------------------------------------------------------------------
# 6) Batch / Manual
# ---------------------------------------------------------------------------

def run_batch_from_file(top_k: int = 20):
    print(f"\n[INFO] Loading dataset from {QUESTIONS_PATH}\n")

    if not QUESTIONS_PATH.exists():
        print("[ERROR] golden_answers_dataset.jsonl not found.")
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
