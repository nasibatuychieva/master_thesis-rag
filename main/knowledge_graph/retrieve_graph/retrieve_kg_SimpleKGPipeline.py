import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv, find_dotenv
from neo4j import GraphDatabase

from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.generation import RagTemplate
from main.evaluation.logger import log_antwort

# ---------------------------------------------------------------------------
# 1) Konfiguration & Environment
# ---------------------------------------------------------------------------

load_dotenv(find_dotenv())

import os

URI = os.getenv("NEO4J_URI")
AUTH_USER = os.getenv("NEO4J_USER")
AUTH_PASSWORD = os.getenv("NEO4J_PASSWORD")
DATABASE = "simplekg"

SCRIPT_NAME = "SimpleKG_Vector_KG_Retriever"

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


# Neo4j-Driver
driver = GraphDatabase.driver(URI, auth=(AUTH_USER, AUTH_PASSWORD))
driver.verify_connectivity()

# LLM (Answer Generation)
llm = OpenAILLM(
    model_name=os.getenv("OPENAI_MODEL"),
    model_params={"temperature": 0},
)

# Embedder (Vector Search)
embedder = OpenAIEmbeddings(model="text-embedding-3-small")

# ---------------------------------------------------------------------------
# 2) Retrieval query (used by VectorCypherRetriever internally)
# ---------------------------------------------------------------------------

retrieval_query = """
WITH node, score

// lokale Produkte an diesem Chunk
OPTIONAL MATCH (node)<-[:FROM_CHUNK]-(p0:Product)

// Kategorien dieser Produkte
WITH node, score,
     collect(DISTINCT p0) AS local_products,
     collect(DISTINCT p0.category) AS cats

UNWIND cats AS cat
OPTIONAL MATCH (p_all:Product {category: cat})

// weitere Entities zum Kontext
OPTIONAL MATCH (node)<-[:FROM_CHUNK]-(e:__Entity__)

// alles wieder einsammeln (und local_products im Scope behalten!)
WITH node, score,
     local_products,
     collect(DISTINCT cat)      AS categories,
     collect(DISTINCT p_all)    AS products_in_category,
     collect(DISTINCT e)        AS entities

RETURN DISTINCT
  node.text AS text,
  score     AS score,
  categories,
  [p IN products_in_category | p.name] AS products_in_category,
  [p IN local_products        | p.name] AS local_products,
  [en IN entities             | en.name] AS entities
"""

retriever = VectorCypherRetriever(
    driver,
    neo4j_database=DATABASE,
    index_name="chunkEmbedding_simplekg",
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
# 3) Logging helper (context_items wird übergeben)
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
# 4) Context retrieval: call retriever directly
# ---------------------------------------------------------------------------

def retrieve_context_items(question: str, top_k: int = 20) -> List[Dict[str, Any]]:
    """
    Retrieve context directly from VectorCypherRetriever (not from GraphRAG response).
    We log ONLY the chunk text (node.text) as content.
    """
    results = None

    # Try APIs across versions
    if hasattr(retriever, "retrieve"):
        try:
            results = retriever.retrieve(query_text=question, top_k=top_k)
        except TypeError:
            results = retriever.retrieve(question, top_k=top_k)
    elif hasattr(retriever, "search"):
        try:
            results = retriever.search(query_text=question, top_k=top_k)
        except TypeError:
            results = retriever.search(question, top_k=top_k)
    else:
        raise RuntimeError("VectorCypherRetriever has no retrieve/search method in this version.")

    context_items: List[Dict[str, Any]] = []


    if isinstance(results, list):
        for r in results:
            if isinstance(r, dict):
                # ✅ ONLY chunk text
                text = str(r.get("text") or "").strip()
                if not text:
                    continue

                score = r.get("score", "")

                context_items.append({
                    "content": text,  
                    "source": "simplekg_vector_index",
                    "score": score,
                })
            else:
                s = str(r).strip()
                if s:
                    context_items.append({"content": s, "source": "simplekg_vector_index", "score": ""})

    else:

        s = str(results).strip()
        if s:
            context_items.append({"content": s, "source": "simplekg_retriever_raw", "score": ""})

    return context_items

# ---------------------------------------------------------------------------
# 5) Answering: GraphRAG answer + Retriever context for logging
# ---------------------------------------------------------------------------

def answer_with_graphrag(question: str, top_k: int = 20) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Returns (answer, context_items).
    Answer from GraphRAG; context from retriever directly (reliable).
    """
    response = rag.search(
        query_text=question,
        retriever_config={"top_k": top_k},
 
    )
    answer = (getattr(response, "answer", None) or "").strip()


    context_items = retrieve_context_items(question, top_k=top_k)

    if not context_items:
        context_items = [{
            "content": "[NO CONTEXT RETURNED BY RETRIEVER]",
            "source": "system",
            "score": "",
        }]

    return answer, context_items

# ---------------------------------------------------------------------------
# 6) Batch mode (JSONL)
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
            question    = obj.get("question") or ""
            gold_answer = obj.get("gold_answer") or ""
            query_type  = obj.get("query_type") or ""

            if not question:
                continue

            print(f"[QID {question_id}] [{query_type}] {question}")

            answer, context_items = answer_with_graphrag(question, top_k=top_k)

            print(f"[ANSWER]\n{answer}\n")
            print(f"[CTX] n_context={len([c for c in context_items if c.get('content')])}\n")

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

# ---------------------------------------------------------------------------
# 7) Manual mode
# ---------------------------------------------------------------------------

def manual_question(top_k: int = 20):
    qid = input("Question ID (optional): ").strip() or ""
    qtype = input("Query type (optional, e.g., factual/relational/summary): ").strip() or "manual"
    question = input("Question: ").strip()
    gold_answer = input("Gold Answer (optional): ").strip() or ""

    if not question:
        print("Empty question, skipping.\n")
        return

    answer, context_items = answer_with_graphrag(question, top_k=top_k)

    print("\nAnswer:\n", answer, "\n")
    print(f"[CTX] n_context={len([c for c in context_items if c.get('content')])}\n")

    safe_log(
        SCRIPT_NAME,
        qid,
        qtype,
        question,
        answer,
        gold_answer,
        context_items=context_items,
    )

# ---------------------------------------------------------------------------
# 8) Main loop
# ---------------------------------------------------------------------------

def main_loop(top_k: int = 20):
    print("SimpleKG Pipeline (GraphRAG answer + Retriever context logging)")
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

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        main_loop(top_k=5)
    finally:
        driver.close()
