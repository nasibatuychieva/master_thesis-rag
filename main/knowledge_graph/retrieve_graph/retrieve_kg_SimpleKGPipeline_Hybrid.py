import os
from pathlib import Path
from typing import Dict
from main.evaluation.logger import log_antwort
from dotenv import load_dotenv, find_dotenv
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.retrievers import HybridCypherRetriever
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.generation import GraphRAG

# ---------------------------------------------------------------------------
# 1) Konfiguration & Environment
# ---------------------------------------------------------------------------

load_dotenv(find_dotenv())

URI = "neo4j://127.0.0.1:7687"
AUTH_USER = "neo4j"
AUTH_PASSWORD = "master2025"
DATABASE = "simplekg"

# Neo4j-Driver (DB wird über ENV oder default gewählt)
driver = GraphDatabase.driver(URI, auth=(AUTH_USER, AUTH_PASSWORD))
driver.verify_connectivity()

# LLM für Schema + KG-Build
llm = OpenAILLM(
    model_name="gpt-4o-mini",
    model_params={"temperature": 0}
)

# Embedder für KG-Embeddings (für Knoten/Kanten)
embedder = OpenAIEmbeddings(
    model="text-embedding-3-small"
)
from neo4j_graphrag.generation.prompts import ERExtractionTemplate


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

# Create retriever
retriever = HybridCypherRetriever(
    driver,
    neo4j_database=DATABASE,
    vector_index_name="chunkEmbedding_simplekg",
    fulltext_index_name="chunkFulltext_simplekg",
    embedder=embedder,
    retrieval_query=retrieval_query,
)
rag = GraphRAG(retriever=retriever, llm=llm)

# Search

import json
from pathlib import Path

SCRIPT_NAME = "SimpleKGPipeline_HybridCypherRetriever"

# Pfad zu deinem Gold-Datensatz (wie in der anderen Pipeline)
QUESTIONS_PATH = Path(
    r"C:\Users\Nasiba\Documents\1 Master Data Science\Master Thesis\VS Code New\master_thesis-rag\main\evaluation\graphrag\golden_answers_dataset.jsonl"
)

# ---------------------------------------------------------------------------
# Logging-Helfer (nutzt deine neue log_antwort-Signatur)
# ---------------------------------------------------------------------------

def safe_log(script, question_id, query_type, question, answer, gold_answer):
    """
    Unified logging helper.
    """
    try:
        log_antwort(script, question_id, query_type, question, answer, gold_answer)
    except Exception:
        # absolute Fallback – zur Not ohne gold_answer
        try:
            log_antwort(script, question_id, query_type, question, answer, "")
        except Exception:
            # minimaler Fallback – ohne IDs/Typ
            log_antwort(script, "", "", question, answer, "")


# ---------------------------------------------------------------------------
# Antwortfunktion für diese Pipeline (GraphRAG über SimpleKGPipeline-KG)
# ---------------------------------------------------------------------------

def answer_with_graphrag(question: str, top_k: int = 20) -> str:
    """
    Ruft GraphRAG mit dem VectorCypherRetriever auf und gibt nur die Antwort zurück.
    """
    response = rag.search(
        query_text=question,
        retriever_config={"top_k": top_k},
        return_context=True,
    )
    return response.answer

# ---------------------------------------------------------------------------
# Batch-Modus: Fragen + Gold-Antworten aus JSONL
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

            # Robust: id / question_id / query_id akzeptieren
            question_id = obj.get("id") or obj.get("question_id") or obj.get("query_id")
            question    = obj.get("question")
            gold_answer = obj.get("gold_answer")
            query_type  = obj.get("query_type")  # z.B. factual / relational / summary

            if not question:
                continue

            print(f"[QID {question_id}] [{query_type}] {question}")
            answer = answer_with_graphrag(question)
            print(f"[ANSWER] {answer}\n")

            safe_log(SCRIPT_NAME, question_id, query_type, question, answer, gold_answer)

    print("\n[INFO] Batch processing completed.\n")


# ---------------------------------------------------------------------------
# Manueller Modus
# ---------------------------------------------------------------------------

def manual_question(top_k: int = 20):
    qid = input("Question ID (optional): ").strip() or None
    question = input("Question: ").strip()
    gold_answer = input("Gold Answer (optional): ").strip() or None

    if not question:
        print("Empty question, skipping.\n")
        return

    answer = answer_with_graphrag(question, top_k=top_k)
    print("\nAnswer:\n", answer, "\n")

    safe_log(SCRIPT_NAME, qid, question, answer, gold_answer)

# ---------------------------------------------------------------------------
# Main-Loop: User wählt manuell vs. Batch
# ---------------------------------------------------------------------------

def main_loop(top_k: int = 20):
    print("Retrieve_kg_SimpleKGPipeline (VectorCypher + GraphRAG)")
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
