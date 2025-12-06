import os
from pathlib import Path
from typing import Dict
from main.evaluation.logger import log_antwort
from dotenv import load_dotenv, find_dotenv
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.retrievers import HybridCypherRetriever
import json
# ---------------------------------------------------------------------------
# 1) Konfiguration & Environment
# ---------------------------------------------------------------------------

load_dotenv(find_dotenv())

URI = "neo4j://127.0.0.1:7687"
AUTH_USER = "neo4j"
AUTH_PASSWORD = "master2025"
DATABASE = "llmagraphtrkg"

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

// Entities, die im Chunk erwähnt werden
OPTIONAL MATCH (node)-[:MENTIONS]->(e:Entity)

// optional: zugehöriges Dokument
OPTIONAL MATCH (node)-[:FROM_DOCUMENT]->(d:Document)

WITH node, score,
     collect(DISTINCT e) AS entities,
     collect(DISTINCT d) AS docs

RETURN DISTINCT
  node.text AS text,
  score     AS score,
  [en IN entities | en.id] AS entities,
  [d IN docs | d.id]       AS documents


"""


# Create retriever
retriever = HybridCypherRetriever(
    driver,
    neo4j_database=DATABASE,
    vector_index_name="chunkEmbedding_llmagraphtrkg",
    fulltext_index_name="chunkFulltext_llmagraphtrkg",
    embedder=embedder,
    retrieval_query=retrieval_query,
)
rag = GraphRAG(retriever=retriever, llm=llm)

# Search

def safe_log(script, question_id, query_type, question, answer, gold_answer):
    """
    Unified logging helper.
    """
    try:
        log_antwort(script, question_id, query_type, question, answer, gold_answer)
    except Exception:
        try:
            log_antwort(script, question_id, query_type, question, answer, "")
        except Exception:
            log_antwort(script, "", "", question, answer, "")

SCRIPT_NAME = "LLMGraphTransformer_HybridCypherRetriever"

QUESTIONS_PATH = Path(
    r"C:\Users\Nasiba\Documents\1 Master Data Science\Master Thesis\VS Code New\master_thesis-rag\main\evaluation\graphrag\golden_answers_dataset.jsonl"
)

def answer_with_rag(question: str, top_k: int = 20) -> str:
    """
    Verwendet deine neue RAG-Pipeline (rag.search), um eine Antwort zu generieren.
    """
    response = rag.search(
        query_text=question,
        retriever_config={"top_k": top_k},
        return_context=True,
    )
    return response.answer


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

            # id / question_id / query_id robust behandeln
            question_id = obj.get("id") or obj.get("question_id") or obj.get("query_id")
            question    = obj.get("question")
            gold_answer = obj.get("gold_answer")
            query_type  = obj.get("query_type")  # factual / relational / summary / ...

            if not question:
                continue

            print(f"[QID {question_id}] [{query_type}] {question}")
            answer = answer_with_rag(question, top_k=top_k)
            print(f"[ANSWER] {answer}\n")

            # Einheitliches Logging
            safe_log(SCRIPT_NAME, question_id, query_type, question, answer, gold_answer)

    print("\n[INFO] Batch processing completed.\n")


def manual_question(top_k: int = 20):
    qid = input("Question ID (optional): ").strip() or None
    question = input("Question: ").strip()
    gold_answer = input("Gold Answer (optional): ").strip() or None

    if not question:
        print("Empty question, skipping.\n")
        return

    answer = answer_with_rag(question, top_k=top_k)
    print("\nAnswer:\n", answer, "\n")

    safe_log(SCRIPT_NAME, qid, question, answer, gold_answer)

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

if __name__ == "__main__":
    try:
        main_loop(top_k=5)
    finally:
        driver.close()    