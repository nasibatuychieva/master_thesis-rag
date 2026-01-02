import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from neo4j_graphrag.generation import RagTemplate
from dotenv import load_dotenv, find_dotenv
from neo4j import GraphDatabase

from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.retrievers import HybridCypherRetriever
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

SCRIPT_NAME = "LLMGraph_Hybrid_Retriever"

from pathlib import Path
import os

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT")).expanduser().resolve()

QUESTIONS_PATH = (
    PROJECT_ROOT
    / "main"
    / "evaluation"
    / "evaluation_datasets"
    / "golden_answers_dataset.jsonl"
)


# Neo4j driver
driver = GraphDatabase.driver(URI, auth=(AUTH_USER, AUTH_PASSWORD))
driver.verify_connectivity()

# LLM for GraphRAG answer generation
llm = OpenAILLM(
    model_name=os.getenv("OPENAI_MODEL"),
    
)

# Embedder for vector search
embedder = OpenAIEmbeddings(model="text-embedding-3-small")

# ---------------------------------------------------------------------------
# 2) Retrieval query (MATCHES YOUR SCHEMA!)
#    Chunks: id, text, file_name, embedding
#    Entities: often via :MENTIONS. We keep it flexible: any non-Chunk neighbor.
# ---------------------------------------------------------------------------

retrieval_query = """
WITH node, score
WHERE node:Chunk
  AND node.text IS NOT NULL
  AND trim(node.text) <> ""

OPTIONAL MATCH (node)-[:MENTIONS]-(e1)
WHERE NOT e1:Chunk

OPTIONAL MATCH (e1)-[:MENTIONS]-(node2:Chunk)
WHERE node2 <> node
  AND node2.text IS NOT NULL
  AND trim(node2.text) <> ""

OPTIONAL MATCH (e1)--(e2)
WHERE NOT e2:Chunk AND e2 <> e1

WITH
  node,
  score,
  collect(DISTINCT e1)    AS direct_entities,
  collect(DISTINCT node2) AS related_chunks,
  collect(DISTINCT e2)    AS related_entities

RETURN DISTINCT
  coalesce(node.chunk_id, node.id, elementId(node)) AS chunk_id,
  node.text                                         AS text,
  coalesce(node.file_name, node.file, "")           AS file_name,
  score                                             AS score,

  [e IN direct_entities |
    {
      id: coalesce(e.id, e.name, e.entity_id, elementId(e)),
      entityType: e.entityType
    }
  ] AS direct_entities,

  [n IN related_chunks | n.text][0..10] AS related_chunk_texts,

  [e IN related_entities |
    {
      id: coalesce(e.id, e.name, e.entity_id, elementId(e)),
      entityType: e.entityType
    }
  ] AS related_entities,

  CASE
    WHEN coalesce(node.file_name, node.file, "") = "" THEN []
    ELSE [coalesce(node.file_name, node.file, "")]
  END AS documents

"""

import re

LUCENE_SPECIAL = r'(\+|\-|\&\&|\|\||\!|\(|\)|\{|\}|\[|\]|\^|"|~|\*|\?|\:|\\|\/)'

def lucene_escape(s: str) -> str:
    s = re.sub(r"\s+", " ", s.strip())
    s = re.sub(LUCENE_SPECIAL, r"\\\1", s)  # slash "/" wird zu "\/"
    return s


retriever = HybridCypherRetriever(
    driver,
    vector_index_name="chunkEmbedding_llmagraphtrkg",
    fulltext_index_name="chunkFulltext_llmagraphtrkg",
    neo4j_database=DATABASE,
    embedder=embedder,
    retrieval_query=retrieval_query,
)


prompt_template = RagTemplate(
    template=(
       "You are a technical support assistant for Arduino Products.\n"
    "Use ONLY the provided context. Do not use outside knowledge.\n"
    "If the context does not contain the answer, say exactly what information is missing.\n"
    "Answer in complete sentences.\n"
    "Answer as completely as possible.\n"
    "Adapt the structure and style of the answer to the type of the question "
    "(e.g., list items for 'which' questions, explain processes for 'how' questions, "
    "and compare variants for 'difference' questions).\n\n"
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
# 4) Retrieve context directly from retriever (robust across versions)
# ---------------------------------------------------------------------------

def _call_retriever(question: str, top_k: int):
    """
    HybridCypherRetriever API differs between versions.
    Try common method names/signatures.
    """

    if hasattr(retriever, "search"):
        try:
            return retriever.search(query_text=question, top_k=top_k)
        except TypeError:
            return retriever.search(question, top_k=top_k)

    if hasattr(retriever, "retrieve"):
        try:
            return retriever.retrieve(query_text=question, top_k=top_k)
        except TypeError:
            return retriever.retrieve(question, top_k=top_k)

    raise RuntimeError("HybridCypherRetriever has no search/retrieve method in this version.")

def retrieve_context_items(question: str, top_k: int) -> List[Dict[str, Any]]:
    """
    Ensures the logged context is the CHUNK TEXT (node.text) and sets node_type='Chunk'
    so your logger's context_types_json is meaningful.
    """
    results = _call_retriever(question, top_k=top_k)

    context_items: List[Dict[str, Any]] = []

    # In neo4j_graphrag, retrieval_query RETURN usually yields list[dict]
    if isinstance(results, list):
        for r in results:
            if not isinstance(r, dict):
                s = str(r).strip()
                if s:
                    context_items.append(
                        {
                            "content": s,
                            "source": "",
                            "id": "",
                            "score": "",
                            "node_type": "Chunk",
                        }
                    )
                continue

            # --- THIS is the chunk text to log ---
            chunk_text = str(r.get("text") or "").strip()
            if not chunk_text:
                continue

            docs = r.get("documents", [])
            if not isinstance(docs, list):
                docs = [str(docs)]

            file_name = str(r.get("file_name") or "").strip()

            context_items.append(
                {
                    # Logger reads THIS field as prompt_context_text etc.
                    "content": chunk_text,

                  
                    "node_type": "Chunk",
                    "id": str(r.get("chunk_id") or ""),
                    "score": r.get("score", ""),
                    "entities": r.get("entities", []),
                    "documents": docs,
                    "file_name": file_name,

                
                    "source": ", ".join([str(x) for x in docs if x]),
                }
            )

        return context_items

    # Fallback: object/string
    s = str(results).strip()
    if s:
        context_items.append(
            {
                "content": s,
                "source": "retriever_raw",
                "id": "",
                "score": "",
                "node_type": "Chunk",
            }
        )
    return context_items

# ---------------------------------------------------------------------------
# 5) Answering: GraphRAG answer + retriever context (stable logging)
# ---------------------------------------------------------------------------

def answer_with_rag(question: str, top_k: int = 20) -> Tuple[str, List[Dict[str, Any]]]:
    safe_q = lucene_escape(question)

    # GraphRAG generation
    response = rag.search(
        query_text=safe_q,
        retriever_config={"top_k": top_k},
    )

    answer = (getattr(response, "answer", None) or "").strip()

    # Context from retriever directly 
    context_items = retrieve_context_items(safe_q, top_k=top_k)
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
    print("LLMGraphTransformer_HybridCypherRetriever")
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
