import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import re
from dotenv import load_dotenv, find_dotenv
from neo4j import GraphDatabase
from neo4j_graphrag.generation import RagTemplate
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.retrievers import VectorCypherRetriever, HybridCypherRetriever
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.generation import GraphRAG

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

SCRIPT_NAME = "SimpleKG_Hybrid_Retriever"
from pathlib import Path
import os

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT")).expanduser().resolve()

QUESTIONS_PATH = (
    PROJECT_ROOT
    / "main"
    / "evaluation"
    / "evaluation_datasets"
    / "golden_answers_dataset_filtered.jsonl"
)

# Neo4j-Driver
driver = GraphDatabase.driver(URI, auth=(AUTH_USER, AUTH_PASSWORD))
driver.verify_connectivity()

# LLM (Answer Generation)
llm = OpenAILLM(
    model_name=os.getenv("OPENAI_MODEL"),
)

# Embedder (Vector Search)
embedder = OpenAIEmbeddings(model="text-embedding-3-small")

# ---------------------------------------------------------------------------
# 2) Retrieval query (used by VectorCypherRetriever internally)
# ---------------------------------------------------------------------------
retrieval_query = """
WITH node, score
WHERE node:Chunk
  AND node.text IS NOT NULL
  AND trim(node.text) <> ""

// 1) Direct entities am Seed-Chunk
OPTIONAL MATCH (node)-[:MENTIONS]-(e1:Entity)
WITH node, score, collect(DISTINCT e1)[0..20] AS e1s

// 2) Related chunks: Top N via shared entities
CALL {
  WITH node, e1s
  UNWIND e1s AS e
  MATCH (e)-[:MENTIONS]-(c:Chunk)
  WHERE c <> node
    AND c.text IS NOT NULL
    AND trim(c.text) <> ""
  WITH c, count(DISTINCT e) AS evidence
  ORDER BY evidence DESC
  LIMIT 10
  RETURN collect(DISTINCT c) AS related_chunks
}

// 3) Related entities (Neighbors der direkten Entities)
CALL {
  WITH e1s
  UNWIND e1s AS e
  MATCH (e)--(e2:Entity)
  WHERE e2 <> e
  RETURN collect(DISTINCT e2)[0..30] AS rel_ents
}

// 4) Entity–Entity Beziehungen um direkte Entities (1..2 Hops),
//    OHNE :MENTIONS und OHNE Chunk-Knoten im Pfad
CALL {
  WITH e1s
  UNWIND e1s AS e
  MATCH p = (e)-[rs*1..2]-(nb)
  WHERE nb:Entity
    AND ALL(r IN rs WHERE type(r) <> 'MENTIONS')
    AND ALL(n IN nodes(p) WHERE n:Entity)
  UNWIND relationships(p) AS rel
  RETURN collect(DISTINCT rel) AS rels
}

WITH node, score, e1s, rel_ents, related_chunks, rels,
     ([node] + related_chunks) AS chunks

// -------- Chunk-Text (ohne APOC) --------
WITH node, score, e1s, rel_ents, related_chunks, rels, chunks,
     reduce(txt = "", c IN chunks |
       txt + CASE WHEN txt = "" THEN "" ELSE "\n" END + coalesce(c.text, "")
     ) AS chunk_text

// -------- Entity-Text (direct + related) --------
WITH node, score, e1s, rel_ents, related_chunks, rels, chunk_text,
     // Direct entities als Textzeilen
     reduce(et = "", e IN e1s |
       et + CASE WHEN et = "" THEN "" ELSE "\n" END +
       ("- " + coalesce(e.id, e.description, elementId(e), "?")
        + " [" + coalesce(e.entityType, "") + "]")
     ) AS direct_ent_text,
     // Related entities als Textzeilen
     reduce(rt = "", e IN rel_ents |
       rt + CASE WHEN rt = "" THEN "" ELSE "\n" END +
       ("- " + coalesce(e.id, e.description, elementId(e), "?")
        + " [" + coalesce(e.entityType, "") + "]")
     ) AS related_ent_text

// -------- Relation-Text robust (LIST<STRING> -> String) --------
WITH node, score, e1s, rel_ents, related_chunks, chunk_text, direct_ent_text, related_ent_text,
     [r IN rels |
       coalesce(startNode(r).id, startNode(r).description, elementId(startNode(r)), "?")
       + " - " + type(r) + " " +
       CASE
         WHEN r.details IS NULL AND r.description IS NULL THEN ""
         WHEN r.details IS NOT NULL THEN
           CASE
             WHEN valueType(r.details) STARTS WITH "LIST" THEN
               reduce(s = "", x IN r.details |
                 s + CASE WHEN s = "" THEN "" ELSE "; " END + toString(x)
               )
             ELSE toString(r.details)
           END
         ELSE
           CASE
             WHEN valueType(r.description) STARTS WITH "LIST" THEN
               reduce(s = "", x IN r.description |
                 s + CASE WHEN s = "" THEN "" ELSE "; " END + toString(x)
               )
             ELSE toString(r.description)
           END
       END
       + " -> " +
       coalesce(endNode(r).id, endNode(r).description, elementId(endNode(r)), "?")
     ] AS rel_lines

WITH node, score, e1s, rel_ents, related_chunks,
     chunk_text, direct_ent_text, related_ent_text,
     reduce(reltext = "", line IN rel_lines |
       reltext + CASE WHEN reltext = "" THEN "" ELSE "\n" END + line
     ) AS rel_text

// -------- Final Output --------
RETURN
  score AS score,

  [e IN e1s |
    { id: coalesce(e.id, e.description, elementId(e)),
      entityType: coalesce(e.entityType, "") }
  ] AS direct_entities,

  [c IN related_chunks | c.text][0..10] AS related_chunk_texts,

  [e IN rel_ents |
    { id: coalesce(e.id, e.description, elementId(e)),
      entityType: coalesce(e.entityType, "") }
  ] AS related_entities,

  (
    "=== CHUNKS ===\n" + chunk_text
    + "\n\n=== DIRECT ENTITIES ===\n" + coalesce(direct_ent_text, "")
    + "\n\n=== RELATED ENTITIES ===\n" + coalesce(related_ent_text, "")
    + "\n\n=== RELATIONS ===\n" + coalesce(rel_text, "")
  ) AS context_text

"""
# retrieval_query = """
# WITH node AS node, score

# // 1) Entities direkt am Seed-Chunk (limitiert)
# OPTIONAL MATCH (node)<-[:FROM_CHUNK]-(e1:__Entity__)
# WITH node, score, collect(DISTINCT e1)[0..20] AS e1s

# // 2) Related chunks: Top N pro Seed nach shared-entity-count
# CALL {
#   WITH e1s
#   UNWIND e1s AS e
#   MATCH (e)-[:FROM_CHUNK]->(c:Chunk)
#   WITH c, count(DISTINCT e) AS evidence
#   ORDER BY evidence DESC
#   LIMIT 10
#   RETURN collect(DISTINCT c) AS related_chunks
# }


# CALL {
#   WITH e1s
#   UNWIND e1s AS e
#   MATCH (e)--(e2:__Entity__)
#   RETURN collect(DISTINCT e2)[0..30] AS rel_ents
# }

# // 4) Beziehungen um die direkten Entities (1..2 Hops),
# //    aber OHNE :FROM_CHUNK und OHNE Chunk-Knoten im Pfad
# CALL {
#   WITH e1s
#   UNWIND e1s AS e
#   MATCH p = (e)-[rs*1..2]-(nb)
#   WHERE nb:__Entity__
#     AND ALL(r IN rs WHERE type(r) <> 'FROM_CHUNK')
#     AND ALL(n IN nodes(p) WHERE n:__Entity__)
#   UNWIND relationships(p) AS rel
#   RETURN collect(DISTINCT rel) AS rels
# }

# WITH node, score, e1s, rel_ents, related_chunks, rels,
#      ([node] + related_chunks) AS chunks

# RETURN
#   // node.text AS text,
#   score     AS score,
#   [e IN e1s | {name: coalesce(e.name, e.id, ''), labels: labels(e)}] AS direct_entities,
#   [n IN related_chunks | n.text] AS related_chunk_texts,
#   [e IN rel_ents | {name: coalesce(e.name, e.id, ''), labels: labels(e)}] AS related_entities,

#   // optional: zusätzliches Feld (stört nicht)
#   apoc.text.join([c IN chunks | coalesce(c.text,'')], '\n') +
#   '\n' +
#   apoc.text.join(
#     [r IN rels |
#       coalesce(startNode(r).name, startNode(r).id, '?') + ' - ' +
#       type(r) + ' ' +
#       coalesce(r.details, r.description, '') + ' -> ' +
#       coalesce(endNode(r).name, endNode(r).id, '?')
#     ],
#     '\n'
#   ) AS context_text

# """

LUCENE_SPECIAL = r'(\+|\-|\&\&|\|\||\!|\(|\)|\{|\}|\[|\]|\^|"|~|\*|\?|\:|\\|\/)'

def lucene_escape(s: str) -> str:
    s = re.sub(r"\s+", " ", s.strip())
    s = re.sub(LUCENE_SPECIAL, r"\\\1", s)  
    return s

retriever = HybridCypherRetriever(
    driver,
    neo4j_database=DATABASE,
    vector_index_name="chunkEmbedding_simplekg",
    fulltext_index_name="chunkFulltext_simplekg",
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
# 4) Context retrieval (LlamaIndex-style): call retriever directly
# ---------------------------------------------------------------------------

def retrieve_context_items(question: str,  top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Retrieve context directly from VectorCypherRetriever (not from GraphRAG response).
    This is the reliable way to always have context for faithfulness evaluation.
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
                text = str(r.get("context_text") or r.get("text") or "").strip()
                if not text:
                    continue

                direct_entities = r.get("direct_entities", [])
                related_entities = r.get("related_entities", [])
                related_chunk_texts = r.get("related_chunk_texts", [])
                score = r.get("score", "")

                
    
                def fmt_entities(arr):
                    out = []
                    for e in arr or []:
                        name = e.get("name", "")
                        labels = e.get("labels", [])
                        if name:
                            out.append(f"{name} ({', '.join(labels)})")
                    return out

                meta_lines = []
                if direct_entities:
                    meta_lines.append(f"Direct entities: {fmt_entities(direct_entities)}")
                if related_entities:
                    meta_lines.append(f"Related entities: {fmt_entities(related_entities)}")
                if related_chunk_texts:
                    meta_lines.append(f"Related chunks: {related_chunk_texts}")


                enriched_text = text
                if meta_lines:
                    enriched_text = text + "\n" + "\n".join(meta_lines)

                context_items.append({
    "content": enriched_text,
    "source": "simplekg_vector_index",
    "id": "",
    "score": score,
    "direct_entities": direct_entities,
    "related_entities": related_entities,
    "related_chunk_texts": related_chunk_texts,
})

            else:
                s = str(r).strip()
                if s:
                    context_items.append({"content": s, "source": "simplekg_vector_index", "id": "", "score": ""})

    else:

        s = str(results).strip()
        if s:
            context_items.append({"content": s, "source": "simplekg_retriever_raw", "id": "", "score": ""})

    return context_items

# ---------------------------------------------------------------------------
# 5) Answering: GraphRAG answer + Retriever context for logging
# ---------------------------------------------------------------------------

def answer_with_graphrag(question: str, top_k: int = 3) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Returns (answer, context_items).
    Answer from GraphRAG; context from retriever directly (reliable).
    """
    safe_q = lucene_escape(question)
    response = rag.search(
        query_text=safe_q,
        retriever_config={"top_k": top_k},
        # return_context=True  # optional, but we do NOT depend on it
    )
    answer = (getattr(response, "answer", None) or "").strip()


    context_items = retrieve_context_items(safe_q, top_k=top_k)


    if not context_items:
        context_items = [{
            "content": "[NO CONTEXT RETURNED BY RETRIEVER]",
            "source": "system",
            "id": "",
            "score": "",
        }]

    return answer, context_items

# ---------------------------------------------------------------------------
# 6) Batch mode (JSONL)
# ---------------------------------------------------------------------------

def run_batch_from_file( top_k: int = 3):
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
                context_items=context_items,   #  pass context
            )

    print("\n[INFO] Batch processing completed.\n")

# ---------------------------------------------------------------------------
# 7) Manual mode
# ---------------------------------------------------------------------------

def manual_question( top_k: int = 3):
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

def main_loop( top_k: int = 3):
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
