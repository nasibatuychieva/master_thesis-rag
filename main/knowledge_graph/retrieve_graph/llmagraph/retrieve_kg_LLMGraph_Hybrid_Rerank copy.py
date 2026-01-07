from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import re

from dotenv import load_dotenv, find_dotenv
from neo4j import GraphDatabase

from neo4j_graphrag.generation import RagTemplate, GraphRAG
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.retrievers import HybridCypherRetriever
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers.base import Retriever

from sentence_transformers import CrossEncoder

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

SCRIPT_NAME = "LLMGraph_Hybrid_Reranker_2Stage"

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT")).expanduser().resolve()
QUESTIONS_PATH = (
    PROJECT_ROOT
    / "main"
    / "evaluation"
    / "evaluation_datasets"
    / "golden_answers_dataset_llmgraph.jsonl"
)

# Neo4j driver
driver = GraphDatabase.driver(URI, auth=(AUTH_USER, AUTH_PASSWORD))
driver.verify_connectivity()

# LLM for GraphRAG answer generation
llm = OpenAILLM(model_name=os.getenv("OPENAI_MODEL"))

# Embedder for vector search
embedder = OpenAIEmbeddings(model="text-embedding-3-small")

# 2-stage settings (IMPORTANT)
STAGE1_TOP_K = 20          # candidates for reranking (before final selection)
FINAL_TOP_K = 5            # final items to feed to LLM
MAX_CHARS_STAGE1 = 1200    # clip context_text for reranking safety
MAX_CHARS_STAGE2 = 9000    # clip final context_text that is sent to LLM (safety net)

# Reranker multiplier (less important now, keep moderate)
RERANK_MULTIPLIER = 3

# ---------------------------------------------------------------------------
# 2) Retrieval query (as-is; two-stage will clip & re-fetch full later)
# ---------------------------------------------------------------------------
retrieval_query = """
WITH node, score
WHERE node:Chunk
  AND node.text IS NOT NULL
  AND trim(node.text) <> ""

OPTIONAL MATCH (node)-[:MENTIONS]-(e1:Entity)
WITH node, score, collect(DISTINCT e1)[0..20] AS e1s

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

CALL {
  WITH e1s
  UNWIND e1s AS e
  MATCH (e)--(e2:Entity)
  WHERE e2 <> e
  RETURN collect(DISTINCT e2)[0..10] AS rel_ents
}

CALL {
  WITH e1s
  UNWIND e1s AS e
  MATCH p = (e)-[rs*1..1]-(nb)
  WHERE nb:Entity
    AND ALL(r IN rs WHERE type(r) <> 'MENTIONS')
    AND ALL(n IN nodes(p) WHERE n:Entity)
  UNWIND relationships(p) AS rel
  RETURN collect(DISTINCT rel) AS rels
}

WITH node, score, e1s, rel_ents, related_chunks, rels,
     ([node] + related_chunks) AS chunks

WITH node, score, e1s, rel_ents, related_chunks, rels, chunks,
     reduce(txt = "", c IN chunks |
       txt + CASE WHEN txt = "" THEN "" ELSE "\n" END + coalesce(c.text, "")
     ) AS chunk_text

WITH node, score, e1s, rel_ents, related_chunks, rels, chunk_text,
     reduce(et = "", e IN e1s |
       et + CASE WHEN et = "" THEN "" ELSE "\n" END +
       ("- " + coalesce(e.id, e.description, elementId(e), "?")
        + " [" + coalesce(e.entityType, "") + "]")
     ) AS direct_ent_text,
     reduce(rt = "", e IN rel_ents |
       rt + CASE WHEN rt = "" THEN "" ELSE "\n" END +
       ("- " + coalesce(e.id, e.description, elementId(e), "?")
        + " [" + coalesce(e.entityType, "") + "]")
     ) AS related_ent_text

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

RETURN
  score AS score,

  // IMPORTANT: we return chunk_id so stage-2 can fetch full contexts by id
  coalesce(node.chunk_id, node.id, elementId(node)) AS chunk_id,

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
    + "\n\n=== RELATIONS ===\n" + coalesce(rel_text, "")
  ) AS context_text
"""

# Lucene escaping for fulltext queries
LUCENE_SPECIAL = r'(\+|\-|\&\&|\|\||\!|\(|\)|\{|\}|\[|\]|\^|"|~|\*|\?|\:|\\|\/)'

def lucene_escape(s: str) -> str:
    s = re.sub(r"\s+", " ", s.strip())
    s = re.sub(LUCENE_SPECIAL, r"\\\1", s)
    return s


# ---------------------------------------------------------------------------
# 3) Base retriever (Neo4j Hybrid)
# ---------------------------------------------------------------------------
base_retriever = HybridCypherRetriever(
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

reranker = CrossEncoder("BAAI/bge-reranker-base")


# ---------------------------------------------------------------------------
# 4) Two-stage: rerank on CLIPPED contexts, then fetch FULL contexts for winners
# ---------------------------------------------------------------------------
def _clip_text(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    return s[:max_chars].rstrip() + "\n...[CLIPPED]..."

def _call_retriever_any(r: Any, query_text: str, top_k: int, **kwargs):
    # HybridCypherRetriever API differs by version
    if hasattr(r, "search"):
        try:
            return r.search(query_text=query_text, top_k=top_k, **kwargs)
        except TypeError:
            return r.search(query_text, top_k=top_k, **kwargs)
    if hasattr(r, "retrieve"):
        try:
            return r.retrieve(query_text=query_text, top_k=top_k, **kwargs)
        except TypeError:
            return r.retrieve(query_text, top_k=top_k, **kwargs)
    raise RuntimeError("Retriever has no search/retrieve method.")

def fetch_full_context_by_chunk_ids(chunk_ids: List[str]) -> Dict[str, str]:
    """
    Stage-2: Fetch the FULL context_text for each winner chunk_id via a direct Cypher query.
    We do NOT rely on vector search here—only exact chunk_id lookup.
    """
    if not chunk_ids:
        return {}

    cy = """
    MATCH (node:Chunk)
    WHERE coalesce(node.chunk_id, node.id, elementId(node)) IN $ids
      AND node.text IS NOT NULL AND trim(node.text) <> ""

    OPTIONAL MATCH (node)-[:MENTIONS]-(e1:Entity)
    WITH node, collect(DISTINCT e1)[0..20] AS e1s

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

    CALL {
      WITH e1s
      UNWIND e1s AS e
      MATCH p = (e)-[rs*1..1]-(nb)
      WHERE nb:Entity
        AND ALL(r IN rs WHERE type(r) <> 'MENTIONS')
        AND ALL(n IN nodes(p) WHERE n:Entity)
      UNWIND relationships(p) AS rel
      RETURN collect(DISTINCT rel) AS rels
    }

    WITH node, related_chunks, rels, ([node] + related_chunks) AS chunks

    WITH node,
         reduce(txt = "", c IN chunks |
           txt + CASE WHEN txt = "" THEN "" ELSE "\n" END + coalesce(c.text, "")
         ) AS chunk_text,
         [r IN rels |
           coalesce(startNode(r).id, startNode(r).description, elementId(startNode(r)), "?")
           + " - " + type(r) + " "
           + " -> " + coalesce(endNode(r).id, endNode(r).description, elementId(endNode(r)), "?")
         ] AS rel_lines

    WITH node, chunk_text,
         reduce(reltext = "", line IN rel_lines |
           reltext + CASE WHEN reltext = "" THEN "" ELSE "\n" END + line
         ) AS rel_text

    RETURN
      coalesce(node.chunk_id, node.id, elementId(node)) AS chunk_id,
      (
        "=== CHUNKS ===\n" + chunk_text
        + "\n\n=== RELATIONS ===\n" + coalesce(rel_text, "")
      ) AS context_text
    """

    out: Dict[str, str] = {}
    with driver.session(database=DATABASE) as s:
        rows = s.run(cy, ids=chunk_ids).data()
    for r in rows:
        cid = str(r.get("chunk_id") or "").strip()
        ctx = str(r.get("context_text") or "").strip()
        if cid and ctx:
            out[cid] = ctx
    return out


class TwoStageRerankingRetriever(Retriever):
    """
    Stage 1:
      - get many candidates (top_k * multiplier)
      - rerank using CLIPPED context_text (safe for CrossEncoder & speed)
    Stage 2:
      - pick winners (top_k)
      - fetch FULL context_text for winners by chunk_id
      - return winners with full contexts (optionally clipped by MAX_CHARS_STAGE2 safety net)
    """

    def __init__(self, base, reranker_model, multiplier: int = 3):
        super().__init__(driver=base.driver, neo4j_database=base.neo4j_database)
        self.base = base
        self.reranker = reranker_model
        self.multiplier = multiplier

    def _rerank_stage1(self, raw_query: str, items: List[Dict[str, Any]], keep_k: int) -> List[Dict[str, Any]]:
        candidates = [r for r in items if isinstance(r, dict) and (r.get("context_text") or r.get("text"))]
        if not candidates:
            return items[:keep_k] if isinstance(items, list) else items

        # IMPORTANT: rerank on clipped text
        pairs = [
            [raw_query, _clip_text(str(r.get("context_text") or r.get("text") or ""), MAX_CHARS_STAGE1)]
            for r in candidates
        ]
        scores = self.reranker.predict(pairs)

        for r, s in zip(candidates, scores):
            r["rerank_score"] = float(s)

        candidates.sort(key=lambda r: r.get("rerank_score", 0.0), reverse=True)
        return candidates[:keep_k]

    def search(self, query_text: str, top_k: int = FINAL_TOP_K, **kwargs):
        raw_query = kwargs.pop("raw_query", query_text)

        # Stage-1 fetch: many candidates
        k1 = max(top_k * self.multiplier, top_k)
        base_res = _call_retriever_any(self.base, query_text=query_text, top_k=k1, **kwargs)

        if not isinstance(base_res, list):
            items = getattr(base_res, "items", None)
            if isinstance(items, list):
                base_res = items
            else:
                return base_res

        # Stage-1 rerank (clipped)
        winners = self._rerank_stage1(raw_query, base_res, keep_k=top_k)

        # Stage-2: fetch FULL contexts for winners by chunk_id
        winner_ids = [str(r.get("chunk_id") or "").strip() for r in winners if isinstance(r, dict)]
        winner_ids = [x for x in winner_ids if x]
        full_ctx = fetch_full_context_by_chunk_ids(winner_ids)

        # Attach full contexts + safety-clip (only as last resort to avoid crashes)
        for r in winners:
            if not isinstance(r, dict):
                continue
            cid = str(r.get("chunk_id") or "").strip()
            if cid and cid in full_ctx:
                r["context_text"] = full_ctx[cid]
            # final safety clip (should rarely trigger if your graph isn't enormous)
            r["context_text"] = _clip_text(str(r.get("context_text") or ""), MAX_CHARS_STAGE2)

        return winners

    def retrieve(self, query_text: str, top_k: int = FINAL_TOP_K, **kwargs):
        # keep API compatibility
        return self.search(query_text=query_text, top_k=top_k, **kwargs)


# Use 2-stage retriever
retriever = TwoStageRerankingRetriever(base_retriever, reranker, multiplier=RERANK_MULTIPLIER)

rag = GraphRAG(retriever=retriever, llm=llm, prompt_template=prompt_template)


# ---------------------------------------------------------------------------
# 5) Logging
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

def retrieve_context_items(question: str, top_k: int) -> List[Dict[str, Any]]:
    # directly call the 2-stage retriever so logging matches what LLM saw
    results = retriever.search(query_text=question, top_k=top_k)

    context_items: List[Dict[str, Any]] = []
    if isinstance(results, list):
        for r in results:
            if not isinstance(r, dict):
                s = str(r).strip()
                if s:
                    context_items.append({"content": s, "source": "", "id": "", "score": "", "node_type": "Chunk"})
                continue

            ctx = str(r.get("context_text") or "").strip()
            if not ctx:
                continue

            context_items.append(
                {
                    "content": ctx,
                    "node_type": "Chunk",
                    "id": str(r.get("chunk_id") or ""),
                    "score": r.get("score", ""),
                    "source": "",
                }
            )
        return context_items

    s = str(results).strip()
    if s:
        context_items.append({"content": s, "source": "retriever_raw", "id": "", "score": "", "node_type": "Chunk"})
    return context_items


# ---------------------------------------------------------------------------
# 6) Answering
# ---------------------------------------------------------------------------
def answer_with_rag(question: str, top_k: int = FINAL_TOP_K) -> Tuple[str, List[Dict[str, Any]]]:
    safe_q = lucene_escape(question)

    response = rag.search(
        query_text=safe_q,
        retriever_config={"top_k": top_k},
    )

    answer = (getattr(response, "answer", None) or "").strip()

    context_items = retrieve_context_items(safe_q, top_k=top_k)
    return answer, context_items


# ---------------------------------------------------------------------------
# 7) Batch / Manual
# ---------------------------------------------------------------------------
def run_batch_from_file(top_k: int = FINAL_TOP_K):
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
            query_type = obj.get("query_type") or ""
            question = obj.get("question") or ""
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

def manual_question(top_k: int = FINAL_TOP_K):
    qid = input("Question ID (optional): ").strip()
    qtype = input("Query type (optional): ").strip()
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

def main_loop(top_k: int = FINAL_TOP_K):
    print("LLMGraphTransformer_HybridCypherRetriever (2-stage rerank -> full fetch)")
    print("Type 'exit' to quit.\n")
    print(f"Stage1 candidates={STAGE1_TOP_K} (approx via multiplier={RERANK_MULTIPLIER}), Final top_k={FINAL_TOP_K}")
    print(f"Clip stage1={MAX_CHARS_STAGE1} chars, clip stage2 safety={MAX_CHARS_STAGE2} chars\n")

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
        main_loop(top_k=FINAL_TOP_K)
    finally:
        driver.close()
