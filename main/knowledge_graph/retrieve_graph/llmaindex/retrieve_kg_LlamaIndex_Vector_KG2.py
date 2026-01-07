from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import json
from typing import Any, Dict, List, Optional, Tuple
import os

from neo4j import GraphDatabase

from llama_index.graph_stores.neo4j import Neo4jPGStore
from llama_index.core import PropertyGraphIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.indices.property_graph import VectorContextRetriever

from main.evaluation.logger import log_antwort

load_dotenv(find_dotenv())

# ---------------------------------------------------------------------------
# 1) Config
# ---------------------------------------------------------------------------

embed_model = OpenAIEmbedding(model="text-embedding-3-small")
llm = OpenAI(model=os.getenv("OPENAI_MODEL"), temperature=0)

username = os.getenv("NEO4J_USER")
password = os.getenv("NEO4J_PASSWORD")
uri = os.getenv("NEO4J_URI")
database = "llmakg"

SCRIPT_NAME = "LlamaIndex_Dense_GraphRAG_Traversal"

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT")).expanduser().resolve()
QUESTIONS_PATH = (
    PROJECT_ROOT
    / "main"
    / "evaluation"
    / "evaluation_datasets"
    / "golden_answers_dataset_filtered.jsonl"
)

# ---------------------------------------------------------------------------
# 2) GraphStore & Index (LlamaIndex)
# ---------------------------------------------------------------------------

graph_store = Neo4jPGStore(
    username=username,
    password=password,
    url=uri,
    database=database,
)

index = PropertyGraphIndex.from_existing(
    property_graph_store=graph_store,
    llm=llm,
    embed_model=embed_model,
)

# ---------------------------------------------------------------------------
# 3) Dense seed retriever (only seeds; GraphRAG expansion happens via Neo4j Cypher)
# ---------------------------------------------------------------------------

seed_retriever = VectorContextRetriever(
    graph_store=index.property_graph_store,
    embed_model=embed_model,
    similarity_top_k=5,  # seeds k
)

# ---------------------------------------------------------------------------
# 4) Neo4j driver (for explicit graph traversal / GraphRAG expansion)
# ---------------------------------------------------------------------------

driver = GraphDatabase.driver(uri, auth=(username, password))
driver.verify_connectivity()

# ---------------------------------------------------------------------------
# 5) Logging helper
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
# 6) Helpers: extract id/text from LlamaIndex results
# ---------------------------------------------------------------------------

def _extract_chunk_text(result_obj: Any) -> str:
    node = getattr(result_obj, "node", None)
    if node is not None:
        txt = getattr(node, "text", None)
        if isinstance(txt, str) and txt.strip():
            return txt.strip()
        try:
            c = node.get_content(metadata_mode="none")
            if isinstance(c, str) and c.strip():
                return c.strip()
        except Exception:
            pass
        try:
            c = node.get_content()
            if isinstance(c, str) and c.strip():
                return c.strip()
        except Exception:
            pass

    try:
        c = result_obj.get_content(metadata_mode="none")
        if isinstance(c, str) and c.strip():
            return c.strip()
    except Exception:
        pass

    try:
        c = result_obj.get_content()
        if isinstance(c, str) and c.strip():
            return c.strip()
    except Exception:
        pass

    return str(result_obj or "").strip()

def _extract_node_id(result_obj: Any) -> str:
    # tries common attributes used by LlamaIndex NodeWithScore/TextNode
    for attr in ("id_", "id", "node_id", "ref_doc_id"):
        v = getattr(result_obj, attr, None)
        if v is not None and str(v).strip():
            return str(v).strip()

    node = getattr(result_obj, "node", None)
    if node is not None:
        for attr in ("id_", "id", "node_id", "ref_doc_id"):
            v = getattr(node, attr, None)
            if v is not None and str(v).strip():
                return str(v).strip()

    # also check metadata
    meta = getattr(result_obj, "metadata", None) or getattr(getattr(result_obj, "node", None), "metadata", None) or {}
    if isinstance(meta, dict):
        for k in ("neo4j_id", "elementId", "element_id", "chunk_id"):
            if meta.get(k):
                return str(meta[k]).strip()

    return ""

# ---------------------------------------------------------------------------
# 7) GraphRAG expansion via Cypher: Chunk -> Entities -> Related Chunks (+ relations)
# ---------------------------------------------------------------------------

GRAPH_EXPANSION_QUERY = """
MATCH (c:Chunk)
WHERE c.chunk_id = $chunk_id
WITH c

// (1) Direct entities mentioned in the seed chunk
OPTIONAL MATCH (c)-[:MENTIONS]-(e1:__Entity__)
WITH c, collect(DISTINCT e1)[0..20] AS e1s

// (2) Related chunks via shared entities
CALL (c, e1s) {
  WITH c, e1s
  UNWIND e1s AS e
  MATCH (e)-[:MENTIONS]-(c2:Chunk)
  WHERE c2 <> c
  WITH c2, count(DISTINCT e) AS evidence
  ORDER BY evidence DESC
  LIMIT 10
  RETURN collect(DISTINCT c2) AS related_chunks
}

// (3) Related entities (1-hop neighbors)
CALL (e1s) {
  WITH e1s
  UNWIND e1s AS e
  MATCH (e)--(e2:__Entity__)
  RETURN collect(DISTINCT e2)[0..30] AS rel_ents
}

// (4) Entity--entity relations within 1..2 hops, excluding :MENTIONS and Chunk nodes
CALL (e1s) {
  WITH e1s
  UNWIND e1s AS e
  MATCH p = (e)-[rs*1..2]-(nb:__Entity__)
  WHERE ALL(r IN rs WHERE type(r) <> 'MENTIONS')
    AND ALL(n IN nodes(p) WHERE n:__Entity__)
  UNWIND relationships(p) AS rel
  RETURN collect(DISTINCT rel) AS rels
}

WITH c, e1s, related_chunks, rel_ents, rels, ([c] + related_chunks) AS chunks

RETURN
  coalesce(c.text,'') AS seed_text,
  [e IN e1s | {name: coalesce(e.name, e.id, ''), labels: labels(e)}] AS direct_entities,
  [x IN related_chunks | coalesce(x.text,'')] AS related_chunk_texts,
  [e IN rel_ents | {name: coalesce(e.name, e.id, ''), labels: labels(e)}] AS related_entities,
  apoc.text.join([x IN chunks | coalesce(x.text,'')], '\n') +
  '\n' +
  apoc.text.join(
    [r IN rels |
      coalesce(startNode(r).name, startNode(r).id, '?') + ' - ' +
      type(r) + ' ' +
      apoc.convert.toJson(properties(r)) + ' -> ' +
      coalesce(endNode(r).name, endNode(r).id, '?')
    ],
    '\n'
  ) AS context_text

"""

def build_graphrag_context_for_seed(chunk_id: str) -> Tuple[str, Dict[str, Any]]:
    """
    Returns:
      (context_text, meta_dict)
    where context_text is the GraphRAG-style aggregated context for one seed.
    """
    with driver.session(database=database) as session:
        rec = session.run(GRAPH_EXPANSION_QUERY, chunk_id=chunk_id).single()

    if not rec:
        # Fallback if id mapping fails; keep it explicit for debugging
        return (
            "[GRAPH EXPANSION FAILED: seed chunk not matched in Neo4j for this id]",
            {"seed_id": chunk_id, "direct_entities": [], "related_chunk_texts": [], "rendered_relations": []},
        )

    seed_text = (rec.get("seed_text") or "").strip()
    direct_entities = rec.get("direct_entities") or []
    related_chunk_texts = [t for t in (rec.get("related_chunk_texts") or []) if t and t.strip()]
    rendered_relations = [r for r in (rec.get("rendered_relations") or []) if r and str(r).strip()]

    # Construct a single context string (similar to your Neo4j GraphRAG "info")
    parts: List[str] = []
    if seed_text:
        parts.append(seed_text)

    if related_chunk_texts:
        parts.append("\n".join(related_chunk_texts))

    if rendered_relations:
        parts.append("\n".join(rendered_relations))

    context_text = "\n\n".join([p for p in parts if p.strip()]) or "[EMPTY CONTEXT AFTER GRAPH EXPANSION]"

    meta = {
        "seed_id": chunk_id,
        "direct_entities": direct_entities,
        "related_chunk_texts_count": len(related_chunk_texts),
        "relations_count": len(rendered_relations),
    }
    return context_text, meta

# ---------------------------------------------------------------------------
# 8) Answer with GraphRAG-style context (seeds + traversal expansion)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a technical support assistant for Arduino Products.\n"
    "Use ONLY the provided context. Do not use outside knowledge.\n"
    "If the context does not contain the answer, say exactly what information is missing.\n"
    "Answer in complete sentences.\n"
    "Answer as completely as possible.\n"
)

def answer_with_graphrag_traversal(question: str, top_k_seeds: int = 5) -> Tuple[str, List[Dict[str, Any]]]:
    # 1) Dense seed retrieval (LlamaIndex)
    seed_retriever.similarity_top_k = top_k_seeds
    seed_results = seed_retriever.retrieve(question)

    context_items: List[Dict[str, Any]] = []
    all_context_blocks: List[str] = []

    for r in (seed_results or []):
        seed_id = _extract_node_id(r)
        seed_text = _extract_chunk_text(r)

        # If we cannot map the id, still log the seed text
        if not seed_id:
            context_items.append({
                "content": seed_text or "[SEED WITHOUT ID]",
                "source": "llamaindex_seed_dense",
                "id": "",
                "score": getattr(r, "score", ""),
                "node_type": "seed_chunk",
            })
            all_context_blocks.append(seed_text or "")
            continue

        # 2) Graph traversal expansion (Neo4j Cypher)
        ctx, meta = build_graphrag_context_for_seed(seed_id)

        context_items.append({
            "content": ctx,
            "source": "neo4j_graphrag_traversal",
            "id": str(seed_id),
            "score": getattr(r, "score", ""),
            "node_type": "graphrag_context",
            "meta": meta,
        })
        all_context_blocks.append(ctx)

    if not all_context_blocks:
        all_context_blocks = ["[NO CONTEXT RETURNED]"]
        context_items = [{
            "content": "[NO CONTEXT RETURNED]",
            "source": "system",
            "id": "",
            "score": "",
            "node_type": "system",
        }]

    final_context = "\n\n---\n\n".join([b for b in all_context_blocks if b.strip()])

    prompt = f"""{SYSTEM_PROMPT}

Context:
{final_context}

Question: {question}

Final answer:
"""
    answer = llm.complete(prompt).text.strip()
    return answer, context_items

# ---------------------------------------------------------------------------
# 9) Batch & manual mode
# ---------------------------------------------------------------------------

def run_batch_from_file(top_k_seeds: int = 5):
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

            question_id = obj.get("id") or obj.get("question_id") or obj.get("query_id") or ""
            question = obj.get("question") or ""
            gold_answer = obj.get("gold_answer") or ""
            query_type = obj.get("query_type") or ""

            if not question:
                continue

            print(f"[QID {question_id}] [{query_type}] {question}")

            answer, context_items = answer_with_graphrag_traversal(question, top_k_seeds=top_k_seeds)
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

def manual_question(top_k_seeds: int = 5):
    qid = input("Question ID (optional): ").strip() or ""
    qtype = input("Query type (optional): ").strip() or "manual"
    question = input("Question: ").strip()
    gold_answer = input("Gold Answer (optional): ").strip() or ""

    if not question:
        print("Empty question, skipping.\n")
        return

    answer, context_items = answer_with_graphrag_traversal(question, top_k_seeds=top_k_seeds)

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

def main_loop(top_k_seeds: int = 5):
    print("LlamaIndex seeds + Neo4j traversal GraphRAG")
    print("Type 'exit' to quit.\n")

    while True:
        mode = input("Manual question? (y/n, or 'exit'): ").strip().lower()

        if mode in ("exit", "quit", "q"):
            break
        elif mode in ("y", "yes"):
            manual_question(top_k_seeds=top_k_seeds)
        elif mode in ("n", "no"):
            run_batch_from_file(top_k_seeds=top_k_seeds)
        else:
            print("Please enter 'y', 'n', or 'exit'.\n")

if __name__ == "__main__":
    try:
        main_loop(top_k_seeds=5)
    finally:
        driver.close()
