from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv, find_dotenv
from neo4j import GraphDatabase

from openai import OpenAI

from main.evaluation.logger import log_antwort

load_dotenv(find_dotenv())

# ---------------------------------------------------------------------------
# 1) Config
# ---------------------------------------------------------------------------

URI = os.getenv("NEO4J_URI")
AUTH_USER = os.getenv("NEO4J_USER")
AUTH_PASSWORD = os.getenv("NEO4J_PASSWORD")
DATABASE = "llmakg"

SCRIPT_NAME = "Community_Vector_Retriever_HyDE"

# IMPORTANT: must match your Neo4j index embedding dimension (you saw 1536)
TEXT_EMBEDDING_MODEL = os.getenv("TEXT_EMBEDDING_MODEL", "text-embedding-3-small")

# LLM for HyDE + answering
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")  # adjust if you want

# Neo4j vector index name for communities
COMMUNITY_VEC_INDEX = os.getenv("COMMUNITY_VEC_INDEX", "community_vec")

# retrieval sizes
TOP_K_COMMUNITIES = int(os.getenv("TOP_K_COMMUNITIES", "5"))
MAX_ENTITIES_PER_COMMUNITY = int(os.getenv("MAX_ENTITIES_PER_COMMUNITY", "25"))
MAX_CHUNKS_PER_COMMUNITY = int(os.getenv("MAX_CHUNKS_PER_COMMUNITY", "15"))

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT")).expanduser().resolve()
QUESTIONS_PATH = (
    PROJECT_ROOT
    / "main"
    / "evaluation"
    / "evaluation_datasets"
    / "golden_answers_dataset_filtered_rest.jsonl"
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

driver = GraphDatabase.driver(URI, auth=(AUTH_USER, AUTH_PASSWORD))


# ---------------------------------------------------------------------------
# 2) Prompts
# ---------------------------------------------------------------------------

HYDE_PROMPT = """You generate a hypothetical community-style summary to improve vector search.

User question:
{query}

Write a hypothetical community summary that would likely contain the answer.
Use a similar style/structure as this example community report:

---EXAMPLE COMMUNITY REPORT---
{template}
---END EXAMPLE---

Return ONLY the hypothetical community summary text.
"""

ANSWER_PROMPT = """You are a technical support assistant for Arduino products.
Use ONLY the provided context. Do not use outside knowledge.
If the context does not contain the answer, say exactly what information is missing.
Answer in complete sentences and as completely as possible.
Adapt the structure and style of the answer to the type of the question.

Context:
{context}

Question: {question}

Final answer:
"""


# ---------------------------------------------------------------------------
# 3) Helper: Logging
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
# 4) Neo4j: Fetch template + Community retrieval
# ---------------------------------------------------------------------------

def fetch_random_level1_community_template() -> str:
    cypher = """
    MATCH (c:__Community__)
    WHERE c.level = 1
      AND c.full_content IS NOT NULL
      AND trim(c.full_content) <> ""
    RETURN c.full_content AS template
    ORDER BY rand()
    LIMIT 1
    """
    rows = driver.execute_query(
        cypher,
        database_=DATABASE,
        result_transformer_=lambda r: r.data(),
    )
    if not rows:
        return "### Overview\nThis community report summarizes key components, concepts, and relationships."
    return rows[0]["template"]


def hyde_generation(query: str) -> str:
    template = fetch_random_level1_community_template()
    prompt = HYDE_PROMPT.format(query=query, template=template)

    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[{"role": "user", "content": prompt}],
        reasoning={"effort": "low"},
    )
    return resp.output_text.strip()


def embed_text(text: str) -> List[float]:
    emb = client.embeddings.create(
        model=TEXT_EMBEDDING_MODEL,
        input=text,
    )
    vec = emb.data[0].embedding
    # guard against "dimension mismatch" accidents
    if len(vec) < 100:
        raise ValueError(
            f"Embedding dim looks wrong: len={len(vec)}. "
            f"Your Neo4j community index expects 1536."
        )
    return vec


def community_vector_retrieve(
    embedding: List[float],
    k: int,
) -> List[Dict[str, Any]]:
    """
    Returns rows: communityId, score, community_text, entities[], chunks[]
    Uses your schema:
      (c:__Community__)-[:IN_COMMUNITY]-(e:__Entity__)-[:MENTIONS]-(ch:Chunk)
    """
    cypher = """
    CALL db.index.vector.queryNodes($index_name, $k, $embedding)
    YIELD node AS comm, score
    WHERE comm:__Community__
      AND comm.level = 1

    OPTIONAL MATCH (comm)-[:IN_COMMUNITY]-(e:__Entity__)

    OPTIONAL MATCH (e)-[:MENTIONS]-(ch)
    WHERE ch:Chunk OR ch:__Chunk__

    WITH comm, score,
         collect(DISTINCT e)[0..$max_ents] AS ents,
         collect(DISTINCT ch)[0..$max_chunks] AS chunks

    RETURN
      comm.communityId AS communityId,
      score AS score,
      comm.full_content AS community_text,
      [e IN ents | {
        id: coalesce(e.id, e.pk),
        name: coalesce(e.name, e.label)
      }] AS entities,
      [c IN chunks | {
        id: coalesce(c.id, c.pk),
        text: coalesce(c.text, c.content, c.full_content)
      }] AS chunks
    ORDER BY score DESC
    """

    rows = driver.execute_query(
        cypher,
        database_=DATABASE,
        index_name=COMMUNITY_VEC_INDEX,
        k=k,
        embedding=embedding,
        max_ents=MAX_ENTITIES_PER_COMMUNITY,
        max_chunks=MAX_CHUNKS_PER_COMMUNITY,
        result_transformer_=lambda r: r.data(),
    )
    return rows or []


# ---------------------------------------------------------------------------
# 5) Build context_items (for logger) + context_text (for prompt)
# ---------------------------------------------------------------------------

def build_context_items_from_communities(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str]:
    """
    Produces:
      context_items: list[dict] for log_antwort (typed items)
      context_text: concatenated human-readable context for answering
    """
    context_items: List[Dict[str, Any]] = []
    context_blocks: List[str] = []

    if not rows:
        context_items = [{
            "content": "[NO CONTEXT RETURNED BY COMMUNITY VECTOR RETRIEVER]",
            "source": "system",
            "id": "",
            "score": "",
            "node_type": "system",
        }]
        return context_items, context_items[0]["content"]

    for rank, r in enumerate(rows, start=1):
        community_id = r.get("communityId", "")
        score = r.get("score", "")
        community_text = (r.get("community_text") or "").strip()
        entities = r.get("entities") or []
        chunks = r.get("chunks") or []

        # 1) community as context item
        context_items.append({
            "content": community_text or "[EMPTY community.full_content]",
            "source": "neo4j::__Community__",
            "id": str(community_id),
            "score": score,
            "node_type": "__Community__",
        })

        # 2) entities as context items (short)
        for e in entities:
            e_name = (e.get("name") or "").strip()
            e_id = e.get("id", "")
            if not e_name and not e_id:
                continue
            context_items.append({
                "content": e_name or str(e_id),
                "source": "neo4j::__Entity__",
                "id": str(e_id),
                "score": score,
                "node_type": "__Entity__",
            })

        # 3) chunks as context items
        for c in chunks:
            c_text = (c.get("text") or "").strip()
            c_id = c.get("id", "")
            if not c_text:
                continue
            context_items.append({
                "content": c_text,
                "source": "neo4j::Chunk",
                "id": str(c_id),
                "score": score,
                "node_type": "Chunk",
            })

        # text block for LLM prompt (more readable)
        ent_names = [x.get("name") for x in entities if x.get("name")]
        chunk_lines = "\n".join([f"- {x.get('text','')}" for x in chunks if x.get("text")])

        context_blocks.append(
            f"""[Community #{rank}] communityId={community_id} score={score}
COMMUNITY_FULL_CONTENT:
{community_text}

ENTITIES (sample):
{", ".join(ent_names) if ent_names else "(none)"}

CHUNKS (sample):
{chunk_lines if chunk_lines else "(none)"}
"""
        )

    context_text = "\n\n".join(context_blocks).strip()
    return context_items, context_text


# ---------------------------------------------------------------------------
# 6) Main QA: Community Retriever (HyDE -> vector -> answer) + logging
# ---------------------------------------------------------------------------

def answer_with_community_retriever(question: str) -> Tuple[str, List[Dict[str, Any]]]:
    # 1) HyDE
    hyde_text = hyde_generation(question)

    # 2) embed HyDE text
    embedding = embed_text(hyde_text)

    # 3) vector search communities + expand entities/chunks
    rows = community_vector_retrieve(embedding=embedding, k=TOP_K_COMMUNITIES)

    # 4) build context for logging and prompting
    context_items, context_text = build_context_items_from_communities(rows)

    # 5) answer
    prompt = ANSWER_PROMPT.format(context=context_text, question=question)
    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[{"role": "user", "content": prompt}],
        reasoning={"effort": "low"},
    )
    answer = resp.output_text.strip()

    return answer, context_items


# ---------------------------------------------------------------------------
# 7) Batch Mode (JSONL)
# ---------------------------------------------------------------------------

def run_batch_from_file() -> None:
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
            question = obj.get("question") or ""
            gold_answer = obj.get("gold_answer") or ""
            query_type = obj.get("query_type") or ""

            if not question:
                continue

            print(f"[QID {question_id}] [{query_type}] {question}")

            answer, context_items = answer_with_community_retriever(question)

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
# 8) Manual Mode
# ---------------------------------------------------------------------------

def manual_question() -> None:
    qid = input("Question ID (optional): ").strip() or ""
    qtype = input("Query type (optional): ").strip() or "manual"
    question = input("Question: ").strip()
    gold_answer = input("Gold Answer (optional): ").strip() or ""

    if not question:
        print("Empty question, skipping.\n")
        return

    answer, context_items = answer_with_community_retriever(question)

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
# 9) Main Loop
# ---------------------------------------------------------------------------

def main_loop() -> None:
    print("Community Vector Retriever (HyDE -> community vec search) + Context Logging")
    print("Type 'exit' to quit.\n")

    while True:
        mode = input("Manual question? (y/n, or 'exit'): ").strip().lower()

        if mode in ("exit", "quit", "q"):
            break
        elif mode in ("y", "yes"):
            manual_question()
        elif mode in ("n", "no"):
            run_batch_from_file()
        else:
            print("Please enter 'y', 'n', or 'exit'.\n")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        main_loop()
    finally:
        driver.close()
