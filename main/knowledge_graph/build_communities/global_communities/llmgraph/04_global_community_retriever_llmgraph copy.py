from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv, find_dotenv
from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from main.evaluation.logger import log_antwort


# ---------------------------------------------------------------------------
# 0) ENV & CONFIG
# ---------------------------------------------------------------------------
load_dotenv(find_dotenv())

URI = os.getenv("NEO4J_URI")
AUTH_USER = os.getenv("NEO4J_USER")
AUTH_PASSWORD = os.getenv("NEO4J_PASSWORD")


DATABASE = os.getenv("NEO4J_DATABASE", "llmagraphtrkg")

SCRIPT_NAME = "LLMGraph_Community_KG_Retriever"

# Dataset path (keep your explicit path if you want)
QUESTIONS_PATH = Path(
    os.getenv(
        "QUESTIONS_PATH",
        r"C:\Users\Nasiba\Documents\1 Master Data Science\Master Thesis\VS Code New\master_thesis-rag\main\evaluation\graphrag\golden_answers_dataset_new.jsonl",
    )
)

MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LEVEL = int(os.getenv("COMMUNITY_LEVEL", "1"))

# IMPORTANT: batch = nur Portionierung, NICHT droppen
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))

# Hard limits (Token-Schutz), ohne Communities wegzulassen
MAX_CHARS_PER_COMMUNITY = int(os.getenv("MAX_CHARS_PER_COMMUNITY", "6000"))
MAX_CHARS_PER_BATCH = int(os.getenv("MAX_CHARS_PER_BATCH", "60000"))

# optional resume (falls es mitten drin crasht)
START_BATCH = int(os.getenv("START_BATCH", "0"))

# logging: avoid gigantic CSV rows
MAX_CHUNKS_PER_COMMUNITY = int(os.getenv("MAX_CHUNKS_PER_COMMUNITY", "10"))
MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "2500"))

driver = GraphDatabase.driver(URI, auth=(AUTH_USER, AUTH_PASSWORD))
driver.verify_connectivity()

llm = ChatOpenAI(model=MODEL, temperature=0)


# ---------------------------------------------------------------------------
# 1) Cypher: fetch community summaries for a given level (USED FOR ANSWERING)
# ---------------------------------------------------------------------------
QUERY_COMMUNITIES = """
MATCH (c:__Community__)
WHERE c.level = $level
  AND c.full_content IS NOT NULL
  AND c.full_content <> ""
RETURN c.communityId AS cid, c.full_content AS txt
ORDER BY cid
"""


# ---------------------------------------------------------------------------
# 1b) Cypher: fetch Chunk texts for a set of communities (USED FOR LOGGING)
#    Keep your robust "try relations" logic for llmagraphtrkg
# ---------------------------------------------------------------------------
QUERY_CHUNKS_FOR_COMMUNITIES = """
UNWIND $cids AS cid
MATCH (c:__Community__ {communityId: cid})
WHERE c.level = $level

CALL {
  WITH c
  // --- direct community -> chunk ---
  OPTIONAL MATCH (c)-[
    :HAS_CHUNK|:CONTAINS_CHUNK|:CONTAINS|:HAS_MEMBER|:MEMBER_OF|:IN_COMMUNITY|:PART_OF
  ]-(ch:Chunk)
  RETURN ch

  UNION

  WITH c
  // --- indirect via entities (community -> entity -> chunk) ---
  OPTIONAL MATCH (c)-[
    :HAS_ENTITY|:CONTAINS_ENTITY|:MENTIONS|:HAS_TERM|:HAS_CONCEPT
  ]-(e)
  OPTIONAL MATCH (e)-[
    :MENTIONED_IN|:IN_CHUNK|:APPEARS_IN|:HAS_EVIDENCE|:EVIDENCE_IN
  ]-(ch:Chunk)
  RETURN ch
}

WITH cid, collect(DISTINCT ch) AS chunks
WITH cid, [x IN chunks WHERE x IS NOT NULL][0..$max_chunks] AS chunks_limited
UNWIND chunks_limited AS ch
RETURN
  cid AS communityId,
  elementId(ch) AS chunk_eid,
  coalesce(ch.pk, ch.id, ch.chunk_id, ch.uuid, "") AS chunk_pk,
  coalesce(ch.text, ch.content, ch.chunk, ch.body, "") AS chunk_text
ORDER BY communityId
"""


# ---------------------------------------------------------------------------
# 2) Prompts (batch answering + final synthesis)
# ---------------------------------------------------------------------------
BATCH_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You answer the question using ONLY the provided community summaries. "
     "Synthesize across summaries in THIS BATCH. Do not invent facts. "
     "If this batch does not contain enough information, say what is missing."),
    ("user",
     "Question:\n{question}\n\n"
     "Community summaries (batch):\n{summaries}\n\n"
     "Write a grounded partial answer for this batch.")
])

SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Merge the partial answers into one coherent final answer. "
     "Remove duplicates, keep it structured, and stay faithful to the partial answers. "
     "Do not introduce new facts not present in the partial answers."),
    ("user",
     "Question:\n{question}\n\n"
     "Partial answers:\n{partials}\n\n"
     "Return a single final answer.")
])


# ---------------------------------------------------------------------------
# 3) Logging helper
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
# 4) Retrieval: communities (NO dropping)
# ---------------------------------------------------------------------------
def load_community_summaries(level: int) -> List[Dict[str, Any]]:
    with driver.session(database=DATABASE) as session:
        return session.run(QUERY_COMMUNITIES, level=level).data()


def _truncate(s: str, max_chars: int) -> str:
    s = (s or "")
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3] + "..."


# ---------------------------------------------------------------------------
# 5) Logging context retrieval: chunks for used communities
# ---------------------------------------------------------------------------
def load_chunks_for_communities(
    cids: List[Any],
    level: int,
    max_chunks_per_community: int = MAX_CHUNKS_PER_COMMUNITY,
) -> List[Dict[str, Any]]:
    if not cids:
        return []
    with driver.session(database=DATABASE) as session:
        rows = session.run(
            QUERY_CHUNKS_FOR_COMMUNITIES,
            cids=cids,
            level=level,
            max_chunks=max_chunks_per_community,
        ).data()
    return rows


def build_context_items_from_chunks(
    batch_rows: List[Dict[str, Any]],
    level: int,
) -> List[Dict[str, Any]]:
    """
    LOG ONLY chunk text for communities used in this batch.
    """
    cids = [br.get("cid") for br in batch_rows if br.get("cid") is not None]
    chunk_rows = load_chunks_for_communities(
        cids=cids,
        level=level,
        max_chunks_per_community=MAX_CHUNKS_PER_COMMUNITY,
    )

    items: List[Dict[str, Any]] = []
    for r in chunk_rows:
        community_id = r.get("communityId")
        chunk_text = _truncate(str(r.get("chunk_text", "") or "").strip(), MAX_CHUNK_CHARS)
        if not chunk_text:
            continue

        chunk_eid = str(r.get("chunk_eid") or "")
        chunk_pk = str(r.get("chunk_pk") or "")

        items.append({
            "content": chunk_text,
            "node_type": "Chunk",
            "source": f"Chunk linked to __Community__ level={level} communityId={community_id}",
            "id": chunk_eid or chunk_pk or f"chunk@community={community_id}",
            "score": "",
            "communityId": community_id,
            "level": level,
            "chunk_eid": chunk_eid,
            "chunk_pk": chunk_pk,
        })

    return items


# ---------------------------------------------------------------------------
# 6) Answer generation: process ALL communities sequentially in batches
#    + token-safe truncation
#    + chunk-only logging (like your Retriever 1)
# ---------------------------------------------------------------------------
def answer_global(
    question: str,
    level: int = LEVEL,
    batch_size: int = BATCH_SIZE,
) -> Tuple[str, List[Dict[str, Any]]]:
    rows = load_community_summaries(level=level)
    if not rows:
        return "No community summaries found for the requested level.", []

    partials: List[str] = []
    all_context_items: List[Dict[str, Any]] = []

    total_batches = (len(rows) + batch_size - 1) // batch_size
    print(f"[INFO] Communities loaded: {len(rows)} | batch_size={batch_size} | batches={total_batches}")

    for b in range(START_BATCH, total_batches):
        i = b * batch_size
        batch_rows = rows[i:i + batch_size]

        # Per-community truncation => prevents single huge community from blowing up tokens
        texts: List[str] = []
        for br in batch_rows:
            txt = br.get("txt", "") or ""
            texts.append(_truncate(txt, MAX_CHARS_PER_COMMUNITY))

        summaries_text = "\n\n---\n\n".join(texts)
        summaries_text = _truncate(summaries_text, MAX_CHARS_PER_BATCH)

        print(f"[INFO] Batch {b+1}/{total_batches} | communities={len(batch_rows)} | chars={len(summaries_text)}")

        msg = llm.invoke(BATCH_PROMPT.format_messages(
            question=question,
            summaries=summaries_text
        ))
        partial = (msg.content or "").strip()
        if partial:
            partials.append(partial)

        # ✅ Logging (chunks linked to communities used in THIS batch)
        all_context_items.extend(build_context_items_from_chunks(batch_rows, level=level))

    if not partials:
        return "No partial answers produced. Possibly empty context.", all_context_items

    if len(partials) == 1:
        final_answer = partials[0]
    else:
        msg = llm.invoke(SYNTHESIS_PROMPT.format_messages(
            question=question,
            partials="\n\n---\n\n".join(partials)
        ))
        final_answer = (msg.content or "").strip()

    return final_answer, all_context_items


# ---------------------------------------------------------------------------
# 7) Batch runner (JSONL)
# ---------------------------------------------------------------------------
def run_batch_from_file():
    print(f"\n[INFO] Loading dataset from {QUESTIONS_PATH}\n")
    if not QUESTIONS_PATH.exists():
        print("[ERROR] Dataset not found:", QUESTIONS_PATH)
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

            try:
                answer, context_items = answer_global(question, level=LEVEL, batch_size=BATCH_SIZE)
            except Exception as e:
                print("[ERROR] answering failed:", e)
                answer = f"ERROR during answering: {e}"
                context_items = []

            print(f"[ANSWER]\n{answer}\n")
            print(f"[CTX] n_context={len(context_items)} (chunks)\n")

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
# 8) Manual mode
# ---------------------------------------------------------------------------
def manual_question():
    qid = input("Question ID (optional): ").strip()
    qtype = input("Query type (optional): ").strip()
    question = input("Question: ").strip()
    gold_answer = input("Gold Answer (optional): ").strip()

    if not question:
        print("Empty question, skipping.\n")
        return

    answer, context_items = answer_global(question, level=LEVEL, batch_size=BATCH_SIZE)

    print("\nAnswer:\n", answer, "\n")
    print(f"[CTX] n_context={len(context_items)} (chunks)\n")

    safe_log(
        SCRIPT_NAME,
        qid or "",
        qtype or "manual",
        question,
        answer,
        gold_answer or "",
        context_items=context_items,
    )


def main_loop():
    print(f"{SCRIPT_NAME} | DB={DATABASE} | Level={LEVEL} | Model={MODEL}")
    print(
        f"BATCH_SIZE={BATCH_SIZE} | MAX_CHARS_PER_COMMUNITY={MAX_CHARS_PER_COMMUNITY} | MAX_CHARS_PER_BATCH={MAX_CHARS_PER_BATCH}"
    )
    print(
        f"LOGGING: MAX_CHUNKS_PER_COMMUNITY={MAX_CHUNKS_PER_COMMUNITY} | MAX_CHUNK_CHARS={MAX_CHUNK_CHARS} | START_BATCH={START_BATCH}"
    )
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


if __name__ == "__main__":
    try:
        main_loop()
    finally:
        driver.close()
