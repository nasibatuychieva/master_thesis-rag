from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from main.evaluation.logger import log_antwort

load_dotenv()

# =============================================================================
# Configuration (DB1 schema: simplekg)
# =============================================================================
URI = os.getenv("NEO4J_URI")
AUTH_USER = os.getenv("NEO4J_USER")
AUTH_PASSWORD = os.getenv("NEO4J_PASSWORD")

DATABASE = "simplekg"
SCRIPT_NAME = "SimpleKG_Community_KG_Retriever_Batched"

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", str(Path(__file__).resolve().parents[3]))).expanduser().resolve()
QUESTIONS_PATH = PROJECT_ROOT / "main" / "evaluation" / "graphrag" / "golden_answers_dataset_new.jsonl"

COMMUNITY_LABEL = "__Community__"
LEVEL = int(os.getenv("COMMUNITY_LEVEL", "1"))


BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
MAX_CHARS_PER_COMMUNITY = int(os.getenv("MAX_CHARS_PER_COMMUNITY", "6000"))
MAX_CHARS_PER_BATCH = int(os.getenv("MAX_CHARS_PER_BATCH", "60000"))
START_BATCH = int(os.getenv("START_BATCH", "0"))  # resume if crash


CHUNKS_LOG_LIMIT_TOTAL = int(os.getenv("CHUNKS_LOG_LIMIT_TOTAL", "60"))
CHUNKS_LOG_LIMIT_PER_COMMUNITY = int(os.getenv("CHUNKS_LOG_LIMIT_PER_COMMUNITY", "3"))

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))

# =============================================================================
# Neo4j + LLM
# =============================================================================
driver = GraphDatabase.driver(URI, auth=(AUTH_USER, AUTH_PASSWORD))
driver.verify_connectivity()

llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)

# =============================================================================
# Prompts 
# =============================================================================
BATCH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer the question using ONLY the provided community summaries. "
            "Synthesize across summaries in THIS BATCH. Do not invent facts. "
            "If this batch does not contain enough information, say what is missing.",
        ),
        (
            "user",
            "Question:\n{question}\n\n"
            "Community summaries (batch):\n{summaries}\n\n"
            "Write a grounded partial answer for this batch.",
        ),
    ]
)

SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Merge the partial answers into one coherent final answer. "
            "Remove duplicates, keep it structured, and stay faithful to the partial answers. "
            "Do not introduce new facts not present in the partial answers.",
        ),
        (
            "user",
            "Question:\n{question}\n\n"
            "Partial answers:\n{partials}\n\n"
            "Return a single final answer.",
        ),
    ]
)

# =============================================================================
# Logging helper
# =============================================================================
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

# =============================================================================
# Cypher (schema-safe: avoids UnknownPropertyKeyWarning)
# - We do NOT reference c.community_rank directly (may not exist) to avoid warnings.
# - We return one text field: prefer full_content else summary.
# =============================================================================
QUERY_COMMUNITIES = f"""
MATCH (c:{COMMUNITY_LABEL})
WHERE c.level = $level
  AND (
    (c.full_content IS NOT NULL AND c.full_content <> "")
    OR (c.summary IS NOT NULL AND c.summary <> "")
  )
WITH
  c,
  coalesce(toString(c.communityId), toString(id(c))) AS cid,
  CASE
    WHEN c.full_content IS NOT NULL AND c.full_content <> "" THEN c.full_content
    ELSE coalesce(c.summary, "")
  END AS txt
RETURN cid, txt
ORDER BY cid
"""

# =============================================================================
# Chunk fetch for logging (schema-robust)

# =============================================================================
CHUNKS_FOR_COMMUNITY_QUERY = f"""
MATCH (c:{COMMUNITY_LABEL})
WHERE c.level = $level
  AND toString(c.communityId) = $communityId
MATCH (c)--(ch:Chunk)
WITH DISTINCT ch
RETURN
  coalesce(ch.text, ch.chunk_text, ch.content, ch.full_content, "") AS chunk_text
LIMIT $limit
"""


def load_community_summaries(level: int) -> List[Dict[str, Any]]:
    with driver.session(database=DATABASE) as session:
        return session.run(QUERY_COMMUNITIES, level=level).data()


def _truncate(s: str, max_chars: int) -> str:
    s = s or ""
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3] + "..."


def _fetch_chunk_texts_for_community(community_id: str, level: int, limit: int) -> List[str]:
    if not community_id:
        return []
    with driver.session(database=DATABASE) as session:
        rows = session.run(
            CHUNKS_FOR_COMMUNITY_QUERY,
            communityId=str(community_id),
            level=int(level),
            limit=int(limit),
        ).data()
    out: List[str] = []
    for r in rows:
        txt = (r.get("chunk_text") or "").strip()
        if txt:
            out.append(txt)
    return out


def build_chunk_context_items_for_logging_from_rows(
    rows: List[Dict[str, Any]],
    level: int,
    total_limit: int = CHUNKS_LOG_LIMIT_TOTAL,
    per_community_limit: int = CHUNKS_LOG_LIMIT_PER_COMMUNITY,
) -> List[Dict[str, Any]]:
    """
    Returns ONLY chunk texts in logger format:
      [{"content": "<chunk text>"}, ...]
    """
    items: List[Dict[str, Any]] = []
    seen: set[str] = set()

    # We iterate communities in order (the same order we process batches).

    for r in rows:
        if len(items) >= total_limit:
            break

        cid = str(r.get("cid") or "").strip()
        if not cid:
            continue

        # This assumes communityId exists and matches cid.

        chunk_texts = _fetch_chunk_texts_for_community(cid, level=level, limit=per_community_limit)

        for txt in chunk_texts:
            if len(items) >= total_limit:
                break
            if txt in seen:
                continue
            seen.add(txt)
            items.append({"content": txt})

    return items


# =============================================================================
# Answer generation: process ALL communities sequentially in batches (like 2nd retriever)
# - Uses batch partial answers + final synthesis
# - Logs ONLY chunk texts (context_items)
# =============================================================================
def answer_global(
    question: str,
    level: int = LEVEL,
    batch_size: int = BATCH_SIZE,
) -> Tuple[str, List[Dict[str, Any]]]:
    rows = load_community_summaries(level=level)
    if not rows:
        return "No community summaries found for the requested level.", []

    partials: List[str] = []

    total_batches = (len(rows) + batch_size - 1) // batch_size
    print(f"[INFO] Communities loaded: {len(rows)} | batch_size={batch_size} | batches={total_batches}")

    for b in range(START_BATCH, total_batches):
        i = b * batch_size
        batch_rows = rows[i : i + batch_size]

        # Per-community truncation
        texts: List[str] = []
        for br in batch_rows:
            txt = (br.get("txt") or "").strip()
            txt = _truncate(txt, MAX_CHARS_PER_COMMUNITY)
            if txt:
                texts.append(txt)

        summaries_text = "\n\n---\n\n".join(texts)
        summaries_text = _truncate(summaries_text, MAX_CHARS_PER_BATCH)

        print(f"[INFO] Batch {b+1}/{total_batches} | communities={len(batch_rows)} | chars={len(summaries_text)}")

        msg = llm.invoke(
            BATCH_PROMPT.format_messages(
                question=question,
                summaries=summaries_text,
            )
        )
        partial = (msg.content or "").strip()
        if partial:
            partials.append(partial)

    # Final synthesis
    if not partials:
        final_answer = "No partial answers produced. Possibly empty context."
    elif len(partials) == 1:
        final_answer = partials[0]
    else:
        msg = llm.invoke(
            SYNTHESIS_PROMPT.format_messages(
                question=question,
                partials="\n\n---\n\n".join(partials),
            )
        )
        final_answer = (msg.content or "").strip()



    context_items = build_chunk_context_items_for_logging_from_rows(
        rows=rows,
        level=level,
        total_limit=CHUNKS_LOG_LIMIT_TOTAL,
        per_community_limit=CHUNKS_LOG_LIMIT_PER_COMMUNITY,
    )


    return final_answer, context_items


# =============================================================================
# Batch runner (JSONL)
# =============================================================================
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
            print(f"[CTX] n_context_logged={len(context_items)} (chunk texts only)\n")

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


# =============================================================================
# Manual mode
# =============================================================================
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
    print(f"[CTX] n_context_logged={len(context_items)} (chunk texts only)\n")

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
    print(f"{SCRIPT_NAME} | DB={DATABASE} | Level={LEVEL} | Model={LLM_MODEL}")
    print(f"BATCH_SIZE={BATCH_SIZE} | MAX_CHARS_PER_COMMUNITY={MAX_CHARS_PER_COMMUNITY} | MAX_CHARS_PER_BATCH={MAX_CHARS_PER_BATCH}")
    print(f"CHUNKS_LOG_LIMIT_TOTAL={CHUNKS_LOG_LIMIT_TOTAL} | CHUNKS_LOG_LIMIT_PER_COMMUNITY={CHUNKS_LOG_LIMIT_PER_COMMUNITY}")
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
