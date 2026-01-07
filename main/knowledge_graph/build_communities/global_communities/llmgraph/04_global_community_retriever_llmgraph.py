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

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", str(Path(__file__).resolve().parents[3]))).expanduser().resolve()
QUESTIONS_PATH = (
    PROJECT_ROOT / "main" / "evaluation" / "graphrag" / "golden_answers_dataset_new.jsonl"
)

MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LEVEL = int(os.getenv("COMMUNITY_LEVEL", "1"))


BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))


MAX_CHARS_PER_COMMUNITY = int(os.getenv("MAX_CHARS_PER_COMMUNITY", "6000"))
MAX_CHARS_PER_BATCH = int(os.getenv("MAX_CHARS_PER_BATCH", "60000"))


START_BATCH = int(os.getenv("START_BATCH", "0"))

driver = GraphDatabase.driver(URI, auth=(AUTH_USER, AUTH_PASSWORD))
driver.verify_connectivity()

llm = ChatOpenAI(model=MODEL, temperature=0)


# ---------------------------------------------------------------------------
# 1) Cypher: fetch community summaries for a given level
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
# 4) Retrieval: load all communities (NO dropping)
# ---------------------------------------------------------------------------
def load_community_summaries(level: int) -> List[Dict[str, Any]]:
    with driver.session(database=DATABASE) as session:
        rows = session.run(QUERY_COMMUNITIES, level=level).data()
    return rows


def _truncate(s: str, max_chars: int) -> str:
    s = (s or "")
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3] + "..."


def build_context_items(batch_rows: List[Dict[str, Any]], level: int) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for r in batch_rows:
        items.append({
            "content": str(r.get("txt", "")).strip(),
            "source": f"__Community__ level={level}",
            "id": f"{level}-{r.get('cid')}",
            "score": "",
            "communityId": r.get("cid"),
            "level": level,
            "type": "community_summary",
        })
    return items


# ---------------------------------------------------------------------------
# 5) Answer generation: process ALL communities sequentially in batches
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

    # Process ALL batches 
    for b in range(START_BATCH, total_batches):
        i = b * batch_size
        batch_rows = rows[i:i + batch_size]

        # Per-community truncation
        texts = []
        for br in batch_rows:
            txt = br.get("txt", "") or ""
            txt = _truncate(txt, MAX_CHARS_PER_COMMUNITY)
            texts.append(txt)

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


        all_context_items.extend(build_context_items(batch_rows, level=level))

    # Final synthesis
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
# 6) Batch runner (JSONL)
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
            print(f"[CTX] n_context={len(context_items)} (community summaries)\n")

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
    print(f"[CTX] n_context={len(context_items)} (community summaries)\n")

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
    print(f"BATCH_SIZE={BATCH_SIZE} | MAX_CHARS_PER_COMMUNITY={MAX_CHARS_PER_COMMUNITY} | MAX_CHARS_PER_BATCH={MAX_CHARS_PER_BATCH}")
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