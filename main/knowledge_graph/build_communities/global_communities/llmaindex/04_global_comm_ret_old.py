from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI

from main.evaluation.logger import log_antwort

load_dotenv()

# =============================================================================
# CONFIG
# =============================================================================
URI = os.getenv("NEO4J_URI")
AUTH_USER = os.getenv("NEO4J_USER")
AUTH_PASSWORD = os.getenv("NEO4J_PASSWORD")
DATABASE = os.getenv("NEO4J_DATABASE", "llmakg")

SCRIPT_NAME = "LLmaIndex_Community_KG_Retriever"

from pathlib import Path
import os

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT")).expanduser().resolve()

QUESTIONS_PATH = (
    PROJECT_ROOT
    / "main"
    / "evaluation"
    / "graphrag"
    / "golden_answers_dataset_new.jsonl"
)


print("[DEBUG] PROJECT_ROOT:", PROJECT_ROOT)
print("[DEBUG] QUESTIONS_PATH:", QUESTIONS_PATH)

# Community retrieval parameters
COMMUNITY_LABEL = "__Community__"
COMMUNITY_LEVEL = int(os.getenv("COMMUNITY_LEVEL", "1"))

COMMUNITY_SELECT_K = int(os.getenv("COMMUNITY_SELECT_K", "12"))
USE_COMMUNITY_RANK = os.getenv("USE_COMMUNITY_RANK", "1") == "1"

# LLM parameters
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))

# Compacting parameters (increase if faithfulness drops due to missing evidence)
COMPACT_MAX_ENTITIES = int(os.getenv("COMPACT_MAX_ENTITIES", "30"))
COMPACT_MAX_CHUNKS = int(os.getenv("COMPACT_MAX_CHUNKS", "10"))
COMPACT_MAX_CHUNK_CHARS = int(os.getenv("COMPACT_MAX_CHUNK_CHARS", "2000"))
COMPACT_MAX_CONTEXT_CHARS = int(os.getenv("COMPACT_MAX_CONTEXT_CHARS", "12000"))

# =============================================================================
# Neo4j driver
# =============================================================================
if not URI or not AUTH_USER or not AUTH_PASSWORD:
    raise RuntimeError("Missing Neo4j env vars: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD")

driver = GraphDatabase.driver(URI, auth=(AUTH_USER, AUTH_PASSWORD))
driver.verify_connectivity()

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
) -> None:
    try:
        log_antwort(
            script_name=str(script),
            question_id=str(question_id or ""),
            query_type=str(query_type or ""),
            question=str(question or ""),
            answer=str(answer or ""),
            gold_answer=str(gold_answer or ""),
            context_items=context_items or [],
        )
    except Exception as e:
        print("[WARN] logging failed:", repr(e))

# =============================================================================
# 1) Fetch all level communities (full_content)
# =============================================================================
COMMUNITY_FETCH_QUERY = f"""
MATCH (c:{COMMUNITY_LABEL})
WHERE c.level = $level
  AND c.full_content IS NOT NULL
  AND c.full_content <> ""
RETURN
  c.communityId    AS communityId,
  c.level          AS level,
  c.full_content   AS full_content,
  c.summary        AS summary,
  c.topic_label    AS topic_label,
  c.community_rank AS community_rank
ORDER BY
  CASE WHEN $use_rank THEN coalesce(c.community_rank, 0) ELSE 0 END DESC,
  c.communityId ASC
"""

def fetch_level_communities(level: int) -> List[Dict[str, Any]]:
    with driver.session(database=DATABASE) as session:
        rows = session.run(
            COMMUNITY_FETCH_QUERY,
            level=level,
            use_rank=USE_COMMUNITY_RANK,
        ).data()

    communities: List[Dict[str, Any]] = []
    for r in rows:
        communities.append({
            "communityId": str(r.get("communityId") or ""),
            "level": int(r.get("level") or level),
            "full_content": (r.get("full_content") or "").strip(),
            "summary": (r.get("summary") or "").strip(),
            "topic_label": (r.get("topic_label") or "").strip(),
            "community_rank": r.get("community_rank"),
        })
    return communities

# =============================================================================
# 2) Select relevant communities for a question (keyword overlap scoring)
# =============================================================================
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")

def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(text or "") if len(t) >= 3]

def score_community(question_tokens: List[str], comm_text: str) -> int:
    t = (comm_text or "").lower()
    score = 0
    for w in question_tokens:
        if w in t:
            score += 1
    return score

def select_communities_for_question(
    question: str,
    communities: List[Dict[str, Any]],
    k: int
) -> List[Dict[str, Any]]:
    if k <= 0:
        return communities

    q_tokens = tokenize(question)
    if not q_tokens:
        return communities[:k]

    scored: List[Tuple[int, Dict[str, Any]]] = []
    for c in communities:
        text = c.get("full_content") or c.get("summary") or ""
        s = score_community(q_tokens, text)
        scored.append((s, c))

    scored.sort(
        key=lambda x: (
            x[0],
            (x[1].get("community_rank") or 0) if USE_COMMUNITY_RANK else 0
        ),
        reverse=True
    )

    return [c for s, c in scored[:k]]

# =============================================================================
# 3) Compact community content (PROMPT == LOGGING to keep faithfulness valid)
# =============================================================================
def make_compact_community_content(
    full_content: str,
    question: str,
    *,
    max_entities: int = COMPACT_MAX_ENTITIES,
    max_chunks: int = COMPACT_MAX_CHUNKS,
    max_chunk_chars: int = COMPACT_MAX_CHUNK_CHARS,
) -> str:
    """
    full_content is typically a JSON string produced during community build.
    We parse it and extract:
      - top entities
      - top products/categories
      - evidence chunks (filtered by question keywords first, else fallback to first chunks)
    Return a human-readable block used BOTH in prompt and logging.
    """
    s = (full_content or "").strip()
    if not s:
        return ""

    try:
        obj = json.loads(s)
        if not isinstance(obj, dict):
            raise ValueError("not a dict")
    except Exception:
        # If it's not JSON, treat as text
        return s[: max_chunks * max_chunk_chars]

    lvl = obj.get("level", "")
    cid = obj.get("communityId", "")
    entities = obj.get("entities", []) or []
    top_products = obj.get("top_products", []) or []
    top_categories = obj.get("top_categories", []) or []
    chunks = obj.get("chunks", []) or []

    entities = [str(x) for x in entities[:max_entities]]

    def _fmt_top(lst: List[Dict[str, Any]], title: str, k: int = 10) -> str:
        items = []
        for it in lst[:k]:
            v = it.get("value", "")
            c = it.get("count", "")
            items.append(f"- {v} (count={c})")
        return title + "\n" + ("\n".join(items) if items else "- (none)")

    # Evidence-first chunk selection:
    q = (question or "").lower()
    q_toks = [t for t in tokenize(q) if len(t) >= 3]
    # add common tech synonyms to help hit rate
    extra = []
    if "dac" in q or "digital" in q or "analog" in q:
        extra += ["dac", "digital-to-analog", "digital to analog", "analog out", "dac0", "dac1"]
    q_keys = list(dict.fromkeys(q_toks + extra))  # de-dup, keep order

    filtered = []
    if q_keys:
        for ch in chunks:
            t = (ch.get("text") or "").lower()
            if any(k in t for k in q_keys):
                filtered.append(ch)

    use_chunks = filtered if filtered else chunks

    chunk_blocks = []
    for ch in use_chunks[:max_chunks]:
        prod = ch.get("product", "")
        cat = ch.get("product_category", "")
        chunk_id = ch.get("chunk_id", "")
        text = (ch.get("text", "") or "").strip()
        if len(text) > max_chunk_chars:
            text = text[:max_chunk_chars] + " ...[truncated]"
        chunk_blocks.append(f"[Chunk product={prod} category={cat} id={chunk_id}]\n{text}")

    out = []
    out.append(f"Community level={lvl} id={cid}")
    out.append("Entities (top): " + ", ".join(entities) if entities else "Entities: (none)")
    out.append(_fmt_top(top_products, "Top products:"))
    out.append(_fmt_top(top_categories, "Top categories:"))
    if chunk_blocks:
        out.append("Evidence chunks:\n" + "\n\n".join(chunk_blocks))

    return "\n\n".join(out).strip()

# =============================================================================
# 4) LLM Answering (context = compact community blocks)
# =============================================================================
llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)

SYSTEM_PROMPT = """You are a technical support assistant.
Answer the user's question using ONLY the provided Community context.
If the context does not contain enough information, say what is missing and answer only using what is present (do not invent facts).
Be concise but complete.
"""

def build_context_and_items(selected: List[Dict[str, Any]], question: str) -> Tuple[str, List[Dict[str, Any]]]:
    parts: List[str] = []
    context_items: List[Dict[str, Any]] = []

    for idx, c in enumerate(selected, start=1):
        cid = c.get("communityId", "")
        lvl = c.get("level", "")
        topic = c.get("topic_label", "")
        rank = c.get("community_rank", "")

        compact = make_compact_community_content(
            c.get("full_content") or "",
            question=question,
        )
        if not compact:
            compact = (c.get("summary") or "").strip()

        header = f"[Community level={lvl} id={cid} topic={topic} rank={rank}]".strip()
        block = header + "\n" + compact

        if len(block) > COMPACT_MAX_CONTEXT_CHARS:
            block = block[:COMPACT_MAX_CONTEXT_CHARS] + "\n...[truncated]"

        parts.append(block)

        # IMPORTANT: log EXACTLY what went into the prompt (faithfulness alignment)
        context_items.append({
            "content": block,
            "node_type": "community",
            "rank_in_selection": idx,
            "source": f"community:{cid}",
            "id": str(cid),
            "level": lvl,
            "topic_label": topic,
            "community_rank": rank,
        })

    return "\n\n".join(parts).strip(), context_items

def answer_with_global_communities(
    question: str,
    all_level_communities: List[Dict[str, Any]],
    select_k: int = COMMUNITY_SELECT_K
) -> Tuple[str, List[Dict[str, Any]]]:
    selected = select_communities_for_question(question, all_level_communities, k=select_k)

    context_text, context_items = build_context_and_items(selected, question)

    user_prompt = f"""Question:
{question}

Community Context:
{context_text}

Answer:"""

    resp = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ])
    answer = (resp.content or "").strip()

    return answer, context_items

# =============================================================================
# 5) Batch / Manual loops
# =============================================================================
def run_batch_from_file(select_k: int = COMMUNITY_SELECT_K) -> None:
    print(f"\n[INFO] Loading dataset from {QUESTIONS_PATH}\n")
    if not QUESTIONS_PATH.exists():
        print("[ERROR] Dataset JSONL not found:", QUESTIONS_PATH)
        return

    all_comms = fetch_level_communities(COMMUNITY_LEVEL)
    print(f"[INFO] Loaded {len(all_comms)} community summaries at level={COMMUNITY_LEVEL}\n")

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

            answer, context_items = answer_with_global_communities(
                question=question,
                all_level_communities=all_comms,
                select_k=select_k
            )

            print(f"[ANSWER]\n{answer}\n")
            print(f"[CTX] n_communities_used={len(context_items)}\n")

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

def manual_question(select_k: int = COMMUNITY_SELECT_K) -> None:
    all_comms = fetch_level_communities(COMMUNITY_LEVEL)
    print(f"[INFO] Loaded {len(all_comms)} community summaries at level={COMMUNITY_LEVEL}\n")

    qid = input("Question ID (optional): ").strip()
    qtype = input("Query type (optional, e.g., factual/relational/summary): ").strip()
    question = input("Question: ").strip()
    gold_answer = input("Gold Answer (optional): ").strip()

    if not question:
        print("Empty question, skipping.\n")
        return

    answer, context_items = answer_with_global_communities(
        question=question,
        all_level_communities=all_comms,
        select_k=select_k
    )

    print("\nAnswer:\n", answer, "\n")
    print(f"[CTX] n_communities_used={len(context_items)}\n")

    safe_log(
        SCRIPT_NAME,
        qid or "",
        qtype or "manual",
        question,
        answer,
        gold_answer or "",
        context_items=context_items,
    )

def main_loop(select_k: int = COMMUNITY_SELECT_K) -> None:
    print(f"{SCRIPT_NAME}")
    print(f"- DB: {DATABASE}")
    print(f"- Community level: {COMMUNITY_LEVEL}")
    print(f"- Select top-K communities: {select_k} (0 = ALL)")
    print("Type 'exit' to quit.\n")

    while True:
        mode = input("Manual question? (y/n, or 'exit'): ").strip().lower()

        if mode in ("exit", "quit", "q"):
            break
        elif mode in ("y", "yes"):
            manual_question(select_k=select_k)
        elif mode in ("n", "no"):
            run_batch_from_file(select_k=select_k)
        else:
            print("Please enter 'y', 'n', or 'exit'.\n")

if __name__ == "__main__":
    try:
        main_loop(select_k=COMMUNITY_SELECT_K)
    finally:
        driver.close()
