import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI

from main.evaluation.logger import log_antwort

load_dotenv()

# =============================================================================
# Configuration
# =============================================================================
URI = os.getenv("NEO4J_URI")
AUTH_USER = os.getenv("NEO4J_USER")
AUTH_PASSWORD = os.getenv("NEO4J_PASSWORD")
DATABASE = "simplekg"
SCRIPT_NAME = "SimpleKG_Community_KG_Retriever"

QUESTIONS_PATH = Path(
    r"C:\Users\Nasiba\Documents\1 Master Data Science\Master Thesis\VS Code New\master_thesis-rag\main\evaluation\graphrag\golden_answers_dataset_new.jsonl"
)
# Community retrieval parameters
COMMUNITY_LABEL = "__Community__"
COMMUNITY_LEVEL = 1

# How many communities to consider for answering a question:
# - If COMMUNITY_SELECT_K = 0 -> take ALL level communities 
COMMUNITY_SELECT_K = 12

USE_COMMUNITY_RANK = True

# LLM parameters
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0

# =============================================================================
# Neo4j driver
# =============================================================================
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
):
    try:
        log_antwort(
            script,
            question_id,
            query_type,
            question,
            answer,
            gold_answer or "",
            context_items=context_items,
        )
    except TypeError:
        log_antwort(script, question_id, query_type, question, answer, gold_answer or "")
    except Exception as e:
        print("[WARN] logging failed:", e)
        try:
            log_antwort(script, question_id, query_type, question, answer, "")
        except Exception:
            log_antwort(script, "", "", question, answer, "")


# =============================================================================
# 1) Fetch all level communities (full_content)
# =============================================================================
COMMUNITY_FETCH_QUERY = f"""
MATCH (c:{COMMUNITY_LABEL})
WHERE c.level = $level
  AND (
    (c.full_content IS NOT NULL AND c.full_content <> "")
    OR (c.summary IS NOT NULL AND c.summary <> "")
  )
RETURN
  coalesce(toString(c.communityId), "")                 AS communityId,
  coalesce(c.level, $level)                             AS level,
  coalesce(c.full_content, "")                          AS full_content,
  coalesce(c.summary, "")                               AS summary,
  coalesce(c.topic_label, "")                           AS topic_label,
  coalesce(c.community_rank, 0)                         AS community_rank
ORDER BY
  CASE WHEN $use_rank THEN coalesce(c.community_rank, 0) ELSE 0 END DESC,
  coalesce(toString(c.communityId), "") ASC
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
            "communityId": (r.get("communityId") or "").strip(),
            "level": int(r.get("level") or level),
            "full_content": (r.get("full_content") or "").strip(),
            "summary": (r.get("summary") or "").strip(),
            "topic_label": (r.get("topic_label") or "").strip(),
            "community_rank": r.get("community_rank", 0),
        })
    return communities

# =============================================================================
# 2) Select relevant communities for a question (keyword overlap scoring)
# =============================================================================
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")  # simple tokenization

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

    scored: List[Tuple[int, int, Dict[str, Any]]] = []
    for c in communities:
        text = c.get("full_content") or c.get("summary") or ""
        s = score_community(q_tokens, text)
        rank = int(c.get("community_rank") or 0) if USE_COMMUNITY_RANK else 0
        scored.append((s, rank, c))

    # sort by overlap desc, then by rank desc
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

    top = [c for _, _, c in scored[:k]]

    # if everything scores 0 -> just take top ranked/first k
    if top and all(score_community(q_tokens, (c.get("full_content") or c.get("summary") or "")) == 0 for c in top):
    
        top = sorted(top, key=lambda c: int(c.get("community_rank") or 0), reverse=True)[:k]

    return top


# =============================================================================
# 3) LLM Answering (context = selected community summaries)
# =============================================================================
llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)

SYSTEM_PROMPT = """You are a technical support assistant.
Answer the user's question using ONLY the provided Community Summaries context.
If the context does not contain enough information, say what is missing and provide the best possible answer based on the context without inventing facts.
Write in fluent natural language (no JSON, no Python lists).
Be concise but complete.
"""

def build_context_block(selected: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for c in selected:
        cid = c.get("communityId", "")
        lvl = c.get("level", "")
        topic = c.get("topic_label", "")
        rank = c.get("community_rank", 0)

        # prefer full_content; fallback summary
        content = c.get("full_content") or c.get("summary") or ""
        content = content.strip()

        header = f"[Community level={lvl} id={cid} topic={topic} rank={rank}]".strip()
        parts.append(header + "\n" + content)

    return "\n\n".join(parts).strip()

def answer_with_global_communities(
    question: str,
    all_level_communities: List[Dict[str, Any]],
    select_k: int = 12
) -> Tuple[str, List[Dict[str, Any]]]:
    selected = select_communities_for_question(question, all_level_communities, k=select_k)
    context_text = build_context_block(selected)

    user_prompt = f"""Question:
{question}

Community Summaries Context:
{context_text}

Answer:"""

    resp = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ])
    answer = (resp.content or "").strip()

    # context items for logging
    context_items: List[Dict[str, Any]] = []
    for c in selected:
        context_items.append({
            "content": c.get("full_content") or c.get("summary") or "",
            "source": f"{COMMUNITY_LABEL}:level={c.get('level')} communityId={c.get('communityId')}",
            "id": str(c.get("communityId") or ""),
            "score": "",  # could add overlap score if you want later
            "level": c.get("level"),
            "topic_label": c.get("topic_label", ""),
            "community_rank": c.get("community_rank", 0),
            "type": "community_summary",
        })

    return answer, context_items


# =============================================================================
# 4) Batch / Manual loops
# =============================================================================
def run_batch_from_file(select_k: int = COMMUNITY_SELECT_K):
    print(f"\n[INFO] Loading dataset from {QUESTIONS_PATH}\n")

    if not QUESTIONS_PATH.exists():
        print("[ERROR] Dataset JSONL not found:", QUESTIONS_PATH)
        return

    all_comms = fetch_level_communities(COMMUNITY_LEVEL)
    print(f"[INFO] Loaded {len(all_comms)} community summaries at level={COMMUNITY_LEVEL}\n")

    if len(all_comms) == 0:
        print("[ERROR] No community summaries found. Check label/properties in simplekg.")
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
                answer, context_items = answer_with_global_communities(
                    question=question,
                    all_level_communities=all_comms,
                    select_k=select_k
                )
            except Exception as e:
                print("[ERROR] answering failed:", e)
                answer = f"ERROR during answering: {e}"
                context_items = []

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


def manual_question(select_k: int = COMMUNITY_SELECT_K):
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


def main_loop(select_k: int = COMMUNITY_SELECT_K):
    print(f"{SCRIPT_NAME}")
    print(f"- DB: {DATABASE}")
    print(f"- Community label: {COMMUNITY_LABEL}")
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
