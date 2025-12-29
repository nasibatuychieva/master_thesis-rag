import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI

from main.evaluation.logger import log_antwort

load_dotenv()
# =============================================================================

# =============================================================================
import os

URI = os.getenv("NEO4J_URI")
AUTH_USER = os.getenv("NEO4J_USER")
AUTH_PASSWORD = os.getenv("NEO4J_PASSWORD")
DATABASE = "llmakg"  

SCRIPT_NAME = "LLmaIndex_Community_KG_Retriever"

QUESTIONS_PATH = Path(
    r"C:\Users\Nasiba\Documents\1 Master Data Science\Master Thesis\VS Code New\master_thesis-rag\main\evaluation\graphrag\golden_answers_dataset_new.jsonl"
)

# Community retrieval parameters
COMMUNITY_LABEL = "__Community__"
COMMUNITY_LEVEL = 1

# How many communities to consider for answering a question:

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
  AND c.full_content IS NOT NULL
  AND c.full_content <> ""
RETURN
  c.communityId   AS communityId,
  c.level         AS level,
  c.full_content  AS full_content,
  c.summary       AS summary,
  c.topic_label   AS topic_label,
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
#    (Default GraphRAG global retriever uses ALL communities;
#     we do a lightweight selection to keep context smaller & more relevant.)
# =============================================================================
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")  # simple tokenization

def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(text or "") if len(t) >= 3]

def score_community(question_tokens: List[str], comm_text: str) -> int:
    # naive overlap score: count occurrences
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

    scored = []
    for c in communities:
        text = c.get("full_content") or c.get("summary") or ""
        s = score_community(q_tokens, text)
        scored.append((s, c))

    # sort by score desc, fallback by rank if exists
    scored.sort(
        key=lambda x: (
            x[0],
            (x[1].get("community_rank") or 0) if USE_COMMUNITY_RANK else 0
        ),
        reverse=True
    )

    # if everything scores 0, just take top ranked / first
    top = [c for s, c in scored[:k]]
    return top


# =============================================================================
# 3) LLM Answering (context = selected community summaries)
# =============================================================================
llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)

SYSTEM_PROMPT = """You are a technical support assistant.
Answer the user's question using ONLY the provided Community Summaries context.
If the context does not contain enough information, say what is missing and provide the best possible answer based on the context without inventing facts.
Be concise but complete.
"""

def build_context_block(selected: List[Dict[str, Any]]) -> str:
    parts = []
    for c in selected:
        cid = c.get("communityId", "")
        lvl = c.get("level", "")
        topic = c.get("topic_label", "")
        rank = c.get("community_rank", "")
        content = c.get("full_content", "") or c.get("summary", "")

        header = f"[Community level={lvl} id={cid} topic={topic} rank={rank}]".strip()
        parts.append(header + "\n" + content.strip())
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
    context_items = []
    for c in selected:
        context_items.append({
            "content": c.get("full_content") or c.get("summary") or "",
            "source": f"community:{c.get('communityId')}",
            "id": str(c.get("communityId") or ""),
            "score": "",  
            "level": c.get("level"),
            "topic_label": c.get("topic_label", ""),
            "community_rank": c.get("community_rank", ""),
        })

    return answer, context_items



# 4) Batch / Manual loops

def run_batch_from_file(select_k: int = COMMUNITY_SELECT_K):
    print(f"\n[INFO] Loading dataset from {QUESTIONS_PATH}\n")

    if not QUESTIONS_PATH.exists():
        print("[ERROR] Dataset JSONL not found.")
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
