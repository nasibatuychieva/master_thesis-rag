import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv, find_dotenv
from neo4j import GraphDatabase

from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

from main.evaluation.logger import log_antwort

# ---------------------------------------------------------------------------
# 1) Config
# ---------------------------------------------------------------------------
load_dotenv(find_dotenv())

URI = os.getenv("NEO4J_URI")
AUTH_USER = os.getenv("NEO4J_USER")
AUTH_PASSWORD = os.getenv("NEO4J_PASSWORD")
DATABASE = "llmagraphtrkg"

VECTOR_INDEX = "chunkEmbedding_llmagraphtrkg"
FULLTEXT_INDEX = "chunkFulltext_llmagraphtrkg"

SCRIPT_NAME = "LLMGraph_Hybrid_KG_Retriever_Rerank"

QUESTIONS_PATH = Path(
    r"C:\Users\Nasiba\Documents\1 Master Data Science\Master Thesis\VS Code New\master_thesis-rag\main\evaluation\graphrag\golden_answers_dataset_new.jsonl"
)

driver = GraphDatabase.driver(URI, auth=(AUTH_USER, AUTH_PASSWORD))
driver.verify_connectivity()

embedder = OpenAIEmbeddings(model="text-embedding-3-small")

llm_rerank = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_answer = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ---------------------------------------------------------------------------
# 2) Logging helper
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

# ---------------------------------------------------------------------------
# 3) Fulltext-safe query builder
# ---------------------------------------------------------------------------
_LUCENE_BAD = re.compile(r'[\+\-\!\(\)\{\}\[\]\^"~\*\?:\\\/]|&&|\|\|')

def build_fulltext_query(question: str, *, max_terms: int = 18) -> str:
    q = (question or "").strip()
    if not q:
        return ""

    q = _LUCENE_BAD.sub(" ", q)
    q = re.sub(r"\s+", " ", q).strip()

    tokens = []
    for t in q.split(" "):
        tt = t.strip()
        if len(tt) < 3:
            continue
        if tt.isdigit():
            continue
        tokens.append(tt)

    seen = set()
    uniq = []
    for t in tokens:
        low = t.lower()
        if low in seen:
            continue
        uniq.append(t)
        seen.add(low)

    uniq = uniq[:max_terms]
    if not uniq:
        return ""

    return " OR ".join(uniq)

# ---------------------------------------------------------------------------

HYBRID_QUERY = """
CALL {
  // VECTOR
  CALL db.index.vector.queryNodes($vector_index, $k_vec, $qvec)
  YIELD node, score
  RETURN elementId(node) AS eid, score AS score, 'vec' AS src

  UNION ALL

  // FULLTEXT
  CALL db.index.fulltext.queryNodes($fulltext_index, $qtext, {limit: $k_ft})
  YIELD node, score
  RETURN elementId(node) AS eid, score AS score, 'ft' AS src
}
RETURN eid, score, src
"""

POST_QUERY = """
UNWIND $rows AS row
MATCH (node) WHERE elementId(node) = row.eid
WITH node, row.score AS score

OPTIONAL MATCH (node)-[:MENTIONS]->(e)
WHERE e IS NOT NULL AND NOT e:Chunk

WITH node, score,
     collect(DISTINCT coalesce(e.name, e.id, elementId(e))) AS entities

RETURN DISTINCT
  coalesce(node.id, elementId(node)) AS chunk_id,
  coalesce(node.text, "")            AS text,
  coalesce(node.file_name, "")       AS file_name,
  score                              AS score,
  entities                           AS entities
ORDER BY score DESC
"""

def retrieve_candidates_manual_hybrid(question: str, *, candidate_k: int = 60) -> List[Dict[str, Any]]:
    q_original = (question or "").strip()
    if not q_original:
        return []

    qvec = embedder.embed_query(q_original)

    qtext = build_fulltext_query(q_original)
    if not qtext:
        qtext = _LUCENE_BAD.sub(" ", q_original)
        qtext = re.sub(r"\s+", " ", qtext).strip()

    k_vec = max(10, candidate_k)
    k_ft  = max(10, candidate_k)

    with driver.session(database=DATABASE) as session:
        raw = session.run(
            HYBRID_QUERY,
            vector_index=VECTOR_INDEX,
            fulltext_index=FULLTEXT_INDEX,
            k_vec=k_vec,
            k_ft=k_ft,
            qvec=qvec,
            qtext=qtext,
        ).data()

    if not raw:
        print("[DEBUG] hybrid raw returned 0 rows")
        return []

    vec_scores = [r["score"] for r in raw if r.get("src") == "vec" and isinstance(r.get("score"), (int, float))]
    ft_scores  = [r["score"] for r in raw if r.get("src") == "ft"  and isinstance(r.get("score"), (int, float))]

    vec_max = max(vec_scores) if vec_scores else 1.0
    ft_max  = max(ft_scores)  if ft_scores  else 1.0

    merged: Dict[str, float] = {}
    for r in raw:
        eid = r.get("eid")
        score = r.get("score", 0.0)
        src = r.get("src")

        if not eid:
            continue
        if not isinstance(score, (int, float)):
            score = 0.0

        if src == "vec":
            score_norm = float(score) / float(vec_max) if vec_max else 0.0
        else:
            score_norm = float(score) / float(ft_max) if ft_max else 0.0

        prev = merged.get(eid, 0.0)
        if score_norm > prev:
            merged[eid] = score_norm

    rows = [{"eid": eid, "score": sc} for eid, sc in merged.items()]
    rows.sort(key=lambda x: x["score"], reverse=True)
    rows = rows[:candidate_k]

    print(
        "[DEBUG] hybrid",
        f"qtext='{qtext[:120]}'",
        f"raw_rows={len(raw)}",
        f"merged_unique={len(merged)}",
        f"topk={len(rows)}",
    )

    with driver.session(database=DATABASE) as session:
        out = session.run(POST_QUERY, rows=rows).data()

    candidates: List[Dict[str, Any]] = []
    for r in out:
        text = (r.get("text") or "").strip()
        if not text:
            continue
        candidates.append({
            "chunk_id": str(r.get("chunk_id") or ""),
            "file_name": str(r.get("file_name") or ""),
            "score": float(r.get("score") or 0.0),
            "entities": r.get("entities", []) or [],
            "text": text,
        })

    return candidates

# ---------------------------------------------------------------------------
# 5) Rerank + Answer
# ---------------------------------------------------------------------------
def _truncate(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= max_chars else s[:max_chars].rstrip() + " ..."

def rerank_candidates(question: str, candidates: List[Dict[str, Any]], rerank_top_k: int) -> List[Dict[str, Any]]:
    if not candidates:
        return []

    show_n = min(len(candidates), 25)
    blocks: List[str] = []
    for i in range(show_n):
        c = candidates[i]
        blocks.append(
            f"[DOC {i}] chunk_id={c.get('chunk_id','')} file={c.get('file_name','')} score={c.get('score','')}\n"
            f"{_truncate(c.get('text',''), 1200)}"
        )

    joined_blocks = "\n\n".join(blocks)

    prompt = (
        "You are a reranker for technical Arduino documentation QA.\n"
        "Select documents that contain DIRECT evidence for the question.\n\n"
        f"Question:\n{question}\n\n"
        f"Candidate documents:\n{joined_blocks}\n\n"
        f"Return ONLY a comma-separated list of the {rerank_top_k} most relevant DOC indices "
        f"(choose from 0..{show_n-1})."
    )

    raw = (llm_rerank.invoke(prompt).content or "").strip()

    indices: List[int] = []
    for tok in raw.replace("\n", ",").split(","):
        tok = tok.strip()
        if tok.isdigit():
            idx = int(tok)
            if 0 <= idx < show_n:
                indices.append(idx)

    if not indices:
        return candidates[: min(rerank_top_k, show_n)]

    seen = set()
    uniq: List[int] = []
    for i in indices:
        if i not in seen:
            uniq.append(i)
            seen.add(i)

    return [candidates[i] for i in uniq[:rerank_top_k]]

def build_context(selected: List[Dict[str, Any]], *, max_total_chars: int = 12000) -> str:
    parts = []
    total = 0
    for i, c in enumerate(selected, start=1):
        header = (
            f"Result {i}\n"
            f"chunk_id: {c.get('chunk_id','')}\n"
            f"file: {c.get('file_name','')}\n"
            f"score: {c.get('score','')}\n"
            f"entities: {c.get('entities', [])}\n"
        )
        block = header + "\n" + (c.get("text", "") or "")

        if total + len(block) > max_total_chars:
            remaining = max_total_chars - total
            if remaining > 200:
                parts.append(_truncate(block, remaining))
            break

        parts.append(block)
        total += len(block)

    return "NO CONTEXT FOUND." if not parts else "\n\n" + ("\n\n" + "-" * 60 + "\n\n").join(parts)

def answer_question(
    question: str,
    *,
    candidate_k: int = 60,
    rerank_top_k: int = 8,
) -> Tuple[str, List[Dict[str, Any]]]:
    candidates = retrieve_candidates_manual_hybrid(question, candidate_k=candidate_k)
    selected = rerank_candidates(question, candidates, rerank_top_k=rerank_top_k)

    context = build_context(selected, max_total_chars=12000)
    print(f"[DEBUG] candidates={len(candidates)} selected={len(selected)} context_chars={len(context)}")

    answer_prompt = (
        "You are a technical assistant for Arduino product documentation.\n"
        "Use ONLY the provided context as evidence.\n"
        "If the context is insufficient, explain WHAT is missing (do not guess).\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n"
    )

    answer = (llm_answer.invoke(answer_prompt).content or "").strip()

    context_items: List[Dict[str, Any]] = []
    for c in selected:
        context_items.append({
            "id": c.get("chunk_id", ""),
            "source": c.get("file_name", ""),
            "score": c.get("score", ""),
            "entities": c.get("entities", []),
            "content": c.get("text", ""),
        })

    return answer, context_items

# ---------------------------------------------------------------------------
# 6) Batch / Manual
# ---------------------------------------------------------------------------
def run_batch_from_file(candidate_k: int = 60, rerank_top_k: int = 8):
    print(f"\n[INFO] Loading dataset from {QUESTIONS_PATH}\n")
    if not QUESTIONS_PATH.exists():
        print("[ERROR] golden_answers_dataset_new.jsonl not found.")
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

            question_id = str(obj.get("id") or obj.get("question_id") or obj.get("query_id") or "")
            query_type  = str(obj.get("query_type") or "")
            question    = str(obj.get("question") or "")
            gold_answer = str(obj.get("gold_answer") or "")

            if not question:
                continue

            print(f"[QID {question_id}] [{query_type}] {question}")

            answer, context_items = answer_question(
                question,
                candidate_k=candidate_k,
                rerank_top_k=rerank_top_k,
            )

            print(f"[ANSWER]\n{answer}\n")
            print(f"[CTX] used={len(context_items)}\n")

            safe_log(
                SCRIPT_NAME,
                question_id,
                query_type,
                question,
                answer,
                gold_answer,
                context_items=context_items,
            )

    print("\n[INFO] Batch processing completed.\n")

def manual_question(candidate_k: int = 60, rerank_top_k: int = 8):
    qid = input("Question ID (optional): ").strip()
    qtype = input("Query type (optional): ").strip()
    question = input("Question: ").strip()
    gold = input("Gold Answer (optional): ").strip()

    if not question:
        print("Empty question, skipping.\n")
        return

    answer, context_items = answer_question(
        question,
        candidate_k=candidate_k,
        rerank_top_k=rerank_top_k,
    )

    print("\nAnswer:\n", answer, "\n")
    print(f"[CTX] used={len(context_items)}\n")

    safe_log(
        SCRIPT_NAME,
        qid or "",
        qtype or "manual",
        question,
        answer,
        gold or "",
        context_items=context_items,
    )

def main_loop():
    print("Manual Hybrid (Vector + Fulltext) + LLM Rerank + Answer")
    print("Type 'exit' to quit.\n")

    while True:
        mode = input("Manual question? (y/n, or 'exit'): ").strip().lower()
        if mode in ("exit", "quit", "q"):
            break
        elif mode in ("y", "yes"):
            manual_question(candidate_k=60, rerank_top_k=8)
        elif mode in ("n", "no"):
            run_batch_from_file(candidate_k=60, rerank_top_k=8)
        else:
            print("Please enter 'y', 'n', or 'exit'.\n")

if __name__ == "__main__":
    try:
        main_loop()
    finally:
        driver.close()
