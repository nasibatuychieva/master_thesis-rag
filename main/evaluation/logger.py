import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# BASE_DIR = Path(
#     r"C:\Users\Nasiba\Documents\1 Master Data Science\Master Thesis\VS Code New\master_thesis-rag\main\evaluation\answers_log_new copy.csv"
# )
BASE_DIR = Path(
    r"C:\Users\Nasiba\Documents\1 Master Data Science\Master Thesis\VS Code New\master_thesis-rag\main\evaluation\answers_log_new_dataset.csv"
)
# BASE_DIR = Path(
#     r"C:\Users\Nasiba\Documents\1 Master Data Science\Master Thesis\VS Code New\master_thesis-rag\main\evaluation\answers_log_new.csv"
# )


CSV_HEADER = [
    "script",
    "question_id",
    "query_type",
    "question",
    "answer",
    "gold_answer",
    # NEW for faithfulness/debug:
    "n_context",
    "context_preview",
    "context_json",
]

def log_antwort(
    script_name: str,
    question_id: Optional[str],
    query_type: Optional[str],
    question: str,
    answer: str,
    gold_answer: Optional[str] = "",
    context_items: Optional[List[Dict[str, Any]]] = None,   # NEW
    logfile: Path = BASE_DIR,
):
    """
    Logs one row to CSV:
      script, question_id, query_type, question, answer, gold_answer,
      n_context, context_preview, context_json
    """

    logfile = str(logfile)

    question_id = "" if question_id is None else str(question_id)
    query_type = "" if query_type is None else str(query_type)
    gold_answer = "" if gold_answer is None else str(gold_answer)
    question = "" if question is None else str(question)
    answer = "" if answer is None else str(answer)

    context_items = context_items or []
    # Normalize context items: ensure at least {"content": "..."}
    norm_ctx: List[Dict[str, Any]] = []
    for c in context_items:
        if not isinstance(c, dict):
            continue
        content = str(c.get("content", "")).strip()
        if not content:
            continue
        norm_ctx.append({
            "content": content,
            "source": c.get("source", ""),
            "id": c.get("id", ""),
            "score": c.get("score", ""),
        })

    n_context = len(norm_ctx)
    # quick preview (first ~500 chars)
    context_preview = ""
    if n_context > 0:
        joined = "\n\n".join(x["content"] for x in norm_ctx)
        context_preview = joined[:500].replace("\n", "\\n")

    context_json = json.dumps(norm_ctx, ensure_ascii=False)

    file_exists = os.path.isfile(logfile)

    with open(logfile, mode="a", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(CSV_HEADER)

        writer.writerow([
            script_name,
            question_id,
            query_type,
            question,
            answer,
            gold_answer,
            n_context,
            context_preview,
            context_json,
        ])
