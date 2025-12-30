from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


# -----------------------------------------------------------------------------
# Logger config
# -----------------------------------------------------------------------------
LOGGER_DEBUG = os.getenv("LOGGER_DEBUG", "0").strip().lower() in {
    "1", "true", "yes", "y"
}


def _dbg(msg: str, **kv: Any) -> None:
    """
    Debug logger that supports keyword arguments.
    """
    if not LOGGER_DEBUG:
        return

    if kv:
        extras = " | " + " ".join(f"{k}={repr(v)}" for k, v in kv.items())
    else:
        extras = ""

    print(f"[LOGGER DEBUG] {msg}{extras}")


# -----------------------------------------------------------------------------
# Project root detection
# -----------------------------------------------------------------------------
def _guess_project_root() -> Path:
    """
    Priority:
      1) env PROJECT_ROOT
      2) walk up from this file until folder containing 'main'
      3) fallback: parent of this file
    """
    env_root = os.getenv("PROJECT_ROOT")
    if env_root and env_root.strip():
        return Path(env_root).expanduser().resolve()

    here = Path(__file__).resolve()
    for p in [here.parent] + list(here.parents):
        if (p / "main").is_dir():
            return p

    return here.parent


PROJECT_ROOT = _guess_project_root()

DEFAULT_LOGFILE = (
    PROJECT_ROOT
    / "main"
    / "evaluation"
    / "graphrag"
    / "answers_log_new_dataset.csv"
)

BASE_DIR = Path(
    os.getenv("QUESTIONS_PATH", str(DEFAULT_LOGFILE))
).expanduser().resolve()

BASE_DIR.parent.mkdir(parents=True, exist_ok=True)

_dbg("Initialized logger", PROJECT_ROOT=PROJECT_ROOT, LOGFILE=BASE_DIR)


# -----------------------------------------------------------------------------
# CSV schema
# -----------------------------------------------------------------------------
CSV_HEADER = [
    "script",
    "question_id",
    "query_type",
    "question",
    "answer",
    "gold_answer",
    "n_context",
    "context_preview",
    "context_json",
    "context_types_json",
    "prompt_context_text",
    "timestamp_utc",
]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _json_safe(val: Any) -> Any:
    try:
        json.dumps(val, ensure_ascii=False)
        return val
    except TypeError:
        return str(val)


def _normalize_context_items(
    context_items: Optional[List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    context_items = context_items or []
    out: List[Dict[str, Any]] = []

    for c in context_items:
        if not isinstance(c, dict):
            continue

        content = str(c.get("content", "") or "").strip()
        if not content:
            continue

        safe = {k: _json_safe(v) for k, v in c.items()}
        safe["content"] = content
        out.append(safe)

    return out


def _context_preview(norm_ctx: List[Dict[str, Any]], max_chars: int = 500) -> str:
    if not norm_ctx:
        return ""
    joined = "\n\n".join(str(x.get("content", "")) for x in norm_ctx)
    return joined[:max_chars].replace("\n", "\\n")


def _context_types(norm_ctx: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for x in norm_ctx:
        t = str(x.get("node_type", "unknown") or "unknown")
        counts[t] = counts.get(t, 0) + 1
    return counts


def _prompt_context_text(norm_ctx: List[Dict[str, Any]]) -> str:
    if not norm_ctx:
        return ""
    return "\n\n".join(
        str(x.get("content", "") or "").strip() for x in norm_ctx
    ).strip()


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def log_antwort(
    script_name: str,
    question_id: Optional[str],
    query_type: Optional[str],
    question: str,
    answer: str,
    gold_answer: Optional[str] = "",
    context_items: Optional[List[Dict[str, Any]]] = None,
    logfile: Path = BASE_DIR,
) -> None:
    """
    Logs one row to CSV with full retrieval context.
    """

    logfile_path = Path(str(logfile)).expanduser().resolve()
    logfile_path.parent.mkdir(parents=True, exist_ok=True)

    qid = "" if question_id is None else str(question_id)
    qt = "" if query_type is None else str(query_type)
    ga = "" if gold_answer is None else str(gold_answer or "")
    q = "" if question is None else str(question or "")
    a = "" if answer is None else str(answer or "")

    norm_ctx = _normalize_context_items(context_items)
    n_context = len(norm_ctx)

    context_preview = _context_preview(norm_ctx)
    context_json = json.dumps(norm_ctx, ensure_ascii=False)
    context_types_json = json.dumps(_context_types(norm_ctx), ensure_ascii=False)
    prompt_context_text = _prompt_context_text(norm_ctx)

    ts_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    file_exists = logfile_path.exists()

    with open(logfile_path, mode="a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)

        if not file_exists:
            writer.writerow(CSV_HEADER)

        writer.writerow(
            [
                script_name,
                qid,
                qt,
                q,
                a,
                ga,
                n_context,
                context_preview,
                context_json,
                context_types_json,
                prompt_context_text,
                ts_utc,
            ]
        )

    _dbg(
        "logged row",
        script=script_name,
        question_id=qid,
        n_context=n_context,
        logfile=str(logfile_path),
    )
