from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# -----------------------------------------------------------------------------
# Logger config
# -----------------------------------------------------------------------------
LOGGER_DEBUG = os.getenv("LOGGER_DEBUG", "0").strip().lower() in {"1", "true", "yes", "y"}


def _dbg(msg: str, **kv: Any) -> None:
    if not LOGGER_DEBUG:
        return
    extras = ""
    if kv:
        extras = " | " + " ".join(f"{k}={repr(v)}" for k, v in kv.items())
    print(f"[LOGGER DEBUG] {msg}{extras}")


# -----------------------------------------------------------------------------
# Project root detection
# -----------------------------------------------------------------------------
def _guess_project_root() -> Path:
    env_root = os.getenv("PROJECT_ROOT")
    if env_root and env_root.strip():
        return Path(env_root).expanduser().resolve()

    here = Path(__file__).resolve()
    for p in [here.parent] + list(here.parents):
        if (p / "main").is_dir():
            return p
    return here.parent


PROJECT_ROOT = _guess_project_root()

# Default output: JSONL file (append)
DEFAULT_LOGFILE = (
    PROJECT_ROOT / "main" / "evaluation" / "graphrag" / "answers_log_new_dataset.jsonl"
)

# allow override
LOGFILE_PATH = Path(os.getenv("ANSWERS_LOG_PATH", str(DEFAULT_LOGFILE))).expanduser().resolve()
LOGFILE_PATH.parent.mkdir(parents=True, exist_ok=True)

_dbg("Initialized JSONL logger", PROJECT_ROOT=str(PROJECT_ROOT), LOGFILE=str(LOGFILE_PATH))


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _json_safe(val: Any) -> Any:
    try:
        json.dumps(val, ensure_ascii=False)
        return val
    except TypeError:
        return str(val)


def _pick_content(d: Dict[str, Any]) -> str:
    """
    Robustly pick content from various common keys.
    Priority: content -> text -> txt -> chunk_text -> full_content -> summary
    """
    for k in ("content", "text", "txt", "chunk_text", "full_content", "summary"):
        v = d.get(k, None)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return ""


def _normalize_context_items(context_items: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    context_items = context_items or []
    out: List[Dict[str, Any]] = []

    dropped_no_dict = 0
    dropped_empty = 0

    for c in context_items:
        if not isinstance(c, dict):
            dropped_no_dict += 1
            continue

        content = _pick_content(c)
        if not content:
            dropped_empty += 1
            continue

        safe = {k: _json_safe(v) for k, v in c.items()}
        # enforce normalized key
        safe["content"] = content

        # ensure node_type exists (helps your type counting)
        if "node_type" not in safe:
            # fall back to "type" if present
            safe["node_type"] = str(safe.get("type") or "unknown")

        out.append(safe)

    _dbg(
        "normalize_context_items",
        in_len=len(context_items),
        out_len=len(out),
        dropped_no_dict=dropped_no_dict,
        dropped_empty=dropped_empty,
    )
    return out


def _context_type_counts(norm_ctx: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for x in norm_ctx:
        t = str(x.get("node_type", "unknown") or "unknown")
        counts[t] = counts.get(t, 0) + 1
    return counts


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
    logfile: Path = LOGFILE_PATH,
) -> None:
    """
    Append one record as ONE JSON line (JSONL).
    This writes immediately (not "at the end").
    """

    logfile_path = Path(str(logfile)).expanduser().resolve()
    logfile_path.parent.mkdir(parents=True, exist_ok=True)

    qid = "" if question_id is None else str(question_id)
    qt = "" if query_type is None else str(query_type)
    ga = "" if gold_answer is None else str(gold_answer or "")
    q = "" if question is None else str(question or "")
    a = "" if answer is None else str(answer or "")

    norm_ctx = _normalize_context_items(context_items)
    ts_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    record = {
        "script": str(script_name or ""),
        "question_id": qid,
        "query_type": qt,
        "question": q,
        "answer": a,
        "gold_answer": ga,
        "timestamp_utc": ts_utc,
        "n_context": len(norm_ctx),
        "context_type_counts": _context_type_counts(norm_ctx),
        "context_items": norm_ctx,
    }

    #  append immediately
    with open(logfile_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


    _dbg(
        "logged jsonl record",
        script=record["script"],
        question_id=record["question_id"],
        n_context=record["n_context"],
        logfile=str(logfile_path),
    )
