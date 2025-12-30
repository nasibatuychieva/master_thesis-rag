"""
LLM-as-a-Judge metrics for technical QA (Arduino-style) with DEBUG LOGGING.

FULL VERSION (all metrics):
- Answer Relevance
- Completeness (single/multi)
- Correctness (strict/relaxed)
- Faithfulness (improved for long contexts + list answers)

Option A changes (robust faithfulness, not "milder cheating"):
- Faithfulness uses evidence-focused context (extracts Evidence/Chunk blocks when present)
- Faithfulness statement extraction:
  - max N statements
  - keep list enumerations as ONE statement (do not split into 20)
  - focus on claims that require evidence
- Faithfulness verification supports: Yes | Partial | No
  - scoring: Yes=1.0, Partial=0.5, No=0.0
- Hard truncation of faithfulness context to MAX_CONTEXT_CHARS
- Resume mode: handles EOFError, supports env vars:
  - JUDGE_START_ROW (int, default 1)
  - JUDGE_MAX_ROWS (int or empty)

Run as a single .py file.
"""

from __future__ import annotations

import csv
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

# -----------------------------
# Controls
# -----------------------------
VERBOSE = True
PREVIEW_CHARS = 400

# Faithfulness Option A knobs
FAITH_MAX_STATEMENTS = int(os.getenv("FAITH_MAX_STATEMENTS", "7"))
MAX_CONTEXT_CHARS = int(os.getenv("FAITH_MAX_CONTEXT_CHARS", "12000"))
EVIDENCE_ONLY = os.getenv("FAITH_EVIDENCE_ONLY", "1") == "1"

# -----------------------------
# Small logging
# -----------------------------
def _ts() -> str:
    return time.strftime("%H:%M:%S")


def log(msg: str, **kv: Any) -> None:
    extra = " | " + " ".join(f"{k}={repr(v)}" for k, v in kv.items()) if kv else ""
    print(f"[{_ts()}] {msg}{extra}")


def preview(text: str, n: int = PREVIEW_CHARS) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[:n] + " ... [truncated]"


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if not v:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _env_opt_int(name: str) -> Optional[int]:
    v = os.getenv(name)
    if v is None:
        return None
    v = v.strip()
    if not v:
        return None
    try:
        return int(v)
    except ValueError:
        return None


def ask_int(prompt: str, default: int) -> int:
    try:
        raw = input(f"{prompt} [{default}]: ").strip()
    except EOFError:
        return default
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        log("[WARN] Invalid integer. Using default.", raw=raw, default=default)
        return default


def ask_optional_int(prompt: str) -> Optional[int]:
    try:
        raw = input(f"{prompt} [empty = no limit]: ").strip()
    except EOFError:
        return None
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        log("[WARN] Invalid integer. No limit will be applied.", raw=raw)
        return None


# -----------------------------
# Load env
# -----------------------------
load_dotenv(find_dotenv())
if VERBOSE:
    log("Loaded .env", env_file=find_dotenv() or "NOT FOUND")

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found. Ensure .env is loaded or set the env var in your shell.")


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class TestCase:
    script: str
    question_id: str
    query_type: str
    question: str
    answer: str
    expected_steps: str
    expected_causes: Optional[str] = None
    context: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class EvaluationResult:
    script: str
    question_id: str
    query_type: str
    metrics: Dict[str, Any]


# -----------------------------
# Prompts
# -----------------------------
GEN_QUESTION_PROMPT = """
You are given an answer. Generate exactly {num_questions} questions that could
reasonably be answered with this answer.

Input Answer:
{answer}

Instructions:
- Only generate questions that this answer could directly respond to.
- If the answer indicates uncertainty/refusal/no-information, return an empty list.
- Return STRICT JSON only:
{{"questions": ["Q1?", "Q2?", "Q3?"]}}
"""

COMPLETENESS_SINGLE_PROMPT = """
You are evaluating the completeness of a technical support answer.

Question:
{question}

Actual answer:
{actual_answer}

Expected answer (gold steps):
{expected_steps}

Task:
- Identify procedural steps in the expected answer and in the actual answer.
- Count total expected steps and total actual steps.
- Count wrong steps and missing steps (semantic equivalence allowed).
- Compute rates over expected steps.
- Provide overall_completeness_score in [0,1].

Return STRICT JSON with exactly these keys:
{{
  "total_actual_steps": <int>,
  "total_expected_steps": <int>,
  "wrong_step_count": <int>,
  "missing_step_count": <int>,
  "step_error_rate": <float>,
  "step_error_rate_justification": "<string>",
  "step_omission_rate": <float>,
  "step_omission_rate_justification": "<string>",
  "overall_completeness_score": <float>
}}
"""

COMPLETENESS_MULTI_PROMPT = """
You are evaluating the completeness of a technical troubleshooting answer.

Question:
{question}

Actual answer:
{actual_answer}

Expected troubleshooting steps (gold):
{expected_steps}

Expected problem causes (gold):
{expected_causes}

Task:
A) Steps: counts + error/omission rates over expected steps
B) Causes: counts + error/omission rates over expected causes
Return overall_completeness_score in [0,1].

Return STRICT JSON with exactly these keys:
{{
  "total_actual_steps": <int>,
  "total_expected_steps": <int>,
  "wrong_step_count": <int>,
  "missing_step_count": <int>,
  "step_error_rate": <float>,
  "step_error_rate_justification": "<string>",
  "step_omission_rate": <float>,
  "step_omission_rate_justification": "<string>",

  "total_actual_cause_count": <int>,
  "total_expected_cause_count": <int>,
  "wrong_cause_count": <int>,
  "missing_cause_count": <int>,
  "cause_error_rate": <float>,
  "cause_error_rate_justification": "<string>",
  "cause_omission_rate": <float>,
  "cause_omission_rate_justification": "<string>",

  "overall_completeness_score": <float>
}}
"""

CORRECTNESS_PROMPT = """
You are evaluating whether a support bot correctly responded to user questions.

Question:
{question}

Expected causes (may be empty):
{expected_causes}

Expected steps/instructions:
{expected_steps}

Actual answer generated by the bot:
{actual_answer}

Task:
1) If the answer is a meta-response that does not address the question, classify as FN.
2) Otherwise decide TP/FP using these definitions:

TP: {tp_definition}
FP: {fp_definition}

Return STRICT JSON only:
{{"category":"TP|FP|TN|FN","justification":"<short explanation>"}}
"""

# Option A: statement generation prompt (focus on evidence-bearing claims; keep lists intact)
GEN_STATEMENTS_PROMPT_V2 = """
Given a question and answer, extract up to {max_statements} evidence-bearing atomic statements for faithfulness checking.

Question:
{question}

Answer:
{answer}

Rules:
- Focus only on claims that require support from context (facts, numbers, named items, "X uses Y", "steps", "causes").
- DO NOT split long enumerations into many statements. If the answer lists many items (e.g., products), keep it as ONE statement:
  Example: "Products: A, B, C, ...".
- Ignore filler text (greetings, hedging, formatting).
- If the answer is a meta-response (apology/no-info/refusal/request for more details), return exactly:
  {{"statements":["I am sorry."]}}

Return STRICT JSON only:
{{"statements":["...","..."]}}
"""

# Option A: verification prompt with Yes/Partial/No
VERIFY_STATEMENTS_PROMPT_V2 = """
Consider Context and the Statements. Determine whether each statement is supported by the context.

Verdicts:
- "Yes": explicitly supported by context.
- "Partial": the main claim is supported but details are missing/unclear (e.g., list is incomplete in context, or only some items supported).
- "No": not supported or contradicted.

Rules:
- If context is empty: return "Yes" only for "I am sorry.", otherwise "No".
- Provide a brief explanation per statement.

Context:
{context}

Statements:
{statements}

Return STRICT JSON only:
{{
  "1": {{"verdict":"Yes|Partial|No","explanation":"..." }},
  "2": {{"verdict":"Yes|Partial|No","explanation":"..." }}
}}
"""


# -----------------------------
# Helpers
# -----------------------------
def _extract_first_json_object(text: str) -> Optional[Any]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
    if not m:
        return None
    candidate = m.group(1).strip()
    try:
        return json.loads(candidate)
    except Exception:
        return None


def _require_fields(obj: Any, fields: List[Tuple[str, type]]) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        raise ValueError("Expected a JSON object (dict).")
    out: Dict[str, Any] = {}
    for name, typ in fields:
        if name not in obj:
            raise ValueError(f"Missing field: {name}")
        val = obj[name]
        if typ is float and isinstance(val, (int, float)):
            out[name] = float(val)
        elif typ is int and isinstance(val, bool):
            raise ValueError(f"Field {name} must be int, not bool.")
        elif typ is int and isinstance(val, (int, np.integer)):
            out[name] = int(val)
        elif not isinstance(val, typ):
            raise ValueError(f"Field {name} must be {typ.__name__}, got {type(val).__name__}")
        else:
            out[name] = val
    return out


def is_meta_response(answer: str) -> bool:
    a = (answer or "").strip().lower()
    if not a:
        return True
    patterns = [
        "i'm sorry", "i am sorry", "cannot answer", "can't answer", "unable to answer",
        "no information", "couldn't find", "could not find", "i don't have", "do not have",
        "please provide more details", "need more details", "not enough information",
        "as an ai", "i can't help with that"
    ]
    return any(p in a for p in patterns)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


# -----------------------------
# Context parsing from CSV
# -----------------------------
def parse_context_cell(cell: Any) -> List[Dict[str, Any]]:
    if cell is None:
        return []

    if isinstance(cell, list):
        return [x for x in cell if isinstance(x, dict)]
    if isinstance(cell, dict):
        return [cell]

    s = str(cell).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return []

    s = s.replace('""', '"')

    def try_load(x: str):
        try:
            return json.loads(x)
        except Exception:
            return None

    obj: Any = s
    for _ in range(6):
        if isinstance(obj, (list, dict)):
            break

        if isinstance(obj, str):
            t = obj.strip()
            loaded = try_load(t)
            if loaded is not None:
                obj = loaded
                continue

            if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
                obj = t[1:-1]
                continue

            if "\n" in t or "\r" in t:
                repaired = t.replace("\r\n", "\\n").replace("\n", "\\n").replace("\r", "\\n")
                loaded2 = try_load(repaired)
                if loaded2 is not None:
                    obj = loaded2
                    continue

            break

    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        return [obj]

    if isinstance(obj, str):
        cand = obj.strip().replace('""', '"')
        cand = cand.replace("\r\n", "\\n").replace("\n", "\\n").replace("\r", "\\n")
        loaded = try_load(cand)
        if isinstance(loaded, list):
            return [x for x in loaded if isinstance(x, dict)]
        if isinstance(loaded, dict):
            return [loaded]

    return []


def normalize_context_list(ctx: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for d in (ctx or []):
        if not isinstance(d, dict):
            continue
        content = d.get("content") or d.get("page_content") or d.get("text") or ""
        md = d.get("metadata") if isinstance(d.get("metadata"), dict) else {}
        out.append(
            {
                "content": str(content),
                "source": str(d.get("source", md.get("source", ""))),
                "id": str(d.get("id", d.get("chunk_id", md.get("id", "")))),
                "score": str(d.get("score", md.get("score", ""))),
            }
        )
    return [x for x in out if x.get("content", "").strip()]


def detect_context_column(df: pd.DataFrame) -> Optional[str]:
    priority = [
        "context_json", "contexts_json",
        "retrieved_context", "retrieved_contexts",
        "retrieved_chunks", "chunks",
        "source_documents", "documents",
        "contexts", "context",
        "context_preview",
    ]
    for c in priority:
        if c in df.columns:
            return c
    return None


# -----------------------------
# Option A: evidence-focused context builder for faithfulness
# -----------------------------
_CHUNK_HEADER_RE = re.compile(r"^\[Chunk .*?\]$", flags=re.MULTILINE)

def extract_evidence_from_context(norm_ctx: List[Dict[str, str]]) -> str:
    """
    Tries to extract only chunk blocks if your logged context contains:
    - "Evidence chunks:" sections
    - lines like: [Chunk product=... category=... id=...]
    Otherwise fallback: join full content.
    """
    contents = [(d.get("content") or "").strip() for d in (norm_ctx or []) if (d.get("content") or "").strip()]
    if not contents:
        return ""

    joined = "\n\n".join(contents)

    # Prefer "Evidence chunks:" blocks if present
    if "Evidence chunks:" in joined:
        parts = []
        # Split per context doc, keep only after "Evidence chunks:"
        for c in contents:
            if "Evidence chunks:" in c:
                after = c.split("Evidence chunks:", 1)[1].strip()
                if after:
                    parts.append(after)
        if parts:
            return "\n\n".join(parts).strip()

    # Else, try to keep only [Chunk ...] blocks if present
    if _CHUNK_HEADER_RE.search(joined):
        lines = joined.splitlines()
        out_lines: List[str] = []
        keep = False
        for line in lines:
            if _CHUNK_HEADER_RE.match(line.strip()):
                keep = True
                out_lines.append(line)
                continue
            if keep:
                # stop if we hit an empty line + next non-chunk section header (heuristic)
                out_lines.append(line)
        text = "\n".join(out_lines).strip()
        return text if text else joined

    return joined


def truncate_context(text: str, max_chars: int) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    # keep head+tail to avoid cutting only introductions
    head = int(max_chars * 0.75)
    tail = max_chars - head - 50
    if tail < 0:
        tail = 0
    if tail == 0:
        return t[:max_chars] + " ...[truncated]"
    return t[:head] + "\n...[truncated]...\n" + t[-tail:]


# -----------------------------
# Judge client
# -----------------------------
@dataclass
class JudgeConfig:
    api_key: str
    base_url: Optional[str] = None
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    temperature: float = 0.0


class LLMJudge:
    def __init__(self, cfg: JudgeConfig):
        self.cfg = cfg
        self.client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
        if VERBOSE:
            log("Initialized LLMJudge", chat_model=cfg.chat_model, embedding_model=cfg.embedding_model, temperature=cfg.temperature)

    def chat(self, prompt: str, *, tag: str = "") -> str:
        if VERBOSE:
            log("LLM.chat() ->", tag=tag, prompt_preview=preview(prompt))
        t0 = time.time()
        resp = self.client.chat.completions.create(
            model=self.cfg.chat_model,
            temperature=self.cfg.temperature,
            messages=[
                {"role": "system", "content": "You output strict JSON when asked. Do not add markdown."},
                {"role": "user", "content": prompt},
            ],
        )
        dt = time.time() - t0
        text = (resp.choices[0].message.content or "").strip()
        if VERBOSE:
            log("LLM.chat() <-", tag=tag, seconds=round(dt, 3), response_preview=preview(text))
        return text

    def embed(self, texts: List[str], *, tag: str = "") -> List[np.ndarray]:
        if VERBOSE:
            log("LLM.embed() ->", tag=tag, n_texts=len(texts), first_text_preview=preview(texts[0] if texts else ""))
        t0 = time.time()
        resp = self.client.embeddings.create(model=self.cfg.embedding_model, input=texts)
        dt = time.time() - t0
        vecs = [np.array(item.embedding, dtype=np.float32) for item in resp.data]
        if VERBOSE:
            log("LLM.embed() <-", tag=tag, seconds=round(dt, 3), dim=len(vecs[0]) if vecs else None)
        return vecs


# -----------------------------
# Metric 1: Answer Relevance
# -----------------------------
def eval_answer_relevance(
    judge: LLMJudge,
    tc: TestCase,
    num_questions: int = 3,
    similarity_threshold: float = 0.80,
    aggregation: str = "continuous",
) -> Dict[str, Any]:
    metric = "answer_relevance"
    if not tc.answer or not tc.answer.strip():
        return {metric: {"score": 0.0, "reason": "Empty answer"}}

    raw = judge.chat(GEN_QUESTION_PROMPT.format(answer=tc.answer, num_questions=num_questions), tag=f"{metric}:gen_questions")
    obj = _extract_first_json_object(raw)

    if not isinstance(obj, dict) or "questions" not in obj or not isinstance(obj["questions"], list):
        return {metric: {"score": 0.0, "reason": "Invalid question generation JSON", "raw": preview(raw, 800)}}

    questions = [q.strip() for q in obj["questions"] if isinstance(q, str) and q.strip()][:num_questions]
    if not questions:
        return {metric: {"score": 0.0, "reason": "No questions generated"}}

    vecs = judge.embed([tc.question] + questions, tag=f"{metric}:embeddings")
    r = vecs[0]
    sims = [cosine(r, v) for v in vecs[1:]]

    score_cont = float(np.mean(sims)) if sims else 0.0
    score_thr = float(sum(1 for s in sims if s >= similarity_threshold) / len(sims)) if sims else 0.0
    score_used = score_cont if aggregation == "continuous" else score_thr
    score_used = clamp01(score_used)

    return {
        metric: {
            "score": score_used,
            "score_continuous": score_cont,
            "score_thresholded": score_thr,
            "generated_questions": questions,
            "cosine_similarities": sims,
            "aggregation": aggregation,
            "threshold": similarity_threshold,
        }
    }


# -----------------------------
# Metric 2: Completeness
# -----------------------------
def eval_completeness(judge: LLMJudge, tc: TestCase) -> Dict[str, Any]:
    metric = "completeness"
    if not tc.answer or not tc.answer.strip():
        return {metric: {"overall_completeness_score": 0.0, "reason": "Empty answer"}}

    if tc.expected_causes and tc.expected_causes.strip():
        prompt = COMPLETENESS_MULTI_PROMPT.format(
            question=tc.question,
            actual_answer=tc.answer,
            expected_steps=tc.expected_steps,
            expected_causes=tc.expected_causes,
        )
        raw = judge.chat(prompt, tag=f"{metric}:multi")
        obj = _extract_first_json_object(raw)
        fields = [
            ("total_actual_steps", int),
            ("total_expected_steps", int),
            ("wrong_step_count", int),
            ("missing_step_count", int),
            ("step_error_rate", float),
            ("step_error_rate_justification", str),
            ("step_omission_rate", float),
            ("step_omission_rate_justification", str),
            ("total_actual_cause_count", int),
            ("total_expected_cause_count", int),
            ("wrong_cause_count", int),
            ("missing_cause_count", int),
            ("cause_error_rate", float),
            ("cause_error_rate_justification", str),
            ("cause_omission_rate", float),
            ("cause_omission_rate_justification", str),
            ("overall_completeness_score", float),
        ]
    else:
        prompt = COMPLETENESS_SINGLE_PROMPT.format(
            question=tc.question,
            actual_answer=tc.answer,
            expected_steps=tc.expected_steps,
        )
        raw = judge.chat(prompt, tag=f"{metric}:single")
        obj = _extract_first_json_object(raw)
        fields = [
            ("total_actual_steps", int),
            ("total_expected_steps", int),
            ("wrong_step_count", int),
            ("missing_step_count", int),
            ("step_error_rate", float),
            ("step_error_rate_justification", str),
            ("step_omission_rate", float),
            ("step_omission_rate_justification", str),
            ("overall_completeness_score", float),
        ]

    try:
        validated = _require_fields(obj, fields)
    except Exception as e:
        return {metric: {"overall_completeness_score": 0.0, "error": f"Invalid judge JSON: {e}", "raw": preview(raw, 1200)}}

    validated["overall_completeness_score"] = clamp01(float(validated["overall_completeness_score"]))
    return {metric: validated}


# -----------------------------
# Metric 3: Correctness
# -----------------------------
def eval_correctness(judge: LLMJudge, tc: TestCase, relaxed: bool = False) -> Dict[str, Any]:
    metric = "correctness"
    if not tc.expected_steps or not tc.expected_steps.strip():
        raise ValueError("expected_steps (gold) must be provided for correctness.")
    if is_meta_response(tc.answer):
        return {metric: {"category": "FN", "justification": "Meta-response / no answer."}}

    expected_causes = tc.expected_causes.strip() if (tc.expected_causes and tc.expected_causes.strip()) else "No causes are expected."

    tp_definition = (
        "The response is related to the user's question and contains at least one expected step/cause (semantic match allowed)."
        if relaxed
        else "ALL expected steps/causes are present in the response (semantic match allowed), even if worded differently."
    )
    fp_definition = (
        "A response was provided but it is missing at least one expected step/cause, "
        "OR it contradicts the expected answer, OR it contains technically incorrect claims. "
        "Do NOT penalize additional correct details; they are allowed. "
        "Do NOT classify meta-responses as FP; those are FN."
    )

    raw = judge.chat(
        CORRECTNESS_PROMPT.format(
            question=tc.question,
            expected_causes=expected_causes,
            expected_steps=tc.expected_steps,
            actual_answer=tc.answer,
            tp_definition=tp_definition,
            fp_definition=fp_definition,
        ),
        tag=f"{metric}:judge",
    )
    obj = _extract_first_json_object(raw)
    try:
        validated = _require_fields(obj, [("category", str), ("justification", str)])
    except Exception as e:
        return {metric: {"category": "FN", "justification": f"Invalid judge JSON: {e}", "raw": preview(raw, 1200)}}

    cat = validated["category"].strip().upper()
    if cat not in {"TP", "FP", "TN", "FN"}:
        cat = "FN"
        validated["justification"] = f"Invalid category from judge; coerced to FN. Original: {validated['justification']}"
    validated["category"] = cat
    validated["relaxed"] = relaxed
    return {metric: validated}


def aggregate_correctness(results: List[EvaluationResult]) -> Dict[str, float]:
    cats = [r.metrics.get("correctness", {}).get("category", "").upper() for r in results if "correctness" in r.metrics]
    s = pd.Series([c for c in cats if c in {"TP", "FP", "TN", "FN"}])
    counts = s.value_counts()

    tp = int(counts.get("TP", 0))
    fp = int(counts.get("FP", 0))
    tn = int(counts.get("TN", 0))
    fn = int(counts.get("FN", 0))

    denom = tp + fp + tn + fn
    accuracy = (tp + tn) / denom if denom else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {"accuracy": float(accuracy), "precision": float(precision), "recall": float(recall)}


# -----------------------------
# Metric 4: Faithfulness (Option A)
# -----------------------------
def eval_faithfulness(judge: LLMJudge, tc: TestCase) -> Dict[str, Any]:
    metric = "faithfulness"

    # Build context text
    norm_ctx = tc.context or []
    if EVIDENCE_ONLY:
        context_text = extract_evidence_from_context(norm_ctx)
    else:
        context_text = "\n".join((d.get("content") or "").strip() for d in norm_ctx).strip()

    context_text = truncate_context(context_text, MAX_CONTEXT_CHARS)

    if VERBOSE:
        log(
            "Faithfulness context debug",
            question_id=tc.question_id,
            evidence_only=EVIDENCE_ONLY,
            context_docs=len(norm_ctx),
            context_text_len=len(context_text),
            context_text_preview=preview(context_text, 500),
        )

    # Step 1: extract statements (Option A prompt)
    raw1 = judge.chat(
        GEN_STATEMENTS_PROMPT_V2.format(question=tc.question, answer=tc.answer, max_statements=FAITH_MAX_STATEMENTS),
        tag=f"{metric}:gen_statements_v2",
    )
    obj1 = _extract_first_json_object(raw1)

    try:
        validated1 = _require_fields(obj1, [("statements", list)])
    except Exception as e:
        return {metric: {"score": 0.0, "error": f"Invalid statement JSON: {e}", "raw": preview(raw1, 1200)}}

    statements = [s.strip() for s in validated1["statements"] if isinstance(s, str) and s.strip()]
    if not statements:
        return {metric: {"score": 0.0, "reason": "No statements extracted"}}

    # Step 2: verify statements (Yes/Partial/No)
    st_map = {str(i + 1): st for i, st in enumerate(statements)}
    st_block = "\n".join(f"{i}: {st}" for i, st in st_map.items())

    raw2 = judge.chat(
        VERIFY_STATEMENTS_PROMPT_V2.format(context=context_text, statements=st_block),
        tag=f"{metric}:verify_v2",
    )
    obj2 = _extract_first_json_object(raw2)
    if not isinstance(obj2, dict):
        return {metric: {"score": 0.0, "error": "Invalid verification JSON (not a dict)", "raw": preview(raw2, 1200)}}

    # Scoring: Yes=1, Partial=0.5, No=0
    verdict_score = {"yes": 1.0, "partial": 0.5, "no": 0.0}
    total = 0.0
    details: List[Dict[str, Any]] = []

    for key, judgement in obj2.items():
        if key not in st_map or not isinstance(judgement, dict) or "verdict" not in judgement:
            continue
        v = str(judgement["verdict"]).strip().lower()
        sc = verdict_score.get(v, 0.0)
        total += sc
        if v != "yes":
            details.append({"statement": st_map[key], "judgement": judgement})

    denom = max(len(st_map), 1)
    score = clamp01(total / denom)

    return {
        metric: {
            "score": score,
            "n_statements": len(st_map),
            "unsupported_or_partial": details,
            "statements": statements,
            "evidence_only": EVIDENCE_ONLY,
            "max_context_chars": MAX_CONTEXT_CHARS,
        }
    }


# -----------------------------
# End-to-end driver
# -----------------------------
def evaluate_testcases(
    judge: LLMJudge,
    testcases: List[TestCase],
    *,
    answer_relevance_kwargs: Optional[Dict[str, Any]] = None,
    correctness_relaxed: bool = False,
) -> List[EvaluationResult]:
    answer_relevance_kwargs = answer_relevance_kwargs or {}

    out: List[EvaluationResult] = []
    total = len(testcases)
    log("Starting evaluation", n_testcases=total)

    for idx, tc in enumerate(testcases, start=1):
        t0 = time.time()
        log("Testcase START", idx=idx, total=total, script=tc.script, question_id=tc.question_id, query_type=tc.query_type)

        metrics: Dict[str, Any] = {}
        metrics.update(eval_answer_relevance(judge, tc, **answer_relevance_kwargs))
        metrics.update(eval_completeness(judge, tc))
        metrics.update(eval_correctness(judge, tc, relaxed=correctness_relaxed))
        metrics.update(eval_faithfulness(judge, tc))

        out.append(EvaluationResult(script=tc.script, question_id=tc.question_id, query_type=tc.query_type, metrics=metrics))

        dt = time.time() - t0
        log(
            "Testcase END",
            idx=idx,
            seconds=round(dt, 3),
            answer_relevance=round(metrics.get("answer_relevance", {}).get("score", 0.0), 4),
            completeness=round(metrics.get("completeness", {}).get("overall_completeness_score", 0.0), 4)
            if isinstance(metrics.get("completeness", {}), dict) else None,
            correctness=metrics.get("correctness", {}).get("category"),
            faithfulness=round(metrics.get("faithfulness", {}).get("score", 0.0), 4),
        )

    log("Finished evaluation", n_results=len(out))
    return out


def results_to_dataframe(results: List[EvaluationResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        row = {"script": r.script, "question_id": r.question_id, "query_type": r.query_type}
        ar = r.metrics.get("answer_relevance", {})
        row["answer_relevance_score"] = ar.get("score", None)
        comp = r.metrics.get("completeness", {})
        row["completeness_score"] = comp.get("overall_completeness_score", None) if isinstance(comp, dict) else None
        corr = r.metrics.get("correctness", {})
        row["correctness_category"] = corr.get("category", None)
        faith = r.metrics.get("faithfulness", {})
        row["faithfulness_score"] = faith.get("score", None)
        row["metrics_json"] = json.dumps(r.metrics, ensure_ascii=False)
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_summary(results: List[EvaluationResult]) -> Dict[str, Any]:
    dfm = results_to_dataframe(results)
    summary: Dict[str, Any] = {
        "answer_relevance_mean": float(dfm["answer_relevance_score"].mean()) if "answer_relevance_score" in dfm else 0.0,
        "completeness_mean": float(dfm["completeness_score"].mean()) if "completeness_score" in dfm else 0.0,
        "faithfulness_mean": float(dfm["faithfulness_score"].mean()) if "faithfulness_score" in dfm else 0.0,
    }
    summary.update(aggregate_correctness(results))
    return summary


# -----------------------------
# Entry
# -----------------------------
def main() -> None:
    CSV_IN = Path(
        r"C:\Users\Nasiba\Documents\1 Master Data Science\Master Thesis\VS Code New\master_thesis-rag\main\evaluation\graphrag\answers_log_new_dataset_fixed.csv"
    )

    log("Reading input CSV", path=str(CSV_IN))

    df = pd.read_csv(
        CSV_IN,
        engine="python",
        sep=",",
        quotechar='"',
        escapechar="\\",
        quoting=csv.QUOTE_MINIMAL,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )

    log("CSV columns", columns=df.columns.tolist())

    print("\n=== Judge Resume Mode ===")
    print("Row numbers are DATA rows after the header (1 = first row after header).")

    # Prefer env vars in non-interactive runs
    start_row_default = _env_int("JUDGE_START_ROW", 1)
    max_rows_default = _env_opt_int("JUDGE_MAX_ROWS")

    start_row = ask_int("Start evaluating from data row", default=start_row_default)
    max_rows = ask_optional_int("Evaluate at most N data rows (optional)")
    if max_rows is None:
        max_rows = max_rows_default

    if start_row < 1:
        start_row = 1

    n_total = len(df)
    start_idx = start_row - 1
    if start_idx >= n_total:
        log("[ERROR] start_row beyond dataset length", start_row=start_row, n_total=n_total)
        raise SystemExit(1)

    end_idx = n_total if max_rows is None else min(n_total, start_idx + max_rows)
    df = df.iloc[start_idx:end_idx].copy()

    log("Resume selection applied", start_row=start_row, start_idx=start_idx, end_idx_exclusive=end_idx, selected_rows=len(df), total_rows=n_total)

    context_col = detect_context_column(df)
    preview_col = "context_preview" if "context_preview" in df.columns else None
    log("Detected context column", context_col=context_col, preview_col=preview_col)

    testcases: List[TestCase] = []
    empty_context_qids: List[str] = []

    for _, row in df.iterrows():
        raw_ctx = parse_context_cell(row.get(context_col)) if context_col else []
        norm_ctx = normalize_context_list(raw_ctx)

        if len(norm_ctx) == 0 and preview_col:
            pv = str(row.get(preview_col, "") or "").strip()
            if pv:
                norm_ctx = [{"content": pv, "source": "context_preview_fallback", "id": "", "score": ""}]

        qid = str(row.get("question_id", ""))
        if len(norm_ctx) == 0:
            empty_context_qids.append(qid)

        testcases.append(
            TestCase(
                script=str(row.get("script", "")),
                question_id=qid,
                query_type=str(row.get("query_type", "")),
                question=str(row.get("question", "")),
                answer=str(row.get("answer", "")),
                expected_steps=str(row.get("gold_answer", "")),
                expected_causes=None,
                context=norm_ctx,
            )
        )

    log("Context load summary", total=len(testcases), empty_context_count=len(empty_context_qids), empty_context_sample=empty_context_qids[:10])

    judge = LLMJudge(
        JudgeConfig(
            api_key=API_KEY,
            base_url=None,
            chat_model="gpt-4o-mini",
            embedding_model="text-embedding-3-small",
            temperature=0.0,
        )
    )

    results = evaluate_testcases(
        judge,
        testcases,
        answer_relevance_kwargs={"num_questions": 3, "aggregation": "continuous"},
        correctness_relaxed=False,
    )

    log("Aggregate summary", **aggregate_summary(results))

    out_df = results_to_dataframe(results)

    out_dir = Path(
        r"C:\Users\Nasiba\Documents\1 Master Data Science\Master Thesis\VS Code New\master_thesis-rag\main\evaluation\graphrag\out"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    end_row_inclusive = end_idx
    CSV_OUT = out_dir / f"llm_judge_results_{CSV_IN.stem}_rows_{start_row}_to_{end_row_inclusive}_optionA.csv"

    log("Writing output CSV", path=str(CSV_OUT), n_rows=len(out_df))
    out_df.to_csv(CSV_OUT, index=False)

    log("DONE")


if __name__ == "__main__":
    main()
