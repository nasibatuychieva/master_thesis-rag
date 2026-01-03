"""
LLM-as-a-Judge metrics for technical QA (Arduino-style) with DEBUG LOGGING.

FULL VERSION (all metrics):
- Answer Relevance
- Completeness (single/multi)
- Correctness (coverage-based TP/PARTIAL/FP/FN)
- Faithfulness (improved for long contexts + list answers)
- Helpfulness (utility-style rating 1–5 + faithfulness-gated final)

MILD + POSITIVE CONTEXT BIAS (this version):
- Faithfulness is made "positive-evidence biased":
  - "No" is used ONLY when context contradicts a claim or strongly implies it is false.
  - For long/truncated contexts, prefer "Partial" when plausible.
  - Optional Anchor-Focus: filter context to chunks that mention answer anchors (product names / key terms),
    reducing false "No" due to truncation/noise.
- Keeps internal scores in [0,1], plus 1–5 mapped scores.
- Correctness outputs coverage + coverage_1to5.

INPUT FORMAT:
- Reads from a JSONL file (one JSON object per line) with keys:
  script, question_id, query_type, question, answer, gold_answer, context_items [...]

Run as a single .py file.
"""

from __future__ import annotations

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

# Faithfulness (milder defaults)
FAITH_MAX_STATEMENTS = int(os.getenv("FAITH_MAX_STATEMENTS", "5"))
MAX_CONTEXT_CHARS = int(os.getenv("FAITH_MAX_CONTEXT_CHARS", "12000"))
EVIDENCE_ONLY = os.getenv("FAITH_EVIDENCE_ONLY", "1") == "1"

# NEW: Positive bias knobs
FAITH_POSITIVE_BIAS = os.getenv("FAITH_POSITIVE_BIAS", "1") == "1"
FAITH_USE_ANCHOR_FOCUS = os.getenv("FAITH_USE_ANCHOR_FOCUS", "1") == "1"
FAITH_ANCHOR_MAX_CHUNKS = int(os.getenv("FAITH_ANCHOR_MAX_CHUNKS", "25"))

# Helpfulness gating
HELP_GATE_BASE = float(os.getenv("HELP_GATE_BASE", "0.5"))

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
    raise RuntimeError("OPENAI_API_KEY not found")


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
- Identify procedural steps/items in the expected answer and in the actual answer.
- Count total expected steps/items and total actual steps/items.
- Count wrong items and missing items (semantic equivalence allowed).
- Compute rates over expected items.
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
You are evaluating whether a support bot correctly responded to a technical question.

Question:
{question}

Gold answer (expected key facts/items):
{expected_steps}

Actual answer:
{actual_answer}

Task:
1) If the answer is a meta-response that does not address the question, return FN.
2) Otherwise, compare the actual answer to the gold answer and estimate:
   - coverage: fraction of key facts/items from the gold answer that are correctly present in the actual answer (0.0 to 1.0).
   - error_severity: "low" if extra details are minor/harmless, "medium" if they add some incorrect specifics,
     "high" if they include major incorrect claims that would mislead a user.

Rules (IMPORTANT):
- Do NOT treat additional correct information as an error.
- Do NOT automatically penalize extra items in a list unless they are clearly incorrect or misleading.
- Missing some gold items is allowed in PARTIAL.
- If the actual answer contradicts core gold facts, set error_severity="high".

Classification:
- TP: coverage >= {tp_cov} AND error_severity != "high"
- PARTIAL: coverage >= {partial_cov} AND coverage < {tp_cov} AND error_severity != "high"
- FP: coverage < {partial_cov} OR error_severity == "high"
- FN: meta-response / no answer

Return STRICT JSON only:
{{"category":"TP|PARTIAL|FP|FN","coverage":<float>,"error_severity":"low|medium|high","justification":"<short>"}}
"""

HELPFULNESS_PROMPT = """
You are evaluating the HELPFULNESS (utility) of a technical support answer.

Question:
{question}

Answer:
{actual_answer}

Rate helpfulness on a 1–5 scale:

5 = Highly helpful: directly addresses the topic, gives concrete and actionable information or steps, clear structure.
4 = Helpful: mostly actionable and relevant, minor gaps.
3 = Somewhat helpful: relevant but vague, incomplete, limited actionable guidance.
2 = Slightly helpful: only weakly related or mostly generic.
1 = Not helpful: off-topic, empty, or purely meta.

Rules:
- Do NOT judge factual correctness against any gold answer here.
- Penalize answers that are overly generic, hand-wavy, or only restate the question.
- Reward answers that provide steps, diagnostics, definitions, or how-to guidance.

Return STRICT JSON only:
{{"helpfulness_1to5": <int>, "justification":"<short>"}}
"""

GEN_STATEMENTS_PROMPT_V2 = """
Given a question and answer, extract up to {max_statements} evidence-bearing atomic statements for faithfulness checking.

Question:
{question}

Answer:
{answer}

Rules:
- Focus only on claims that require support from context (facts, numbers, named items, "X uses Y", "steps", "causes").
- DO NOT split long enumerations into many statements. If the answer lists many items (e.g., products), keep it as ONE statement.
- Ignore filler text.
- If the answer is a meta-response, return exactly:
  {{"statements":["I am sorry."]}}

Return STRICT JSON only:
{{"statements":["...","..."]}}
"""

# NEW: Anchor extraction prompt
GEN_ANCHORS_PROMPT = """
Extract up to {max_anchors} short "anchors" (keywords/phrases) that would likely appear in supporting context.
These anchors are used to FILTER context, so prefer:
- product names (e.g., "Nano ESP32", "GIGA R1 WiFi")
- key terms (e.g., "USB-C", "DAC", "solid-state relay")

Question:
{question}

Answer:
{answer}

Rules:
- Keep anchors short (1–5 tokens).
- No duplicates.
- Return STRICT JSON only:
{{"anchors":["...","..."]}}
"""

# UPDATED: Verification prompt with positive-evidence bias
VERIFY_STATEMENTS_PROMPT_V3 = """
Consider Context and the Statements. Determine whether each statement is supported by the context.

Verdicts:
- "Yes": clearly supported by context (explicitly or unambiguously).
- "Partial": broadly consistent with context OR plausible given partial/indirect evidence OR the core claim is supported but details are missing.
- "No": use ONLY if the context CONTRADICTS the statement OR strongly implies the opposite.

IMPORTANT POSITIVE EVIDENCE BIAS:
- When context is long, noisy, or truncated, DO NOT punish the answer for missing explicit wording.
  Prefer "Partial" unless there is contradiction.
- If a statement mentions named entities (products/terms) and those entities appear anywhere in the context,
  that is evidence toward "Partial" rather than "No" (unless contradicted).
- For lists: if at least one listed item is supported, output "Partial" (not "No") unless the context contradicts the list.
- Only output "No" when you can point to contradictory wording or a clear mismatch.

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
        "i'm sorry",
        "i am sorry",
        "cannot answer",
        "can't answer",
        "unable to answer",
        "no information",
        "couldn't find",
        "could not find",
        "i don't have",
        "do not have",
        "please provide more details",
        "need more details",
        "not enough information",
        "as an ai",
        "i can't help with that",
    ]
    return any(p in a for p in patterns)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def to_1_5(score01: float) -> int:
    s = clamp01(float(score01))
    return int(round(1 + 4 * s))


def score_1to5_to_01(s: int) -> float:
    try:
        si = int(s)
    except Exception:
        si = 1
    si = max(1, min(5, si))
    return float((si - 1) / 4.0)


# -----------------------------
# JSONL loading
# -----------------------------
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise RuntimeError(f"Invalid JSON on line {lineno}: {e}") from e
            if not isinstance(obj, dict):
                raise RuntimeError(f"JSONL line {lineno} is not an object/dict.")
            items.append(obj)
    return items


def normalize_context_items_from_jsonl(ctx_items: Any) -> List[Dict[str, str]]:
    if not ctx_items:
        return []
    if not isinstance(ctx_items, list):
        return []
    out: List[Dict[str, str]] = []
    for d in ctx_items:
        if not isinstance(d, dict):
            continue
        content = d.get("content") or d.get("page_content") or d.get("text") or ""
        md = d.get("metadata") if isinstance(d.get("metadata"), dict) else {}
        source = d.get("source") or md.get("source") or d.get("type") or d.get("node_type") or ""
        cid = d.get("id") or d.get("chunk_id") or md.get("id") or md.get("chunk_id") or ""
        score = d.get("score") or md.get("score") or ""
        out.append({"content": str(content), "source": str(source), "id": str(cid), "score": str(score)})
    return [x for x in out if x.get("content", "").strip()]


# -----------------------------
# Evidence-focused context builder
# -----------------------------
_CHUNK_HEADER_RE = re.compile(r"^\[Chunk .*?\]$", flags=re.MULTILINE)


def extract_evidence_from_context(norm_ctx: List[Dict[str, str]]) -> str:
    contents = [(d.get("content") or "").strip() for d in (norm_ctx or []) if (d.get("content") or "").strip()]
    if not contents:
        return ""
    joined = "\n\n".join(contents)

    if "Evidence chunks:" in joined:
        parts = []
        for c in contents:
            if "Evidence chunks:" in c:
                after = c.split("Evidence chunks:", 1)[1].strip()
                if after:
                    parts.append(after)
        if parts:
            return "\n\n".join(parts).strip()

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
                out_lines.append(line)
        text = "\n".join(out_lines).strip()
        return text if text else joined

    return joined


def truncate_context(text: str, max_chars: int) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    head = int(max_chars * 0.75)
    tail = max_chars - head - 50
    if tail < 0:
        tail = 0
    if tail == 0:
        return t[:max_chars] + " ...[truncated]"
    return t[:head] + "\n...[truncated]...\n" + t[-tail:]


# NEW: Anchor-focused context filter to reduce false negatives
def filter_context_by_anchors(norm_ctx: List[Dict[str, str]], anchors: List[str], max_chunks: int) -> List[Dict[str, str]]:
    if not norm_ctx or not anchors:
        return norm_ctx
    a = [x.strip().lower() for x in anchors if isinstance(x, str) and x.strip()]
    if not a:
        return norm_ctx

    hits: List[Dict[str, str]] = []
    for d in norm_ctx:
        txt = (d.get("content") or "").lower()
        if any(k in txt for k in a):
            hits.append(d)

    # If nothing matched, fallback to original (don’t accidentally wipe context)
    if not hits:
        return norm_ctx

    # Keep first max_chunks (usually context already roughly ranked)
    return hits[: max(1, max_chunks)]


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
def eval_correctness(
    judge: LLMJudge,
    tc: TestCase,
    relaxed: bool = False,
    tp_cov: float = 0.70,
    partial_cov: float = 0.30,
) -> Dict[str, Any]:
    metric = "correctness"
    if not tc.expected_steps or not tc.expected_steps.strip():
        raise ValueError("expected_steps (gold) must be provided for correctness.")
    if is_meta_response(tc.answer):
        return {metric: {"category": "FN", "coverage": 0.0, "coverage_1to5": 1, "error_severity": "low", "justification": "Meta-response / no answer.", "relaxed": relaxed}}

    raw = judge.chat(
        CORRECTNESS_PROMPT.format(
            question=tc.question,
            expected_steps=tc.expected_steps,
            actual_answer=tc.answer,
            tp_cov=tp_cov if not relaxed else 0.60,
            partial_cov=partial_cov if not relaxed else 0.20,
        ),
        tag=f"{metric}:judge",
    )

    obj = _extract_first_json_object(raw)
    try:
        validated = _require_fields(obj, [("category", str), ("coverage", float), ("error_severity", str), ("justification", str)])
    except Exception as e:
        return {metric: {"category": "FN", "coverage": 0.0, "coverage_1to5": 1, "error_severity": "high", "justification": f"Invalid judge JSON: {e}", "raw": preview(raw, 1200), "relaxed": relaxed}}

    cat = validated["category"].strip().upper()
    if cat not in {"TP", "PARTIAL", "FP", "FN"}:
        cat = "FN"
        validated["justification"] = f"Invalid category from judge; coerced to FN. Original: {validated['justification']}"

    cov = clamp01(float(validated["coverage"]))
    sev = str(validated["error_severity"]).strip().lower()
    if sev not in {"low", "medium", "high"}:
        sev = "high"

    out = dict(validated)
    out["category"] = cat
    out["coverage"] = cov
    out["coverage_1to5"] = to_1_5(cov)
    out["error_severity"] = sev
    out["relaxed"] = relaxed
    out["tp_cov"] = tp_cov if not relaxed else 0.60
    out["partial_cov"] = partial_cov if not relaxed else 0.20
    return {metric: out}


def aggregate_correctness(results: List[EvaluationResult]) -> Dict[str, float]:
    # treat PARTIAL as FP for precision/recall summary (optional)
    cats = []
    for r in results:
        c = str(r.metrics.get("correctness", {}).get("category", "")).upper()
        if c == "PARTIAL":
            c = "FP"
        cats.append(c)

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
# Metric 4: Faithfulness (positive-evidence biased)
# -----------------------------
def eval_faithfulness(judge: LLMJudge, tc: TestCase) -> Dict[str, Any]:
    metric = "faithfulness"

    norm_ctx = tc.context or []

    # Optional: anchor-focus to reduce truncation false negatives
    anchors: List[str] = []
    if FAITH_USE_ANCHOR_FOCUS and tc.answer and tc.answer.strip():
        rawA = judge.chat(GEN_ANCHORS_PROMPT.format(question=tc.question, answer=tc.answer, max_anchors=10), tag=f"{metric}:gen_anchors")
        objA = _extract_first_json_object(rawA)
        if isinstance(objA, dict) and isinstance(objA.get("anchors"), list):
            anchors = [str(x).strip() for x in objA["anchors"] if isinstance(x, str) and x.strip()][:10]
            norm_ctx = filter_context_by_anchors(norm_ctx, anchors, FAITH_ANCHOR_MAX_CHUNKS)

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
            anchor_focus=FAITH_USE_ANCHOR_FOCUS,
            anchors=anchors[:8],
            context_text_preview=preview(context_text, 500),
        )

    # Step 1: extract statements
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

    st_map = {str(i + 1): st for i, st in enumerate(statements)}
    st_block = "\n".join(f"{i}: {st}" for i, st in st_map.items())

    # Step 2: verify statements (V3 prompt)
    promptV = VERIFY_STATEMENTS_PROMPT_V3 if FAITH_POSITIVE_BIAS else VERIFY_STATEMENTS_PROMPT_V3
    raw2 = judge.chat(promptV.format(context=context_text, statements=st_block), tag=f"{metric}:verify_v3")
    obj2 = _extract_first_json_object(raw2)

    if not isinstance(obj2, dict):
        return {metric: {"score": 0.0, "error": "Invalid verification JSON (not a dict)", "raw": preview(raw2, 1200)}}

    # scoring: Yes=1, Partial=0.75, No=0.25
    verdict_score = {"yes": 1.0, "partial": 0.75, "no": 0.25}

    total = 0.0
    details: List[Dict[str, Any]] = []

    for key, judgement in obj2.items():
        if key not in st_map or not isinstance(judgement, dict) or "verdict" not in judgement:
            continue
        v = str(judgement["verdict"]).strip().lower()
        sc = verdict_score.get(v, 0.75 if FAITH_POSITIVE_BIAS else 0.25)
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
            "faith_max_statements": FAITH_MAX_STATEMENTS,
            "verdict_scoring": {"Yes": 1.0, "Partial": 0.75, "No": 0.25},
            "positive_bias": FAITH_POSITIVE_BIAS,
            "anchor_focus": FAITH_USE_ANCHOR_FOCUS,
            "anchor_max_chunks": FAITH_ANCHOR_MAX_CHUNKS,
            "anchors_used": anchors,
        }
    }


# -----------------------------
# Metric 5: Helpfulness
# -----------------------------
def eval_helpfulness(judge: LLMJudge, tc: TestCase, faithfulness_score01: float) -> Dict[str, Any]:
    metric = "helpfulness"
    if not tc.answer or not tc.answer.strip():
        return {metric: {"helpfulness_raw_1to5": 1, "helpfulness_raw_score": 0.0, "helpfulness_final_score": 0.0, "helpfulness_final_1to5": 1, "justification": "Empty answer"}}

    raw = judge.chat(HELPFULNESS_PROMPT.format(question=tc.question, actual_answer=tc.answer), tag=f"{metric}:judge")
    obj = _extract_first_json_object(raw)

    try:
        validated = _require_fields(obj, [("helpfulness_1to5", int), ("justification", str)])
    except Exception as e:
        return {metric: {"helpfulness_raw_1to5": 1, "helpfulness_raw_score": 0.0, "helpfulness_final_score": 0.0, "helpfulness_final_1to5": 1, "justification": f"Invalid judge JSON: {e}", "raw": preview(raw, 1200)}}

    h15 = max(1, min(5, int(validated["helpfulness_1to5"])))
    raw01 = score_1to5_to_01(h15)

    f01 = clamp01(float(faithfulness_score01))
    base = max(0.0, min(1.0, HELP_GATE_BASE))
    final01 = clamp01(raw01 * (base + (1.0 - base) * f01))

    return {
        metric: {
            "helpfulness_raw_1to5": h15,
            "helpfulness_raw_score": raw01,
            "helpfulness_final_score": final01,
            "helpfulness_final_1to5": to_1_5(final01),
            "faithfulness_used": f01,
            "gate_base": base,
            "justification": str(validated["justification"]),
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

        faith01 = float(metrics.get("faithfulness", {}).get("score", 0.0) or 0.0)
        metrics.update(eval_helpfulness(judge, tc, faithfulness_score01=faith01))

        out.append(EvaluationResult(script=tc.script, question_id=tc.question_id, query_type=tc.query_type, metrics=metrics))

        dt = time.time() - t0
        log(
            "Testcase END",
            idx=idx,
            seconds=round(dt, 3),
            answer_relevance=round(metrics.get("answer_relevance", {}).get("score", 0.0), 4),
            completeness=round(metrics.get("completeness", {}).get("overall_completeness_score", 0.0), 4)
            if isinstance(metrics.get("completeness", {}), dict)
            else None,
            correctness=metrics.get("correctness", {}).get("category"),
            correctness_cov=round(float(metrics.get("correctness", {}).get("coverage", 0.0) or 0.0), 4),
            correctness_cov_1to5=metrics.get("correctness", {}).get("coverage_1to5"),
            faithfulness=round(faith01, 4),
            faithfulness_1to5=to_1_5(faith01),
            helpful_raw_1to5=metrics.get("helpfulness", {}).get("helpfulness_raw_1to5"),
            helpful_final_1to5=metrics.get("helpfulness", {}).get("helpfulness_final_1to5"),
        )

    log("Finished evaluation", n_results=len(out))
    return out


def results_to_dataframe(results: List[EvaluationResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        row = {"script": r.script, "question_id": r.question_id, "query_type": r.query_type}

        ar = r.metrics.get("answer_relevance", {})
        ar01 = ar.get("score", None)
        row["answer_relevance_score"] = ar01
        row["answer_relevance_1to5"] = to_1_5(ar01) if ar01 is not None else None

        comp = r.metrics.get("completeness", {})
        comp01 = comp.get("overall_completeness_score", None) if isinstance(comp, dict) else None
        row["completeness_score"] = comp01
        row["completeness_1to5"] = to_1_5(comp01) if comp01 is not None else None

        corr = r.metrics.get("correctness", {})
        row["correctness_category"] = corr.get("category", None)
        row["correctness_coverage"] = corr.get("coverage", None)
        row["correctness_coverage_1to5"] = corr.get("coverage_1to5", None)
        row["correctness_error_severity"] = corr.get("error_severity", None)

        faith = r.metrics.get("faithfulness", {})
        faith01 = faith.get("score", None)
        row["faithfulness_score"] = faith01
        row["faithfulness_1to5"] = to_1_5(faith01) if faith01 is not None else None

        helpm = r.metrics.get("helpfulness", {})
        row["helpfulness_raw_1to5"] = helpm.get("helpfulness_raw_1to5", None)
        row["helpfulness_raw_score"] = helpm.get("helpfulness_raw_score", None)
        row["helpfulness_final_score"] = helpm.get("helpfulness_final_score", None)
        row["helpfulness_final_1to5"] = helpm.get("helpfulness_final_1to5", None)
        row["helpfulness_justification"] = helpm.get("justification", None)

        row["metrics_json"] = json.dumps(r.metrics, ensure_ascii=False)
        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_summary(results: List[EvaluationResult]) -> Dict[str, Any]:
    dfm = results_to_dataframe(results)
    summary: Dict[str, Any] = {
        "answer_relevance_mean": float(dfm["answer_relevance_score"].mean()) if "answer_relevance_score" in dfm else 0.0,
        "completeness_mean": float(dfm["completeness_score"].mean()) if "completeness_score" in dfm else 0.0,
        "faithfulness_mean": float(dfm["faithfulness_score"].mean()) if "faithfulness_score" in dfm else 0.0,
        "helpfulness_raw_mean": float(dfm["helpfulness_raw_score"].mean()) if "helpfulness_raw_score" in dfm else 0.0,
        "helpfulness_final_mean": float(dfm["helpfulness_final_score"].mean()) if "helpfulness_final_score" in dfm else 0.0,
    }
    summary.update(aggregate_correctness(results))
    return summary


# -----------------------------
# Entry (JSONL)
# -----------------------------
def main() -> None:
    PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT")).expanduser().resolve()
    ANSWERS_PATH = PROJECT_ROOT / "main" / "evaluation" / "results" / "answers_log_new_dataset_8_part3.jsonl"
    JSONL_IN = Path(os.getenv("ANSWERS_LOG_PATH", str(ANSWERS_PATH))).expanduser().resolve()

    log("Reading input JSONL", path=str(JSONL_IN))
    data = load_jsonl(JSONL_IN)
    log("Loaded JSONL objects", n=len(data))

    print("\n=== Judge Resume Mode (JSONL) ===")
    print("Row numbers are DATA rows in the JSONL (1 = first JSON line).")

    start_row_default = _env_int("JUDGE_START_ROW", 1)
    max_rows_default = _env_opt_int("JUDGE_MAX_ROWS")

    start_row = ask_int("Start evaluating from data row", default=start_row_default)
    max_rows = ask_optional_int("Evaluate at most N data rows (optional)")
    if max_rows is None:
        max_rows = max_rows_default

    if start_row < 1:
        start_row = 1

    n_total = len(data)
    start_idx = start_row - 1
    if start_idx >= n_total:
        log("[ERROR] start_row beyond dataset length", start_row=start_row, n_total=n_total)
        raise SystemExit(1)

    end_idx = n_total if max_rows is None else min(n_total, start_idx + max_rows)
    selected = data[start_idx:end_idx]

    log(
        "Resume selection applied",
        start_row=start_row,
        start_idx=start_idx,
        end_idx_exclusive=end_idx,
        selected_rows=len(selected),
        total_rows=n_total,
    )

    testcases: List[TestCase] = []
    for obj in selected:
        testcases.append(
            TestCase(
                script=str(obj.get("script", "")),
                question_id=str(obj.get("question_id", "")),
                query_type=str(obj.get("query_type", "")),
                question=str(obj.get("question", "")),
                answer=str(obj.get("answer", "")),
                expected_steps=str(obj.get("gold_answer", "")),
                expected_causes=None,
                context=normalize_context_items_from_jsonl(obj.get("context_items", [])),
            )
        )

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
        correctness_relaxed=True,
    )

    log("Aggregate summary", **aggregate_summary(results))

    out_df = results_to_dataframe(results)

    out_dir = PROJECT_ROOT / "main" / "evaluation" / "judge_results" 
    out_dir.mkdir(parents=True, exist_ok=True)

    CSV_OUT = out_dir / f"llm_judge_results_{JSONL_IN.stem}_rows_{start_row}_to_{end_idx}.csv"
    log("Writing output CSV", path=str(CSV_OUT), n_rows=len(out_df))
    out_df.to_csv(CSV_OUT, index=False)

    log("DONE")


if __name__ == "__main__":
    main()
