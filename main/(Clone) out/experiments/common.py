import re
import json
import typing as t
from dataclasses import dataclass, field

from copilot_rag.pipelines.log import Logger


@dataclass
class TestCase:
    """Encapsulate information for response evaluation."""

    case_id: str
    question: str
    category: str
    step: int = 1

    answer: str = ""
    machine_type: str = ""
    expected_docs: t.List[str] = field(default_factory=list)
    expected_pages: t.List[str] = field(default_factory=list)
    expected_answers: t.List[str] = field(default_factory=list)
    context: str = ""


@dataclass
class EvaluationResult:
    """Results from evaluating a RAG system's responses using LLM judge."""

    case_id: str
    step: int
    machine_type: str
    category: str
    actual_answer: str
    prompt: str
    expected_answer: str

    retrieved_documents: t.List[str] = field(default_factory=list)
    document_precision: t.Optional[float] = None
    document_omission_rate: t.Optional[float] = None
    page_precision: t.Dict[str, float] = field(default_factory=dict)
    expected_docs: t.List[str] = field(default_factory=list)
    expected_pages: t.List[str] = field(default_factory=list)
    actual_pages: t.List[t.Dict] = field(default_factory=list)

    durations: t.Dict[str, float] = field(default_factory=dict)
    metrics: t.Dict[str, t.Dict[str, t.Any]] = field(default_factory=dict)


def validate_response(
        response: str,
        fields: t.List[t.Tuple[str, t.Type]] = []
) -> t.Dict:
    """Parse and validate LLM response JSON.

    :param str response: the raw LLM response.
    :param fields: a list of touples representing
    key names and expected Python type.
    :return: a dictionary containing extracted fields.
    """
    try:
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if not json_match:
            Logger.error("No JSON found in response")
            return None

        json_str = json_match.group(0)
        data = json.loads(json_str)

        for field_name, field_type in fields:
            if field_name in data:
                try:
                    if field_type in (int, float):
                        if data[field_name] is not None:
                            data[field_name] = field_type(data[field_name])
                except (ValueError, TypeError):
                    Logger.warning(f"Invalid value for {field_name}",
                                   value=data[field_name])
                    data[field_name] = field_type()

        return data
    except Exception as e:
        Logger.error(f"Error validating response: {e}")
        return {}


def is_apology_response(answer: str) -> bool:
    """Detect if a response is an apology/sorry response that should
    skip certain metrics.

    Args:
        answer: The AI response text to check

    Returns:
        True if this is an apology response, False otherwise
    """
    answer_lower = answer.lower().strip()

    # Common apology patterns
    apology_patterns = [
        "i'm sorry",
        "i am sorry",
        "sorry",
        "i cannot answer",
        "i can't answer",
        "i don't have",
        "i do not have",
        "no relevant information was found",
        "no information was found",
        "i have searched through all",
        "i have searched all",
        "unable to find",
        "cannot find",
        "insufficient information",
        "not enough information",
        "no available information"
    ]

    # Check if any apology pattern is present
    for pattern in apology_patterns:
        if pattern in answer_lower:
            return True

    # Additional check: very short responses that might be apologies
    if (len(answer_lower) < 100 and
            any(word in answer_lower for word in [
                "sorry", "cannot", "can't", "don't", "unable"
            ])):
        return True

    return False
