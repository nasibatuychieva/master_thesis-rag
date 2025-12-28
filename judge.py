import re
import json
from typing import Optional
from dataclasses import dataclass
import dataclasses

from openai import OpenAI
from copilot_rag.pipelines.log import Logger
from copilot_rag.tools.llm_judge.prompts import (
    PROCEDURE_JUDGE_PROMPT,
    SERVICE_REQUEST_JUDGE_PROMPT
)


@dataclass
class JudgeResult:
    """Result from LLM judge evaluation."""
    response_style: int = 0
    response_style_justification: str = ""
    correctness: str = ""
    correctness_justification: str = ""
    total_actual_steps: int = 0
    total_expected_steps: int = 0
    wrong_step_count: int = 0
    missing_step_count: int = 0
    step_error_rate: float = 0.0
    step_error_rate_justification: str = ""
    step_omission_rate: float = 0.0
    step_omission_rate_justification: str = ""
    total_actual_cause_count: Optional[int] = None
    total_expected_cause_count: Optional[int] = None
    wrong_cause_count: Optional[int] = None
    missing_cause_count: Optional[int] = None
    cause_error_rate: Optional[float] = None
    cause_error_rate_justification: Optional[str] = None
    cause_omission_rate: Optional[float] = None
    cause_omission_rate_justification: Optional[str] = None

    def __post_init__(self):
        """Calculate rates after initialization."""
        # Step error rate: percentage of steps in response that are wrong
        if self.total_actual_steps > 0:
            self.step_error_rate = float(
                self.wrong_step_count / self.total_actual_steps
            )
        else:
            self.step_error_rate = 0.0

        if self.total_expected_steps > 0:
            self.step_omission_rate = float(
                self.missing_step_count / self.total_expected_steps
            )
        else:
            self.step_omission_rate = 0.0

        if self.total_actual_cause_count and self.total_actual_cause_count > 0:
            if self.wrong_cause_count is not None:
                self.cause_error_rate = float(
                    self.wrong_cause_count / self.total_actual_cause_count
                )
        else:
            self.cause_error_rate = (
                0.0 if self.wrong_cause_count is not None else None
            )

        if (self.total_expected_cause_count and
                self.total_expected_cause_count > 0):
            if self.missing_cause_count is not None:
                self.cause_omission_rate = float(
                    self.missing_cause_count / self.total_expected_cause_count
                )
        else:
            self.cause_omission_rate = (
                0.0 if self.missing_cause_count is not None else None
            )

    @staticmethod
    def from_response(response: str) -> Optional['JudgeResult']:
        """Parse JSON response from LLM into JudgeResult.

        Args:
            response: Raw response string containing JSON

        Returns:
            JudgeResult instance or None if parsing fails
        """
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                Logger.error("No JSON found in response")
                return None

            json_str = json_match.group(0)
            data = json.loads(json_str)

            field_types = {
                field.name: field.type
                for field in dataclasses.fields(JudgeResult)
            }

            for field_name, field_type in field_types.items():
                if field_name in data:
                    try:
                        if field_type in (int, float):
                            if data[field_name] is not None:
                                data[field_name] = field_type(data[field_name])
                    except (ValueError, TypeError):
                        Logger.warning(f"Invalid value for {field_name}")
                        data[field_name] = JudgeResult().__dict__[field_name]

            return JudgeResult(**data)

        except Exception as e:
            Logger.error(f"Error parsing judge response: {str(e)}")
            return None


class LLMJudge:
    """Judge for evaluating LLM responses."""

    def __init__(
        self,
        judge_client: OpenAI,
        model_name: str
    ):
        """Initialize LLM judge.

        Args:
            judge_client: OpenAI client instance
            model_name: Name of model to use for judging
        """
        self._judge = judge_client
        self._model = model_name

    def judge_procedure(
        self,
        question: str,
        actual_answer: str,
        expected_answer: str,
    ) -> Optional[JudgeResult]:
        """Judge a procedure response."""
        try:
            completion = self._judge.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a response evaluator."
                    },
                    {
                        "role": "user",
                        "content": PROCEDURE_JUDGE_PROMPT.format(
                            question=question,
                            actual_answer=actual_answer,
                            expected_answer=expected_answer,
                        )
                    }
                ],
                model=self._model,
                response_format={"type": "json_object"},
                seed=42
            )

            return JudgeResult.from_response(
                completion.choices[0].message.content.strip()
            )

        except Exception as e:
            Logger.error(f"Error in procedure judge: {str(e)}")
            return None

    def judge_service_request(
        self,
        question: str,
        answer: str,
        expected_causes: str,
        expected_steps: str,
    ) -> Optional[JudgeResult]:
        """Judge a service request response."""
        try:
            completion = self._judge.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a response evaluator."
                    },
                    {
                        "role": "user",
                        "content": SERVICE_REQUEST_JUDGE_PROMPT.format(
                            question=question,
                            actual_answer=answer,
                            expected_causes=expected_causes,
                            expected_steps=expected_steps
                        )
                    }
                ],
                model=self._model,
                response_format={"type": "json_object"},
                seed=42
            )

            return JudgeResult.from_response(
                completion.choices[0].message.content.strip()
            )

        except Exception as e:
            Logger.error(f"Error in service request judge: {str(e)}")
            return None
