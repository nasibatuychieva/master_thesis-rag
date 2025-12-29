import typing as t
import pandas as pd

from copilot_rag.tools.llm_judge.common import (
    validate_response,
    TestCase, EvaluationResult
)
from copilot_rag.tools.llm_judge.metrics.metric import Metric
from copilot_rag.databricks import DatabricksServicePrincipal

from copilot_rag.tools.llm_judge.metrics.prompts import (
    JUDGE_SINGLE_ANSWER_COMPLETENESS_PROMPT,
    JUDGE_MULTI_ANSWER_COMPLETENESS_PROMPT
)


class Completeness(Metric):
    """Evaluate procedure description responses."""

    def __init__(
            self,
            openai_spn: DatabricksServicePrincipal,
            openai_baseurl: str,
            openai_model_name: str,
            openai_embeddings_name: str,
            openai_temperature: float
    ):
        """Initialize a completeness score judge."""
        super().__init__(
            "completeness",
            openai_spn,
            openai_baseurl,
            openai_model_name,
            openai_embeddings_name,
            openai_temperature
        )

    def evaluate(self, context: TestCase) -> t.Dict[str, float]:
        """Evaluate correctness/completeness of procedure descriptions."""
        if len(context.expected_answers) == 1:
            # Expecting a single answer/troubleshooting instructions list.
            response = self._judge(
                JUDGE_SINGLE_ANSWER_COMPLETENESS_PROMPT.format(
                    question=context.question,
                    actual_answer=context.answer,
                    expected_answer=context.expected_answers[0],
                )
            )
            # Fields for single answer evaluation
            validation_fields = [
                ("total_actual_steps", int),
                ("total_expected_steps", int),
                ("wrong_step_count", int),
                ("missing_step_count", int),
                ("step_error_rate", float),
                ("step_error_rate_justification", str),
                ("step_omission_rate", float),
                ("step_omission_rate_justification", str),
                ("overall_completeness_score", float)
            ]
        elif len(context.expected_answers) == 2:
            # The first one is the troubleshooting instruction responses,
            # the second one are the expected problem causes list.
            response = self._judge(
                JUDGE_MULTI_ANSWER_COMPLETENESS_PROMPT.format(
                    question=context.question,
                    actual_answer=context.answer,
                    expected_causes=context.expected_answers[1],
                    expected_steps=context.expected_answers[0]
                )
            )
            # Fields for multi answer evaluation (includes causes)
            validation_fields = [
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
                ("overall_completeness_score", float)
            ]
        else:
            raise NotImplementedError("len(expected_answers) not in [1, 2]")

        return {
            "completeness": validate_response(
                response,
                fields=validation_fields
            )
        }

    def aggregate(self, results: t.List[EvaluationResult]) -> t.Dict:
        """Compute completeness over a set of results."""

        results = filter(lambda x: self.name in x.metrics, results)
        scores = pd.Series([
            res.metrics[self.name]["overall_completeness_score"]
            for res in results
        ])

        return {
            "completeness_score": scores.mean()
        }
