import typing as t

import numpy as np

from copilot_rag.pipelines.log import Logger
from copilot_rag.tools.llm_judge.common import (
    validate_response,
    TestCase,
)
from copilot_rag.tools.llm_judge.metrics.metric import Metric
from copilot_rag.databricks import DatabricksServicePrincipal

GEN_QUESTION_PROMPT = """

You are given an answer. Generate one or more questions that could
reasonably be answered with this answer.

**Input Answer:**
{answer}

**Instructions:**

- Only generate questions that this answer could directly respond to.

- If the answer indicates uncertainty, refusal, or suggests reviewing
  the original question (e.g., "I cannot answer that", "Please check
  the question again"), then **do not** generate any questions.

- Your response should be a JSON object in the following format:

```json
{{
  "questions": [
    "Your first generated question?",
    "Your second generated question?"
  ]
}}
```

If no questions can be generated, return:

```json
{{
  "questions": []
}}
```

**Example Output:**

```json
{{
  "questions": [
    "What is the procedure for replacing the pump?",
    "How many nm to use when tightening the screw?"
  ]
}}
```
"""


class AnswerRelevance(Metric):
    """Compute answer relevance scores.

    For details see:
    Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2023).
    RAGAS: Automated Evaluation of Retrieval Augmented Generation
    (arXiv:2309.15217). arXiv. https://doi.org/10.48550/arXiv.2309.15217
    """

    def __init__(
            self,
            openai_spn: DatabricksServicePrincipal,
            openai_baseurl: str,
            openai_model_name: str,
            openai_embeddings_name: str,
            openai_temperature: float,
            num_questions: int = 3,
            similarity_thresh: float = 0.8
    ):
        """Initialize a correctness judge."""
        super().__init__(
            "answer_relevance",
            openai_spn,
            openai_baseurl,
            openai_model_name,
            openai_embeddings_name,
            openai_temperature
        )

        if num_questions <= 0:
            raise ValueError("num_questions")
        self._num_questions = num_questions
        self._score_thresh = similarity_thresh

    def evaluate(self, test_case: TestCase) -> t.Dict[str, float]:
        """Evaluate correctness/completeness of procedure descriptions."""
        if not test_case.expected_answers:
            raise ValueError("no expected answer specified")

        # Handle empty or invalid answers
        if not test_case.answer or test_case.answer.strip() == "":
            return {
                "answer_relevance": {
                    "score": 0.0,
                    "reason": "Empty answer"
                }
            }

        prompt = GEN_QUESTION_PROMPT.format(
            answer=test_case.answer,
            num_questions=self._num_questions
        )
        questions_response = self._judge(prompt)
        questions = validate_response(
            questions_response,
            fields=[
                ("questions", list)
            ])["questions"]

        Logger.info(
            "Answer relevance: generated questions",
            questions=questions
        )
        if not questions:
            return {
                "answer_relevance": {
                    "score": 0.0,
                    "reason": "No questions generated"
                }
            }

        # r_emb and r_mag are the embedding vector
        # and the embedding vector magnitude for the
        # sample question.
        r_emb = self._embed(test_case.question)
        r_mag = np.linalg.norm(r_emb)

        # q_emb and q_mag are lists of embeddings and
        # magnitudes for generated questions for which
        # we test the similarity.
        q_emb = [self._embed(q) for q in questions]
        q_mag = [np.linalg.norm(qe) for qe in q_emb]

        cos_scores = [
            np.dot(r_emb, qe) / (r_mag * qm)
            for qe, qm in zip(q_emb, q_mag)
        ]

        num_similar = sum(
            cos_score > self._score_thresh
            for cos_score in cos_scores
        )

        if num_similar != len(questions):
            dissimilar_questions = [
                (q, s) for q, s in zip(questions, cos_scores)
                if s < self._score_thresh
            ]
            Logger.info(
                "Found dissimilar questions that affect "
                "the answer relevance score.",
                questions=dissimilar_questions
            )

        return {
            "answer_relevance": {
                "score": float(num_similar / len(questions))
            }
        }
