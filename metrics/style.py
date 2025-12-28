import typing as t

from copilot_rag.tools.llm_judge.common import (
    validate_response,
    TestCase,
)
from copilot_rag.tools.llm_judge.metrics.metric import Metric
from copilot_rag.databricks import DatabricksServicePrincipal


JUDGE_STYLE_PROMPT = """
You are an expert evaluator tasked with assessing the format
of the responses from a retrieval-augmented generation (RAG) pipeline.
Evaluate the response based on the following framework.

## Inputs:

### Question:
{question}

### Expected Answer:
{expected_answer}

### Actual Answer:
{actual_answer}

Evaluate the response style as follows:

### Evaluation framework:

**For apology responses or inability to answer:**
When a response cannot be provided, the bot responds with a message
saying that it has searched all available documentation but failed
to find relevant information.

**For other responses:**
The response should conform to the following guidelines:
- It should use bullet points and text formatting when listing
steps or instructions
- The answer should be in the language of the prompt
- The language used should be technical.
- Clarity and coherence - the response is logically structured
and easy to follow
- Completeness of formatting for the type of content provided

The response style score is a number from 1 to 5 that indicates how well
the response adheres to the appropriate guidelines for its response type.

### Evaluation process
1. First determine if this is an apology/inability response or
a substantive answer.
2. If an apology response was provided, assign score 5. For other responses,
apply the appropriate evaluation criteria.
3. Consider the helpfulness and professionalism of the response.
4. Score accordingly and provide detailed justification.

Output a valid JSON containing:
  - score: the response style score.
  - justification: the justification for the response style score, explaining
    which evaluation criteria were applied and why the specific score was
    given.
"""


class ResponseStyle(Metric):
    """Evaluate procedure description responses."""

    def __init__(
            self,
            openai_spn: DatabricksServicePrincipal,
            openai_baseurl: str,
            openai_model_name: str,
            openai_embeddings_name: str,
            openai_temperature: float
    ):
        """Initialize a style judge."""
        super().__init__(
            "style",
            openai_spn,
            openai_baseurl,
            openai_model_name,
            openai_embeddings_name,
            openai_temperature
        )

    def evaluate(self, context: TestCase) -> t.Dict[str, float]:
        """Evaluate correctness/completeness of procedure descriptions."""
        if not context.expected_answers:
            raise ValueError("no expected answer specified")

        result = self._judge(
            JUDGE_STYLE_PROMPT.format(
                question=context.question,
                actual_answer=context.answer,
                expected_answer=context.expected_answers[0],
            )
        )
        return {
            "style": validate_response(
                result,
                fields=[
                    ("score", int),
                    ("justification", str)
                ])
        }
