import typing as t

from copilot_rag.pipelines.log import Logger
from copilot_rag.tools.llm_judge.common import (
    validate_response,
    TestCase,
)
from copilot_rag.tools.llm_judge.metrics.metric import Metric
from copilot_rag.databricks import DatabricksServicePrincipal


GEN_STATEMENT_PROMPT = """
Given a question and answer, create one or more statements from
each sentence in the given answer.

**Question**:
{question}

**Answer**:
{answer}

**Important Note**
If the answer is refering to not being able to answer the question
and checking the question again, generate a list containing a
single statement, the string 'I am sorry.'.

Respond with a JSON list of statements, each statement being a string.
Use the key "statements" for the JSON object.
"""

VERIFY_STATEMENTS_PROMPT = """
Consider the information in **Context** and the statements in **Statements**.
Determine whether each statement is supported by the context.

1. Provide a brief explanation for each statement.
2. Give a verdict of "Yes" or "No" based on whether the statement is supported by the context.
3. Compile all results in the JSON format shown in **Example Output**.
4. **Do not deviate** from the specified JSON structure and keys.

**Important Note**
- If the context is empty, provide the verdict "Yes" only for statements referencing an apology for not being able to answer.
- If the context is not empty, evaluate each statement based on the context provided.

**Context**
{context}

**Statements**
{statements}

**Example Output**
{{
  1: {{
     "verdict": "Yes",
     "explanation": "The troubleshooting instruction can be found in the context."
  }},
  2: {{
     "verdict": "Yes",
     "explanation": "The safety instructions specifically forbid the user from working on the electrical system."
  }},
  3: {{
     "verdict": "No",
     "explanation": "There is no information in the context to support this claim."
  }}
}}

---
Now judge the statements carefully and provide your verdicts.
"""  # noqa


class Faithfulness(Metric):
    """Evaluate answer faithfulness scores.

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
            openai_temperature: float
    ):
        """Initialize a correctness judge."""
        super().__init__(
            "faithfulness",
            openai_spn,
            openai_baseurl,
            openai_model_name,
            openai_embeddings_name,
            openai_temperature
        )

    def evaluate(self, test_case: TestCase) -> t.Dict[str, float]:
        """Evaluate correctness/completeness of procedure descriptions."""
        if not test_case.expected_answers:
            raise ValueError("no expected answer specified")

        prompt = GEN_STATEMENT_PROMPT.format(
            question=test_case.question,
            answer=test_case.answer
        )

        statement_response = self._judge(prompt)
        statements = validate_response(
            statement_response,
            fields=[
                ("statements", list)
            ])["statements"]
        Logger.info(
            "generated statements",
            statements=statements
        )
        if not statements:
            return {}

        statements = {str(i+1): st for i, st in enumerate(statements)}

        prompt = VERIFY_STATEMENTS_PROMPT.format(
            context="\n".join(doc["content"] for doc in test_case.context),
            statements="\n".join(
                f"{i}: {st}" for i, st in statements.items()
            )
        )

        response = self._judge(prompt)

        json_response = validate_response(response)
        if not isinstance(json_response, dict):
            Logger.error("Invalid verify statements response.")
            return {}

        n_yes = 0
        nays = []

        for key, judgement in json_response.items():
            if (
                    not isinstance(judgement, dict) or
                    "verdict" not in judgement
            ):
                Logger.error(
                    "Invalid verdict.",
                    key=key,
                    verdict=judgement
                )
                continue
            if judgement["verdict"].lower() == "yes":
                n_yes += 1
            else:
                nays.append({
                    "statement": statements[key],
                    "judgement": judgement
                })
        return {
            "faithfulness": {
                "score": n_yes / len(statements),
                "unsupported_statements": nays
            }
        }
