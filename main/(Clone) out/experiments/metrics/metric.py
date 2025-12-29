import openai
import backoff

import typing as t
from collections import defaultdict
from abc import ABC, abstractmethod

from copilot_rag.tools.llm_judge.common import (
    TestCase,
    EvaluationResult
)
from copilot_rag.databricks import DatabricksServicePrincipal


class Metric(ABC):
    """Interface for a metric evaluator."""

    def __init__(
            self,
            name: str,
            openai_spn: DatabricksServicePrincipal,
            openai_baseurl: str,
            openai_llm_name: str,
            openai_embeddings_name: str,
            openai_temperature: t.Optional[float] = 1.0
    ):
        """Initialize a new evaluator.

        :param openai_spn: Databricks service principal for authentication.
        :param openai_baseurl: Base URL for OpenAI API.
        :param openai_llm_name: Judge model name.
        :param openai_embeddings_name: Embedding model name.
        :param openai_temperature: Sampling temperature
        """
        if not openai_spn:
            raise ValueError("openai_spn is None")
        if not openai_baseurl:
            raise ValueError("openai_baseurl is None")
        if not openai_llm_name:
            raise ValueError("openai_llm_name is None")

        self._name = name
        self._openai_spn = openai_spn
        self._openai_baseurl = openai_baseurl
        self._llm_name = openai_llm_name
        self._embedding_model = openai_embeddings_name
        self._temperature = openai_temperature

    def _get_openai_client(self) -> openai.OpenAI:
        """Get OpenAI client with validated token."""
        # Check if token is valid, this will refresh if needed
        if not self._openai_spn.authenticated:
            # Force token refresh by accessing access_token
            _ = self._openai_spn.access_token

        return openai.OpenAI(
            api_key=self._openai_spn.access_token,
            base_url=self._openai_baseurl
        )

    @property
    def name(self):
        """Metric name."""
        return self._name

    @backoff.on_exception(backoff.expo, openai.APIError, max_tries=3)
    def _judge(self, prompt: str):
        client = self._get_openai_client()
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a response evaluator."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model=self._llm_name,
            temperature=self._temperature,
            response_format={"type": "json_object"},
            seed=42
        )
        return response.choices[0].message.content.strip()

    @backoff.on_exception(backoff.expo, openai.OpenAIError, max_tries=3)
    def _embed(self, text: str):
        client = self._get_openai_client()
        return client.embeddings.create(
            input=text,
            model=self._embedding_model
        ).data[0].embedding

    @abstractmethod
    def evaluate(self, test_case: TestCase) -> t.Dict[str, float]:
        """Evaluate a single test case.

        :param TestCase test_case: contains the information required
        for judging the user question.
        """
        ...

    def aggregate(
            self, results: t.List[EvaluationResult]
    ) -> t.Dict[str, float]:
        """Compute the metric value over a set of results.

        :param results: a list of EvaluationResult.
        :return:
        """
        sums = defaultdict(float)
        counts = defaultdict(int)

        # collect evaluation results for this metric
        m_res = [res.metrics.get(self.name, {}) for res in results]

        # compute averages over dimensions
        for res in m_res:
            for dim, val in res.items():
                if isinstance(val, (int, float)):
                    sums[dim] += val
                    counts[dim] += 1

        # return averages over dimensions
        return {
            dim_name: sums[dim_name] / counts[dim_name]
            for dim_name in sums
        }
