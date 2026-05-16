# -*- coding: utf-8 -*-

"""Custom LLM anomaly detection primitive.

This module provides an anomaly detection primitive that works with any LLM
backend through the BaseLLMClient interface. Companies can plug in their own
LLM by implementing a simple client class.

For OpenAI or HuggingFace, use the existing gpt.py or huggingface.py primitives.
This module is for custom/internal LLM backends.

Example usage:
    from sigllm.primitives.llm_client import BaseLLMClient
    from sigllm.primitives.prompting.custom import CustomDetect

    class CustomLLM(BaseLLMClient):
        def generate(self, prompts, **kwargs):
            # Your function to generate responses from the LLM here
            return [[response] for response in llm_client.complete(prompts)]

    client = CustomLLM()
    detector = CustomDetect(client=client, samples=5)
    anomalies = detector.detect(X_strings)
"""

import json
import os
from typing import List

from sigllm.primitives.llm_client import BaseLLMClient


PROMPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gpt_messages.json')
PROMPTS = json.load(open(PROMPT_PATH))


class CustomDetect:
    """Detect anomalies in time series using a custom LLM client.

    This primitive wraps any LLM backend (via BaseLLMClient) to perform
    anomaly detection. It handles prompt construction and delegates the
    actual LLM calls to the provided client.

    For OpenAI or HuggingFace, use gpt.py or huggingface.py instead.

    Args:
        client (BaseLLMClient):
            Your custom LLM client instance that implements generate().
        sep (str):
            Separator between values in the sequence. Default ",".
        anomalous_percent (float):
            Expected percentage of time series that are anomalous.
            Used to estimate response length. Default 0.5.
        temp (float):
            Sampling temperature. Default 1.0.
        samples (int):
            Number of detection samples per input. Default 10.

    Example:
        class MyLLM(BaseLLMClient):
            def generate(self, prompts, **kwargs):
                return [[my_api.complete(p)] for p in prompts]

        client = MyLLM()
        detector = CustomDetect(client=client, samples=5)
        anomalies = detector.detect(X)
    """

    def __init__(
        self,
        client: BaseLLMClient,
        sep: str = ",",
        anomalous_percent: float = 0.5,
        temp: float = 1.0,
        samples: int = 10,
    ):
        if not isinstance(client, BaseLLMClient):
            raise TypeError(
                f"client must be a BaseLLMClient instance, got {type(client)}. "
                "For OpenAI, use sigllm.primitives.prompting.gpt.GPT instead."
            )

        self.client = client
        self.sep = sep
        self.anomalous_percent = anomalous_percent
        self.temp = temp
        self.samples = samples

    def _build_prompt(self, sequence: str) -> str:
        """Build the detection prompt from a sequence string."""
        return " ".join([PROMPTS["user_message"], sequence, self.sep])

    def _estimate_max_tokens(self, sequence: str) -> int:
        """Estimate max tokens needed for the detection response."""
        seq_len = len(sequence)
        return int(seq_len * self.anomalous_percent) + 20

    def detect(self, X: List[str], **kwargs) -> List[List[str]]:
        """Detect anomalies in each input sequence.

        Args:
            X (List[str] or ndarray):
                Input sequences as strings. Each string is a comma-separated
                sequence of numeric values.
            **kwargs:
                Additional arguments passed to the LLM client.

        Returns:
            List[List[str]]:
                For each input sequence, a list of detection response strings.
        """
        prompts = [self._build_prompt(seq) for seq in X]
        max_tokens = self._estimate_max_tokens(X[0]) if len(X) > 0 else 100

        responses = self.client.generate(
            prompts=prompts,
            system_message=PROMPTS["system_message"],
            max_tokens=max_tokens,
            temperature=self.temp,
            n_samples=self.samples,
            **kwargs,
        )

        return responses
