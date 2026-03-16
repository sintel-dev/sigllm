"""Custom LLM forecasting primitive.

This module provides a forecasting primitive that works with any LLM backend
through the BaseLLMClient interface. Companies can plug in their own LLM
by implementing a simple client class.

For OpenAI or HuggingFace, use the existing gpt.py or huggingface.py primitives.
This module is for custom/internal LLM backends.

Example usage:
    from sigllm.primitives.llm_client import BaseLLMClient
    from sigllm.primitives.forecasting.custom import CustomForecast

    class CustomLLM(BaseLLMClient):
        def generate(self, prompts, **kwargs):
            # Your function to generate responses from the LLM here
            return [[response] for response in llm_client.complete(prompts)]

    client = CustomLLM()
    forecaster = CustomForecast(client=client, steps=5)
    predictions = forecaster.forecast(X_strings)
"""

import json
import os
from typing import List

import numpy as np

from sigllm.primitives.llm_client import BaseLLMClient

PROMPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TEMPLATE.json')
PROMPTS = json.load(open(PROMPT_PATH))


class CustomForecast:
    """Forecast time series using a custom LLM client.

    This primitive wraps any LLM backend (via BaseLLMClient) to perform
    time series forecasting. It handles prompt construction and response
    parsing, while delegating the actual LLM calls to the provided client.

    For OpenAI or HuggingFace, use gpt.py or huggingface.py instead.

    Args:
        client (BaseLLMClient):
            Your custom LLM client instance that implements generate().
        sep (str):
            Separator between values in the sequence. Default ",".
        steps (int):
            Number of steps ahead to forecast. Default 1.
        temp (float):
            Sampling temperature. Default 1.0.
        samples (int):
            Number of forecast samples per input. Default 1.

    Example:
        class MyLLM(BaseLLMClient):
            def generate(self, prompts, **kwargs):
                return [[my_api.complete(p)] for p in prompts]
        client = MyLLM()
        forecaster = CustomForecast(client=client, steps=5)
        predictions = forecaster.forecast(X)
    """

    def __init__(
        self,
        client: BaseLLMClient,
        sep: str = ",",
        steps: int = 1,
        temp: float = 1.0,
        samples: int = 1,
    ):
        if not isinstance(client, BaseLLMClient):
            raise TypeError(f"client must be a BaseLLMClient instance, got {type(client)}.")

        self.client = client
        self.sep = sep
        self.steps = steps
        self.temp = temp
        self.samples = samples

    def _build_prompt(self, sequence: str) -> str:
        """Build the forecasting prompt from a sequence string."""
        return " ".join([PROMPTS["user_message"], sequence, self.sep])

    def _estimate_max_tokens(self, sequence: str) -> int:
        """Estimate max tokens needed for the forecast."""
        values = sequence.split(self.sep)
        if len(values) > 0:
            avg_len = sum(len(v.strip()) for v in values) / len(values)
            return int((avg_len + 2) * self.steps) + 10
        return 5

    def forecast(self, X: List[str], **kwargs) -> List[List[str]]:
        """Forecast future values for each input sequence.

        Args:
            X (List[str] or ndarray):
                Input sequences as strings. Each string is a comma-separated
                sequence of numeric values.
            **kwargs:
                Additional arguments passed to the LLM client.

        Returns:
            List[List[str]]:
                For each input sequence, a list of forecast strings.
        """
        prompts = [self._build_prompt(seq) for seq in X]
        max_tokens = self._estimate_max_tokens(X[0]) if len(X) > 0 else 5

        responses = self.client.generate(
            prompts=prompts,
            system_message=PROMPTS["system_message"],
            max_tokens=max_tokens,
            temperature=self.temp,
            n_samples=self.samples,
            **kwargs,
        )

        return responses


def rolling_window_sequences(X, y, index, window_size, target_size, step_size, target_column):
    """Create rolling window sequences out of time series data.

    The function creates an array of input sequences and an array of target sequences by rolling
    over the input sequence with a specified window.
    Optionally, certain values can be dropped from the sequences.

    Args:
        X (ndarray):
            N-dimensional input sequence to iterate over.
        y (ndarray):
            N-dimensional target sequence to iterate over.
        index (ndarray):
            Array containing the index values of X.
        window_size (int):
            Length of the input sequences.
        target_size (int):
            Length of the target sequences.
        step_size (int):
            Indicating the number of steps to move the window forward each round.
        target_column (int):
            Indicating which column of X is the target.

    Returns:
        ndarray, ndarray, ndarray, ndarray:
            * input sequences.
            * target sequences.
            * first index value of each input sequence.
            * first index value of each target sequence.
    """
    out_X = list()
    out_y = list()
    X_index = list()
    y_index = list()
    target = y[:, target_column]

    start = 0
    max_start = len(X) - window_size - target_size + 1
    while start < max_start:
        end = start + window_size

        out_X.append(X[start:end])
        out_y.append(target[end : end + target_size])
        X_index.append(index[start])
        y_index.append(index[end])
        start = start + step_size

    return np.asarray(out_X), np.asarray(out_y), np.asarray(X_index), np.asarray(y_index)
