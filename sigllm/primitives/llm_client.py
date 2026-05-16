# -*- coding: utf-8 -*-

"""LLM Client abstraction.

This module provides a simple interface for custom LLM backends. Companies can
implement their own client by subclassing BaseLLMClient and implementing the
generate() method.

For OpenAI or HuggingFace models, use the existing gpt.py or huggingface.py
primitives directly. This module is for custom/internal LLM backends.

See tutorials/custom_llm_forecasting_pipeline.ipynb for a full example
with azure gpt.
"""

from typing import List, Optional


class BaseLLMClient:
    """Base class for custom LLM clients.

    Subclass this and implement generate() to integrate your own LLM backend.

    Example:
        class CustomLLM(BaseLLMClient):
            def __init__(self, llm_responder):
                self.llm_responder = llm_responder

            def generate(self, prompts, **kwargs):
                # Your function to generate responses from the LLM here
                return [[response] for response in self.llm_responder.complete(prompts)]

        client = CustomLLM()
        forecaster = CustomForecast(client=client, steps=5)
    """

    def generate(
        self,
        prompts: List[str],
        system_message: Optional[str] = None,
        max_tokens: int = 100,
        temperature: float = 1.0,
        n_samples: int = 1,
        **kwargs,
    ) -> List[List[str]]:
        """Generate responses for a batch of prompts.

        Args:
            prompts (List[str]):
                List of prompt strings to send to the LLM.
            system_message (str, optional):
                System message to prepend (for chat models). Default None.
            max_tokens (int):
                Maximum tokens to generate per response. Default 100.
            temperature (float):
                Sampling temperature. Default 1.0.
            n_samples (int):
                Number of responses to generate per prompt. Default 1.
            **kwargs:
                Additional provider-specific arguments.

        Returns:
            List[List[str]]:
                For each prompt, a list of n_samples response strings.
                Shape: (len(prompts), n_samples)
        """
        raise NotImplementedError(
            "Subclass BaseLLMClient and implement the generate() method."
        )
