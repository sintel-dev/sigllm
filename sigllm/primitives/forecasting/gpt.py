# -*- coding: utf-8 -*-

import json
import os

import openai
import tiktoken
from openai import OpenAI
from tqdm import tqdm

PROMPT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'TEMPLATE.json'
)

PROMPTS = json.load(open(PROMPT_PATH))

VALID_NUMBERS = list("0123456789 ")
BIAS = 30


class GPT:
    """Prompt GPT models to forecast a time series.

    Args:
        name (str):
            Model name. Default to `'gpt-3.5-turbo'`.
        chat (bool):
            Whether you're using a chat model or not. Default to `True`.
        sep (str):
            String to separate each element in values. Default to `','`.
        steps (int):
            Number of steps ahead to forecast. Default `1`.
        temp (float):
            Sampling temperature to use, between 0 and 2. Higher values like 0.8 will
            make the output more random, while lower values like 0.2 will make it
            more focused and deterministic. Do not use with `top_p`. Default to `1`.
        top_p (float):
            Alternative to sampling with temperature, called nucleus sampling, where the
            model considers the results of the tokens with top_p probability mass.
            So 0.1 means only the tokens comprising the top 10% probability mass are
            considered. Do not use with `temp`. Default to `1`.
        logprobs (bool):
            Whether to return the log probabilities of the output tokens or not.
            Defaults to `False`.
        top_logprobs (int):
            An integer between 0 and 20 specifying the number of most likely tokens
            to return at each token position. Default to `None`.
        samples (int):
            Number of forecasts to generate for each input message. Default to `1`.
        seed (int):
            Beta feature by OpenAI to sample deterministically. Default to `None`.
    """

    def __init__(self, name='gpt-3.5-turbo', chat=True, sep=',', steps=1, temp=1,
                 top_p=1, logprobs=False, top_logprobs=None, samples=1, seed=None):
        self.name = name
        self.chat = chat
        self.sep = sep
        self.steps = steps
        self.temp = temp
        self.top_p = top_p
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs
        self.samples = samples
        self.seed = seed

        self.tokenizer = tiktoken.encoding_for_model(self.name)

        valid_tokens = []
        for number in VALID_NUMBERS:
            token = self.tokenizer.encode(number)
            valid_tokens.extend(token)

        valid_tokens.extend(self.tokenizer.encode(self.sep))
        self.logit_bias = {token: BIAS for token in valid_tokens}

        self.client = OpenAI()

    def forecast(self, X, **kwargs):
        """Use GPT to forecast a signal.

        Args:
            X (ndarray):
                Input sequences of strings containing signal values.

        Returns:
            list, list:
                * List of forecasted signal values.
                * Optionally, a list of the output tokens' log probabilities.
        """
        all_responses, all_probs = [], []
        for text in tqdm(X):
            input_length = len(self.tokenizer.encode(text))
            average_length = (input_length + 1) // len(text.split(','))
            max_tokens = average_length * self.steps

            if self.chat:
                message = ' '.join([PROMPTS['user_message'], text, self.sep])
                response = self.client.chat.completions.create(
                    model=self.name,
                    messages=[
                        {"role": "system", "content": PROMPTS['system_message']},
                        {"role": "user", "content": message}
                    ],
                    max_completion_tokens=max_tokens,
                    temperature=self.temp,
                    top_p=self.top_p,
                    logprobs=self.logprobs,
                    top_logprobs=self.top_logprobs,
                    n=self.samples,
                    seed=self.seed
                )
                responses = [choice.message.content for choice in response.choices]

            else:
                message = ' '.join([text, self.sep])
                response = openai.Completion.create(
                    model=self.name,
                    prompt=message,
                    max_tokens=max_tokens,
                    temperature=self.temp,
                    top_p=self.top_p,
                    logprobs=self.logprobs,
                    top_logprobs=self.top_logprobs,
                    logit_bias=self.logit_bias,
                    n=self.samples,
                    seed=self.seed
                )
                responses = [choice.text for choice in response.choices]

            if self.logprobs:
                probs = [choice.logprobs for choice in response.choices]
                all_probs.append(probs)

            all_responses.append(responses)

        if self.logprobs:
            return all_responses, all_probs

        return all_responses
