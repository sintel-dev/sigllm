# -*- coding: utf-8 -*-

import json
import os

import openai
import tiktoken

PROMPT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'gpt_messages.json'
)

PROMPTS = json.load(open(PROMPT_PATH))

VALID_NUMBERS = list("0123456789 ")
BIAS = 30


class GPT:
    """Prompt GPT models to detect anomalies in a time series.

    Args:
        name (str):
            Model name. Default to `'gpt-3.5-turbo'`.
        sep (str):
            String to separate each element in values. Default to `','`.
    """

    def __init__(self, name='gpt-3.5-turbo', sep=','):
        self.name = name
        self.sep = sep

        self.tokenizer = tiktoken.encoding_for_model(self.name)

        valid_tokens = []
        for number in VALID_NUMBERS:
            token = self.tokenizer.encode(number)
            valid_tokens.append(token)

        valid_tokens.append(self.tokenizer.encode(self.sep))
        self.logit_bias = {token: BIAS for token in valid_tokens}

    def detect(self, text, anomalous_percent = 0.5, temp=1, top_p=1, logprobs=False, top_logprobs=None,
                 samples=10, seed=None):
        """Use GPT to forecast a signal.

        Args:
            text (str):
                A string containing signal values.
            anomalous_percent (float): 
                Expected percentage of time series that are anomalous. Default to `0.5`.
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
                Number of responses to generate for each input message. Default to `10`.
            seed (int):
                Beta feature by OpenAI to sample deterministically. Default to `None`.

        Returns:
            list, list:
                * List of detected anomalous values.
                * Optionally, a list of the output tokens' log probabilities.
        """
        input_length = len(self.tokenizer.encode(text))
        max_tokens = input_length * anomalous_percent

        message = ' '.join(PROMPTS['user_message'], text, self.sep)
        response = openai.ChatCompletion.create(
            model=self.name,
            messages=[
                {"role": "system", "content": PROMPTS['system_message']},
                {"role": "user", "content": message}
            ],
            max_tokens=max_tokens,
            temperature=temp,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            n=samples,
        )
        responses = [choice.message.content for choice in response.choices]
        if logprobs:
            probs = [choice.logprobs for choice in response.choices]
            return responses, probs

        return responses




























import os

from openai import OpenAI


def load_system_prompt(file_path):
    with open(file_path) as f:
        system_prompt = f.read()
    return system_prompt


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

ZERO_SHOT_FILE = 'gpt_system_prompt_zero_shot.txt'
ONE_SHOT_FILE = 'gpt_system_prompt_one_shot.txt'

ZERO_SHOT_DIR = os.path.join(CURRENT_DIR, "..", "template", ZERO_SHOT_FILE)
ONE_SHOT_DIR = os.path.join(CURRENT_DIR, "..", "template", ONE_SHOT_FILE)


GPT_model = "gpt-4"  # "gpt-4-0125-preview" #  #  #"gpt-3.5-turbo" #
client = OpenAI()


def get_gpt_model_response(message, gpt_model=GPT_model):
    completion = client.chat.completions.create(
        model=gpt_model,
        messages=message,
    )
    return completion.choices[0].message.content


def create_message_zero_shot(seq_query, system_prompt_file=ZERO_SHOT_DIR):
    messages = []

    messages.append({"role": "system", "content": load_system_prompt(system_prompt_file)})

    # final prompt
    messages.append({"role": "user", "content": f"Sequence: {seq_query}"})
    return messages


def create_message_one_shot(seq_query, seq_ex, ano_idx_ex, system_prompt_file=ONE_SHOT_DIR):
    messages = []

    messages.append({"role": "system", "content": load_system_prompt(system_prompt_file)})

    # one shot
    messages.append({"role": "user", "content": f"Sequence: {seq_ex}"})
    messages.append({"role": "assistant", "content": ano_idx_ex})

    # final prompt
    messages.append({"role": "user", "content": f"Sequence: {seq_query}"})
    return messages
