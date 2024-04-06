# -*- coding: utf-8 -*-

import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LOGGER = logging.getLogger(__name__)

DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_PAD_TOKEN = "<pad>"

VALID_NUMBERS = list("0123456789+-")

DEFAULT_MODEL = 'mistralai/Mistral-7B-Instruct-v0.2'


class HF:
    """Prompt Pretrained models on HuggingFace to forecast a time series.

    Args:
        name (str):
            Model name. Default to `'mistralai/Mistral-7B-Instruct-v0.2'`.
        sep (str):
            String to separate each element in values. Default to `','`.
    """

    def __init__(self, name=DEFAULT_MODEL, sep=','):
        self.name = name
        self.sep = sep

        self.tokenizer = AutoTokenizer.from_pretrained(self.name, use_fast=False)

        # special tokens
        special_tokens_dict = dict()
        if self.tokenizer.eos_token is None:
            special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
        if self.tokenizer.bos_token is None:
            special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
        if self.tokenizer.unk_token is None:
            special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
        if self.tokenizer.pad_token is None:
            special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN

        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # indicate the end of the time series

        # invalid tokens
        valid_tokens = []
        for number in VALID_NUMBERS:
            token = self.tokenizer.convert_tokens_to_ids(number)
            valid_tokens.append(token)

        valid_tokens.append(self.tokenizer.convert_tokens_to_ids(self.sep))
        self.invalid_tokens = [[i]
                               for i in range(len(self.tokenizer) - 1) if i not in valid_tokens]

        self.model = AutoModelForCausalLM.from_pretrained(
            self.name,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        self.model.eval()

    def forecast(self, text, steps=1, temp=1, top_p=1, raw=False, samples=1, padding=0):
        """Use GPT to forecast a signal.

        Args:
            text (str):
                A string containing signal values.
            steps (int):
                Number of steps ahead to forecast. Default `1`.
            temp (float):
                The value used to modulate the next token probabilities. Default to `1`.
            top_p (float):
                 If set to float < 1, only the smallest set of most probable tokens with
                 probabilities that add up to `top_p` or higher are kept for generation.
                 Default to `1`.
            raw (bool):
                Whether to return the raw output or not. Defaults to `False`.
            samples (int):
                Number of forecasts to generate for each input message. Default to `1`.
            padding (int):
                Additional padding token to forecast to reduce short horizon predictions.
                Default to `0`.

        Returns:
            list, list:
                * List of forecasted signal values.
                * Optionally, a list of dictionaries for raw output.
        """
        tokenized_input = self.tokenizer(
            [text],
            return_tensors="pt"
        ).to("cuda")

        input_length = tokenized_input['input_ids'].shape[1]
        average_length = input_length / len(text.split(','))
        max_tokens = (average_length + padding) * steps

        generate_ids = self.model.generate(
            **tokenized_input,
            do_sample=True,
            max_new_tokens=max_tokens,
            temperature=temp,
            top_p=top_p,
            bad_words_ids=self.invalid_tokens,
            renormalize_logits=True,
            num_return_sequences=samples
        )

        responses = self.tokenizer.batch_decode(
            generate_ids[:, input_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        if raw:
            return responses, generate_ids

        return responses
