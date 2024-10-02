# -*- coding: utf-8 -*-

import json
import logging
import os

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'huggingface_messages.json'
)

PROMPTS = json.load(open(PROMPT_PATH))

LOGGER = logging.getLogger(__name__)

DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_PAD_TOKEN = "<pad>"

VALID_NUMBERS = list("0123456789")

DEFAULT_MODEL = 'mistralai/Mistral-7B-Instruct-v0.2'


class HF:
    """Prompt Pretrained models on HuggingFace to detect anomalies in a time series.

    Args:
        name (str):
            Model name. Default to `'mistralai/Mistral-7B-Instruct-v0.2'`.
        sep (str):
            String to separate each element in values. Default to `','`.
        anomalous_percent (float):
            Expected percentage of time series that are anomalous. Default to `0.5`.
        temp (float):
            The value used to modulate the next token probabilities. Default to `1`.
        top_p (float):
            If set to float < 1, only the smallest set of most probable tokens with
            probabilities that add up to `top_p` or higher are kept for generation.
            Default to `1`.
        raw (bool):
            Whether to return the raw output or not. Defaults to `False`.
        samples (int):
            Number of responsed to generate for each input message. Default to `10`.
        padding (int):
            Additional padding token to forecast to reduce short horizon predictions.
            Default to `0`.
    """

    def __init__(self, name=DEFAULT_MODEL, sep=',', anomalous_percent=0.5, temp=1, top_p=1,
                 raw=False, samples=10, padding=0):
        self.name = name
        self.sep = sep
        self.anomalous_percent = anomalous_percent
        self.temp = temp
        self.top_p = top_p
        self.raw = raw
        self.samples = samples
        self.padding = padding

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

    def detect(self, X, **kwargs):
        """Use HF to detect anomalies of a signal.

        Args:
            X (ndarray):
                Input sequences of strings containing signal values

        Returns:
            list, list:
                * List of detected anomalous values.
                * Optionally, a list of dictionaries for raw output.
        """

        input_length = len(self.tokenizer.encode(X[0].flatten().tolist()[0]))
        max_tokens = input_length * float(self.anomalous_percent)
        all_responses, all_generate_ids = [], []

        for text in tqdm(X):
            system_message = PROMPTS['system_message']
            user_message = PROMPTS['user_message']
            message = ' '.join([system_message, user_message, text, '[RESPONSE]'])

            input_length = len(self.tokenizer.encode(message[0]))

            tokenized_input = self.tokenizer(
                message,
                return_tensors="pt"
            ).to("cuda")

            generate_ids = self.model.generate(
                **tokenized_input,
                do_sample=True,
                max_new_tokens=max_tokens,
                temperature=self.temp,
                top_p=self.top_p,
                bad_words_ids=self.invalid_tokens,
                renormalize_logits=True,
                num_return_sequences=self.samples
            )

            responses = self.tokenizer.batch_decode(
                generate_ids[:, input_length:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            all_responses.append(responses)
            all_generate_ids.append(generate_ids)

        if self.raw:
            return all_responses, all_generate_ids

        return all_responses
