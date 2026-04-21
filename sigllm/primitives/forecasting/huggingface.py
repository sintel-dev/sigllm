# -*- coding: utf-8 -*-

import os
import pickle
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_BOS_TOKEN = '<s>'
DEFAULT_EOS_TOKEN = '</s>'
DEFAULT_UNK_TOKEN = '<unk>'
DEFAULT_PAD_TOKEN = '<pad>'

VALID_NUMBERS = list('0123456789')
VALID_MULTIVARIATE_SYMBOLS = []

DEFAULT_MODEL = 'mistralai/Mistral-7B-Instruct-v0.2'


def _default_timestamped_cache_dir():
    """``cache/<local-datetime-to-millisecond>/`` under the current working directory."""
    now = datetime.now()
    sub = now.strftime('%Y-%m-%dT%H-%M-%S') + f'-{now.microsecond // 1000:03d}'
    return Path('cache') / sub


class HFMaxRetriesExceededError(RuntimeError):
    """Raised when generation never succeeds within max_retries for a window."""

    def __init__(self, window_index, max_retries):
        super().__init__(
            f'Maximum retries ({max_retries}) exceeded for window {window_index} without valid output.'
        )
        self.window_index = window_index
        self.max_retries = max_retries


class HF:
    """Prompt Pretrained models on HuggingFace to forecast a time series.

    ``validate_window`` is optional. When set, it is applied **after each window**
    is generated (and before that window's checkpoint is written), not once at the
    end of the full sequence. Pipelines can assign it via ``run_pipeline`` or by
    setting the attribute on the HF instance.

    If ``cache_dir`` is omitted and ``SIGLLM_CACHE_DIR`` is unset, checkpoints go
    under ``cache/<local-datetime-to-millisecond>/``. Pass ``cache_dir=False`` or
    ``""`` to disable disk storage.
    """

    validate_window = None

    def __init__(
        self,
        name=DEFAULT_MODEL,
        sep=',',
        steps=1,
        temp=1,
        top_p=1,
        raw=False,
        samples=1,
        padding=0,
        multivariate_allowed_symbols=VALID_MULTIVARIATE_SYMBOLS,
        cache_dir=None,
        max_retries=3,
    ):
        self.name = name
        self.sep = sep
        self.steps = steps
        self.temp = temp
        self.top_p = top_p
        self.raw = raw
        self.samples = samples
        self.padding = padding
        self.multivariate_allowed_symbols = multivariate_allowed_symbols
        self.max_retries = max_retries

        if cache_dir is False or cache_dir == '':
            self.cache_dir = None
        elif cache_dir:
            self.cache_dir = Path(cache_dir)
        elif os.getenv('SIGLLM_CACHE_DIR'):
            self.cache_dir = Path(os.getenv('SIGLLM_CACHE_DIR'))
        else:
            self.cache_dir = _default_timestamped_cache_dir()
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.tokenizer = AutoTokenizer.from_pretrained(self.name, use_fast=False)

        special_tokens_dict = dict()
        if self.tokenizer.eos_token is None:
            special_tokens_dict['eos_token'] = DEFAULT_EOS_TOKEN
        if self.tokenizer.bos_token is None:
            special_tokens_dict['bos_token'] = DEFAULT_BOS_TOKEN
        if self.tokenizer.unk_token is None:
            special_tokens_dict['unk_token'] = DEFAULT_UNK_TOKEN
        if self.tokenizer.pad_token is None:
            special_tokens_dict['pad_token'] = DEFAULT_PAD_TOKEN

        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        valid_tokens = []
        for number in VALID_NUMBERS:
            token = self.tokenizer.convert_tokens_to_ids(number)
            valid_tokens.append(token)

        for symbol in self.multivariate_allowed_symbols:
            valid_tokens.append(self.tokenizer.convert_tokens_to_ids(symbol))

        valid_tokens.append(self.tokenizer.convert_tokens_to_ids(self.sep))
        self.invalid_tokens = [
            [i] for i in range(len(self.tokenizer) - 1) if i not in valid_tokens
        ]

        self.model = AutoModelForCausalLM.from_pretrained(
            self.name,
            device_map='auto',
            torch_dtype=torch.float16,
        )

        self.model.eval()
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

    def _delete_window_pkls(self):
        if self.cache_dir is None:
            return

        for pkl_file in self.cache_dir.glob('window_*.pkl'):
            pkl_file.unlink(missing_ok=True)

    def _run_prediction_for_text(self, text):
        tokenized_input = self.tokenizer([text], return_tensors='pt').to('cuda')

        input_length = tokenized_input['input_ids'].shape[1]
        average_length = input_length / len(text.split(self.sep))
        max_tokens = int((average_length + self.padding) * self.steps)

        generate_ids = self.model.generate(
            **tokenized_input,
            do_sample=True,
            max_new_tokens=max_tokens,
            temperature=self.temp,
            top_p=self.top_p,
            bad_words_ids=self.invalid_tokens,
            renormalize_logits=True,
            num_return_sequences=self.samples,
        )

        responses = self.tokenizer.batch_decode(
            generate_ids[:, input_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return responses

    def _generate_validated_responses(self, text, window_index):
        """Generate one window; optionally re-sample until ``validate_window`` passes."""
        retries = self.max_retries
        validate_window = self.validate_window
        for _ in range(retries + 1):
            last_responses = self._run_prediction_for_text(text)
            if validate_window is None or validate_window(last_responses):
                return last_responses
        raise HFMaxRetriesExceededError(window_index, retries)

    def forecast(self, X):
        """Generate forecasts with optional disk checkpointing.

        One ``cache_dir`` holds at most one completed run in ``all_responses.pkl``.
        If that file exists, load it (optionally repair entries), return, and leave
        ``window_*.pkl`` untouched. Otherwise fill from per-window checkpoints and
        generation, then write ``all_responses.pkl`` and remove per-window files.
        """
        if self.cache_dir is None:
            all_responses = []
            for i, text in enumerate(tqdm(X)):
                all_responses.append(self._generate_validated_responses(text, i))
            return all_responses

        validate_window = self.validate_window
        combined_file = self.cache_dir / 'all_responses.pkl'

        if combined_file.exists():
            with open(combined_file, 'rb') as f:
                all_responses = pickle.load(f)['responses']
            if validate_window is not None:
                bad_indices = [
                    i for i, resp in enumerate(all_responses) if not validate_window(resp)
                ]
                if bad_indices:
                    for i in tqdm(bad_indices, desc='repairing cached responses'):
                        all_responses[i] = self._generate_validated_responses(X[i], i)
                    with open(combined_file, 'wb') as f:
                        pickle.dump({'responses': all_responses}, f)
            return all_responses

        all_responses = []

        for i, text in enumerate(tqdm(X)):
            cache_file = self.cache_dir / ('window_%06d.pkl' % i)
            loaded = False

            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    cached = pickle.load(f)
                if not isinstance(cached, dict) or 'responses' not in cached:
                    try:
                        print(f"Invalid cache format at {cache_file}, regenerating")
                        cache_file.unlink()
                    except OSError:
                        pass
                else:
                    responses = cached['responses']
                    if validate_window is None or validate_window(responses):
                        all_responses.append(responses)
                        loaded = True
                    else:
                        try:
                            cache_file.unlink()
                        except OSError:
                            pass

            if not loaded:
                responses = self._generate_validated_responses(text, i)
                all_responses.append(responses)
                with open(cache_file, 'wb') as f:
                    pickle.dump({'responses': responses}, f)

        with open(combined_file, 'wb') as f:
            pickle.dump({'responses': all_responses}, f)

        self._delete_window_pkls()
        return all_responses
