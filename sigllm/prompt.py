# -*- coding: utf-8 -*-

"""
Prompt module.

This module contains functions that are specifically used to create prompt
"""
import os

def load_system_prompt(file_path):
    with open(file_path) as f:
        system_prompt = f.read()
    return system_prompt


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

ZERO_SHOT_FILE = 'gpt_system_prompt_zero_shot.txt'
ONE_SHOT_FILE = 'gpt_system_prompt_one_shot.txt'

ZERO_SHOT_DIR = os.path.join(CURRENT_DIR, "..", "template", ZERO_SHOT_FILE)
ONE_SHOT_DIR = os.path.join(CURRENT_DIR, "..", "template", ONE_SHOT_FILE)

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

