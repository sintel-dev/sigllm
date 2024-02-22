# -*- coding: utf-8 -*-

"""
GPT model module.

This module contains functions that are specifically used for GPT models
"""

import openai

with open("../gpt_model/openai_api_key.txt", "r") as f:
    api_key = f.read()
    
def load_system_prompt(file_path):
    with open(file_path) as f:
        system_prompt = f.read()
    return system_prompt

GPT_model = "gpt-3.5-turbo" #"gpt-4"

client = openai.Client(api_key=api_key)

def get_gpt_model_response(message, gpt_model=GPT_model):
    completion = client.chat.completions.create(
    model=gpt_model,
    messages=message,
    )
    return completion.choices[0].message.content

def create_message_zero_shot(seq_query, system_prompt_file='../gpt_model/system_prompt_zero_shot.txt'):
    messages = []
    
    messages.append({"role": "system", "content":load_system_prompt(system_prompt_file)})

    # final prompt
    messages.append({"role": "user", "content": f"Sequence: {seq_query}"})
    return messages


def create_message_one_shot(seq_query, seq_ex, ano_ind_ex, system_prompt_file='../gpt_model/system_prompt_one_shot.txt'):
    messages = []
    
    messages.append({"role": "system", "content":load_system_prompt(system_prompt_file)})

    # one shot
    messages.append({"role": "user", "content": f"Sequence: {seq_ex}"})
    messages.append({"role": "assistant", "content": ano_ind_ex})

    # final prompt
    messages.append({"role": "user", "content": f"Sequence: {seq_query}"})
    return messages

