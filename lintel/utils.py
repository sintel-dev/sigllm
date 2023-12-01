import openai
import numpy as np

with open("openai_api_key.txt", "r") as f:
    api_key = f.read()
    
def load_system_prompt(file_path):
    with open(file_path) as f:
        system_prompt = f.read()
    return system_prompt

GPT_model = "gpt-4"#"gpt-3.5-turbo"

client = openai.Client(api_key=api_key)

def get_model_response(message, gpt_model=GPT_model):
    completion = client.chat.completions.create(
    model=gpt_model,
    messages=message,
    )
    return completion.choices[0].message.content

def get_message_zero_shot(seq_query, seq_ex, ano_ind_ex, system_prompt_file='system_prompt.txt'):
    messages = []
    
    messages.append({"role": "system", "content":load_system_prompt(system_prompt_file)})

    # final prompt
    messages.append({"role": "user", "content": f"Sequence: {seq_query}"})
    return messages


def get_message_with_example(seq_query, seq_ex, ano_ind_ex, system_prompt_file='system_prompt.txt'):
    messages = []
    
    messages.append({"role": "system", "content":load_system_prompt(system_prompt_file)})

    # one examples
    messages.append({"role": "user", "content": f"Sequence: {seq_ex}"})
    messages.append({"role": "assistant", "content": ano_ind_ex})

    # final prompt
    messages.append({"role": "user", "content": f"Sequence: {seq_query}"})
    return messages

def create_str_seq(sequence, start, stop, digit_round = 2):
    l_1 = (np.round(sequence.iloc[start:stop].value, digit_round)*(10**digit_round)).astype(int)
    l_2 = l_1 - min(l_1)
    res = ''
    for s in l_2: 
        for d in str(s): 
            res += d
            res += ' '
        res += ', '
    return res

def create_str_ind(start, stop): 
    l = range(start, stop+1)
    res = ''
    for s in l: 
        for d in str(s): 
            res += d
            res += ' '
        res += ', '
    return res

def create_list_ind(answer):
    ans = answer.replace(" ", "")
    ans = ans.replace("[", "")
    ans = ans.replace("]", "")
    in_list = ans.split(",")
    ind_list = [int(i) for i in in_list]
    return ind_list

# def avg_llh(answer, label): 
    
    