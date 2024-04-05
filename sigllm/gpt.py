# -*- coding: utf-8 -*-

"""
GPT model module.

This module contains functions that are specifically used for GPT models
"""
import os

from openai import OpenAI
import tiktoken

client = OpenAI()
VALID_NUMBERS = list("0123456789+- ")
BIAS = 30
SEP = ","
GPT_MODEL = "gpt-3.5-turbo"
TOKENIZER = tiktoken.encoding_for_model(GPT_MODEL)
VALID_TOKENS = []
for number in VALID_NUMBERS:
    token = TOKENIZER.encode(number)[0]
    VALID_TOKENS.append(token)

VALID_TOKENS.append(TOKENIZER.encode(SEP)[0])
LOGIT_BIAS = {token: BIAS for token in VALID_TOKENS}


def get_gpt_model_response(
        message, 
#         frequency_penalty = 0, 
#        logit_bias = LOGIT_BIAS,           
#             logprobs = False, (KeyError that can't be fixed as of now)
#         top_logprobs = None, 
#         max_tokens = None, 
#         n = 1,       
#         presence_penalty = 0, 
#             response_format = {"type": "text"},
#         seed = None, 
#         stop = None, 
#             stream = False, 
#         temperature = 1,
#         top_p = 1, 
#             tools = [], 
#             tool_choice = 'none', 
#             user = ""
      ):
    """Return GPT model response to message prompt

    Args: 
        message (List[dict]): 
            prompt written in template format.
        gpt_model (str): 
            GPT model name. Defaults to `"gpt-3.5-turbo"`.
        frequency_penalty (number or null): 
            Number between -2.0 and 2.0. Positive values penalize new tokens based on their 
            existing frequency in the text so far, decreasing the model's likelihood to repeat 
            the same line verbatim. Defaults to `0`.
        logit_bias (map):
            Modify the likelihood of specified tokens appearing in the completion.
            Accepts a JSON object that maps tokens (specified by their token ID in the tokenizer) 
            to an associated bias value from -100 to 100. Mathematically, the bias is added to the 
            logits generated by the model prior to sampling. The exact effect will vary per model, 
            but values between -1 and 1 should decrease or increase likelihood of selection; 
            values like -100 or 100 should result in a ban or exclusive selection of the relevant token.
            Defaults to `None`.
        logprobs (bool or null): 
            Whether to return log probabilities of the output tokens or not. If true, returns the 
            log probabilities of each output token returned in the content of message. This option 
            is currently not available on the gpt-4-vision-preview model. Defaults to `False`.
        top_logprobs (int or null): 
            An integer between 0 and 20 specifying the number of most likely tokens to return at 
            each token position, each with an associated log probability. logprobs must be set to 
            `True` if this parameter is used. Default to `None`.
        max_tokens (int or null): 
            The maximum number of tokens that can be generated in the chat completion. Defaults to `None`.
        n (int or null):
            How many chat completion choices to generate for each input message. Note that you will be charged 
            based on the number of generated tokens across all of the choices. Defaults to `1`.
        present_penalty (number or null): 
            Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the 
            text so far, increasing the model's likelihood to talk about new topics. Defaults to `0`.
        response_format (object):
            An object specifying the format that the model must output. Compatible with GPT-4 Turbo 
            and all GPT-3.5 Turbo models newer than gpt-3.5-turbo-1106. 
            Setting to { "type": "json_object" } enables JSON mode, which guarantees the message 
            the model generates is valid JSON.
            Defaults to `{"type": "text"}`
        seed (int or null):
            This feature is in Beta. If specified, our system will make a best effort to sample deterministically, 
            such that repeated requests with the same seed and parameters should return the same result. 
            Determinism is not guaranteed, and you should refer to the system_fingerprint response parameter 
            to monitor changes in the backend. Defaults to `None`.
        stop (str/array/null): 
            Up to 4 sequences where the API will stop generating further tokens. Defaults to `None`.
        stream (bool or null): 
            If set, partial message deltas will be sent, like in ChatGPT. Tokens will be sent as data-only 
            server-sent events as they become available, with the stream terminated by a data: [DONE] message.
            Defaults to `False`.
        temperature (number or null): 
            What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the 
            output more random, while lower values like 0.2 will make it more focused and deterministic.
            We generally recommend altering this or top_p but not both. Defaults to `1`.
        top_p (number or null): 
            An alternative to sampling with temperature, called nucleus sampling, where the model considers 
            the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising 
            the top 10% probability mass are considered.
            We generally recommend altering this or temperature but not both. Defaults to `1`.
        tools (array): 
            A list of tools the model may call. Currently, only functions are supported as a tool. 
            Use this to provide a list of functions the model may generate JSON inputs for. 
            A max of 128 functions are supported. Defaults to `[]`.
        tool_choice (str or object): 
            Controls which (if any) function is called by the model.
            `none` means the model will not call a function and instead generates a message. 
            `auto` means the model can pick between generating a message or calling a function. 
            Specifying a particular function via {"type": "function", "function": {"name": "my_function"}} 
            forces the model to call that function.
            Defaults to `none`.
        user (str): 
            A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.

    Returns: 
        chat competion object: 
            GPT model response.
    """


    completion = client.chat.completions.create(
        model = GPT_MODEL,
        messages = message,
    #             frequency_penalty = frequency_penalty, 
        logit_bias = LOGIT_BIAS, 
    #             logprobs = logprobs, 
#         top_logprobs = top_logprobs,
#         max_tokens = max_tokens,
#         n = n, 
#         presence_penalty = presence_penalty,
    #             response_format = response_format,
#         seed = seed, 
#         stop = stop, 
    #             stream = stream,
#         temperature = temperature,
#         top_p = top_p,
    #             tools = tools,
    #             tool_choice = tool_choice,
    #             user = user
    )
    return completion.choices[0].message.content