# -*- coding: utf-8 -*-

"""
Main module.

This module contains functions that get LLM's anomaly detection results.
"""
from sigllm.primitives.prompting.anomalies import get_anomaly_list_within_seq, str2idx
from sigllm.primitives.prompting.data import sig2str


def get_anomalies(seq, msg_func, model_func, num_iters=1, alpha=0.5):
    """Get LLM anomaly detection results.

    The function get the LLM's anomaly detection and converts them into an 1D array

    Args:
        seq (ndarray):
            The sequence to detect anomalies.
        msg_func (func):
            Function to create message prompt.
        model_func (func):
            Function to get LLM answer.
        num_iters (int):
            Number of times to run the same query.
        alpha (float):
            Percentage of total number of votes that an index needs to have to be
            considered anomalous. Default: 0.5

    Returns:
        ndarray:
            1D array containing anomalous indices of the sequence.
    """
    message = msg_func(sig2str(seq, space=True))
    res_list = []
    for i in range(num_iters):
        res = model_func(message)
        ano_ind = str2idx(res, len(seq))
        res_list.append(ano_ind)
    return get_anomaly_list_within_seq(res_list, alpha=alpha)
