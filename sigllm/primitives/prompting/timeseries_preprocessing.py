# -*- coding: utf-8 -*-

"""
Data preprocessing module.

This module contains functions that prepare timeseries for a language model.
"""

import numpy as np


def rolling_window_sequences(X, window_size=500, step_size=100):
    """Create rolling window sequences out of time series data.

    This function creates an array of sequences by rolling over the input sequence.

    Args:
        X (ndarray):
            The sequence to iterate over.
        window_size (int):
            Length of window. Defaults to 500
        step_size (int):
            Indicating the number of steps to move the window forward each round. Defaults to 100

    Returns:
        ndarray, ndarray:
            * rolling window sequences.
            * first index value of each input sequence.
    """
    index = range(len(X))
    out_X = list()
    X_index = list()

    start = 0
    max_start = len(X) - window_size + 1
    while start < max_start:
        end = start + window_size
        out_X.append(X[start:end])
        X_index.append(index[start])
        start = start + step_size

    return np.asarray(out_X), np.asarray(X_index), window_size, step_size
