# -*- coding: utf-8 -*-

"""
Data preprocessing module.

This module contains functions that prepare timeseries for a language model.
"""

import numpy as np


def rolling_window_sequences(X, index, window_size, step_size):
    """Create rolling window sequences out of time series data.

    The function creates an array of sequences by rolling over the input sequence.

    Args:
        X (ndarray):
            The sequence to iterate over.
        index (ndarray):
            Array containing the index values of X.
        window_size (int):
            Length of window.
        step_size (int):
            Indicating the number of steps to move the window forward each round.

    Returns:
        ndarray, ndarray:
            * rolling window sequences.
            * first index value of each input sequence.
    """
    out_X = list()
    X_index = list()

    start = 0
    max_start = len(X) - window_size + 1
    while start < max_start:
        end = start + window_size
        out_X.append(X[start:end])
        X_index.append(index[start])
        start = start + step_size

    return np.asarray(out_X), np.asarray(X_index)


def sig2str(values, sep=',', space=False, decimal=0, rescale=True):
    """Convert a signal to a string.

    Convert a 1-dimensional time series into text by casting and rescaling it
    to nonnegative integer values then into a string (optional).

    Args:
        values (numpy.ndarray):
            A sequence of signal values.
        sep (str):
            String to separate each element in values. Default to `","`.
        space (bool):
            Whether to add space between each digit in the result. Default to `False`.
        decimal (int):
            Number of decimal points to keep from the float representation. Default to `0`.
        rescale(bool):
            Whether to rescale the time series. Default to `True`

    Returns:
        str:
            Text containing the elements of `values`.
    """
    sign = 1 * (values >= 0) - 1 * (values < 0)
    values = np.abs(values)

    sequence = sign * (values * 10**decimal).astype(int)

    # Rescale all elements to be nonnegative
    if rescale:
        sequence = sequence - min(sequence)

    res = sep.join([str(num) for num in sequence])
    if space:
        res = ' '.join(res)

    return res
