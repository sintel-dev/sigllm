# -*- coding: utf-8 -*-

"""
Data preprocessing module.

This module contains functions to help parse time series into
text, preparing it for a language model.
"""

import numpy as np


def sig2str(values, sep=',', space=False, decimal=0):
    """Convert a signal to a string.

    Convert a 1-dimensional time series into text by casting it
    to integer values then into a string.

    Args:
        values (numpy.ndarray):
            A sequence of signal values.
        sep (str):
            String to separate each element in values, Default to `","`.
        space (bool):
            Whether to add space between each digit in the result. Default to `False`.
        decimal (int):
            Number of decimal points to keep from the float representation. Default to `0`.

    Returns:
        str:
            Text containing the elements of `values`.
    """
    sign = 1 * (values >= 0) - 1 * (values < 0)
    values = np.abs(values)

    sequence = sign * (values * 10**decimal).astype(int)

    res = sep.join([str(num) for num in sequence])
    if space:
        res = ' '.join(res)

    return res


def str2sig(text, sep=',', decimal=0):
    """Convert a text string to a signal.

    Convert a string containing digits into an array of numbers.

    Args:
        text (str):
            A string containing signal values.
        sep (str):
            String that was used to separate each element in text, Default to `","`.
        decimal (int):
            Number of decimal points to shift each element in text to. Default to `0`.

    Returns:
        numpy.ndarray:
            A 1-dimensional array containing parsed elements in `text`.
    """
    values = np.fromstring(text, dtype=float, sep=sep)
    return values * 10**(-decimal)
