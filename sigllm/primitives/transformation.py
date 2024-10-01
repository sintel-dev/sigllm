#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Transformation module.
"""

import re

import numpy as np


def format_as_string(X, sep=',', space=False):
    """Format X to a list of string.

    Transform a 2-D array of integers to a list of strings,
    seperated by the indicated seperator and space.

    Args:
        sep (str):
            String to separate each element in X. Default to `','`.
        space (bool):
            Whether to add space between each digit in the result. Default to `False`.

    Returns:
        ndarray:
            A list of string representation of each row.
    """
    def _as_string(x):
        text = sep.join(list(map(str, x.flatten())))

        if space:
            text = ' '.join(text)

        return text

    results = list(map(_as_string, X))

    return np.array(results)


def _from_string_to_integer(text, sep=',', trunc=None, errors='ignore'):
    """Convert a text sequence consisting of digits to an array of integers."""

    nospace = re.sub(r"\s+", "", text)
    rule = f'[^0-9|{sep}]'

    if errors == 'raise':
        search = re.search(rule, nospace)
        if bool(search):
            start, end = search.span()
            raise ValueError(f'Encountered a non-digit value {nospace[start:end]}.')

    values = list(filter(None, nospace.split(sep)))

    if errors == 'ignore':
        clean = list(filter(lambda x: not bool(re.search(rule, x)), values))

    elif errors == 'filter':
        clean = list(map(lambda x: re.sub(rule, '', x), values))

    elif errors == 'coerce':
        clean = list(map(lambda x: x if not bool(re.search(rule, x)) else np.nan, values))

    else:
        raise KeyError(f"Unknown errors strategy {errors}.")

    clean = np.array(clean, dtype=float)

    if trunc:
        clean = clean[:trunc]

    return clean


def format_as_integer(X, sep=',', trunc=None, errors='ignore'):
    """Format a nested list of text into an array of integers.

    Transforms a list of list of string input as 3-D array of integers,
    seperated by the indicated seperator and truncated based on `trunc`.

    Args:
        sep (str):
            String to separate each element in values. Default to `','`.
        trunc (int):
            How many values to keep from the ndarray. Default to `None`,
            which retains all values.
        errors (str):
            Strategy to deal with erroneous values (not digits). Default
            to `'ignore'`.
            - If 'ignore', then invalid values will be ignored in the result.
            - If 'filter', then invalid values will be filtered out of the string.
            - If 'raise', then encountering invalud values will raise an exception.
            - If 'coerce', then invalid values will be set as NaN.

    Returns:
        ndarray:
            An array of digits values.
    """
    result = list()
    for string_list in X:
        sample = list()
        if not isinstance(string_list, list):
            raise ValueError("Input is not a list of lists.")

        for text in string_list:
            scalar = _from_string_to_integer(text, sep, trunc, errors)
            sample.append(scalar)

        result.append(sample)

    output = np.array(result, dtype=object)
    if output.ndim >= 3:
        output = output.astype(float)

    return output


class Float2Scalar:
    """Convert an array of float values to scalar.

    Transforms an array of floats to an array integers. With the
    option to rescale such that the minimum value becomes zero
    and you can keep certain decimal points.

        1.05, 2., 3.1, 4.8342, 5, 0 -> 105, 200, 310, 483, 500, 0

    Args:
        decimal (int):
            Number of decimal points to keep from the float representation. Default to `2`.
        rescale (bool):
            Whether to rescale the array such that the minimum value becomes 0. Default to `True`.
    """

    def __init__(self, decimal=2, rescale=True):
        self.decimal = decimal
        self.rescale = rescale
        self.minimum = None

    def fit(self, X):
        self.minimum = np.min(X)

    def transform(self, X):
        if self.rescale:
            X = (X - self.minimum)

        sign = 1 * (X >= 0) - 1 * (X < 0)
        values = np.abs(X)

        values = sign * (values * 10**self.decimal).astype(int)

        return values, self.minimum, self.decimal


class Scalar2Float:
    """Convert an array of integer values to float.

    Transforms an array of integers to an array floats.
    Shift values by minimum and include a predetermined
    number of decimal points.

        105, 200, 310, 483, 500, 0 -> 1.05, 2., 3.1, 4.8342, 5, 0

    Args:
        minimum (float):
            Bias to shift the data. Captured from Float2Scalar.
        decimal (int):
            Number of decimal points to keep from the float representation. Default to `2`.
    """

    def transform(self, X, minimum=0, decimal=2):
        values = X * 10**(-decimal)

        return values + minimum
