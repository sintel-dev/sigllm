#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re

import numpy as np


def format_as_string(X, sep=',', space=False, normal=False):
    """Format X to a list of string.

    Transform an array of integers to string(s), separated by the indicated separator and space.
    Handles two cases:
    - If normal=True, treats X as a single time series (window_size, 1)
    - If normal=False, treats X as multiple windows (num_windows, window_size, 1)

    Args:
        sep (str):
            String to separate each element in X. Default to `','`.
        space (bool):
            Whether to add space between each digit in the result. Default to `False`.
        normal (bool):
            Whether to treat X as a normal time series. If True, expects (window_size, 1)
            and returns a single string. If False, expects (num_windows, window_size, 1)
            and returns a list of strings. Default to `False`.

    Returns:
        ndarray or str:
            If normal=True, returns a single string representation. If normal=False,
            returns a list of string representations for each window.
    """

    def _as_string(x):
        text = sep.join(list(map(str, x.flatten())))
        if space:
            text = ' '.join(text)
        return text

    if normal:
        # Handle as single time series (window_size, 1)
        return _as_string(X)
    else:
        # Handle as multiple windows (num_windows, window_size, 1)
        results = list(map(_as_string, X))
        return np.array(results)


def _from_string_to_integer(text, sep=',', trunc=None, errors='ignore'):
    """Convert a text sequence consisting of digits to an array of integers."""
    nospace = re.sub(r'\s+', '', text)
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
        raise KeyError(f'Unknown errors strategy {errors}.')

    clean = np.array(clean, dtype=float)

    if trunc:
        clean = clean[:trunc]

    return clean


def format_as_integer(X, sep=',', trunc=None, errors='ignore'):
    """Format a nested list of text into an array of integers.

    Transforms a list of list of string input as 3-D array of integers,
    seperated by the indicated seperator and truncated based on `trunc`.
    Handles empty strings by returning empty arrays.

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
            An array of digits values. Empty arrays for empty strings.
    """
    result = list()
    for string_list in X:
        sample = list()
        if not isinstance(string_list, list):
            raise ValueError('Input is not a list of lists.')

        for text in string_list:
            if not text:  # Handle empty string
                sample.append(np.array([], dtype=float))
            else:
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
        """Learn minimum value in fit data."""
        self.minimum = np.min(X)

    def transform(self, X):
        """Transform data."""
        if self.rescale:
            X = X - self.minimum

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
        """Convert data from integer to float."""
        values = X * 10 ** (-decimal)

        return values + minimum


def parse_anomaly_response(X):
    """Parse a list of lists of LLM responses to extract anomaly values and format them as strings.

    Args:
        X (List[List[str]]):
            List of lists of response texts from the LLM in the format
            "Answer: no anomalies" or "Answer: [val1, val2, ..., valN]"

    Returns:
        List[List[str]]:
            List of lists of parsed responses where each element is either
            "val1,val2,...,valN" if anomalies are found, or empty string if
            no anomalies are present
    """

    def _parse_single_response(text: str):
        text = text.strip().lower()

        if 'no anomalies' in text or 'no anomaly' in text:
            return ''

        # match anything that consists of digits and commas
        pattern = r'\[([\d\s,]+)\]'
        match = re.search(pattern, text)

        if match:
            values = match.group(1)
            values = [val.strip() for val in values.split(',') if val.strip()]
            return ','.join(values)

        return ''

    result = []
    for response_list in X:
        parsed_list = [_parse_single_response(response) for response in response_list]
        result.append(parsed_list)

    return result


def format_as_single_string(X, sep=',', space=False):
    """Format a single time series to a string.

    Transform a 1-D array of integers to a single string,
    separated by the indicated separator and space.

    Args:
        sep (str):
            String to separate each element in X. Default to `','`.
        space (bool):
            Whether to add space between each digit in the result. Default to `False`.

    Returns:
        str:
            A string representation of the time series.
    """
    if X.ndim > 1:
        X = X.flatten()

    text = sep.join(list(map(str, X)))

    if space:
        text = ' '.join(text)

    return text
