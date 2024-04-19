#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Preprocessing module for time series forecasting.
"""

import numpy as np


class Signal2String:
    """Convert a signal into a string.

    Prepare a univariate time series signal for LLMs.
    Transforms a float array into a string of integers.

    Args:
        sep (str):
            String to separate each element in values. Default to `','`.
        space (bool):
            Whether to add space between each digit in the result. Default to `False`.
        decimal (int):
            Number of decimal points to keep from the float representation. Default to `0`.
        rescale(bool):
            Whether to rescale the time series. Default to `False`.
    """

    def __init__(self, sep=',', space=False, decimal=0, rescale=False):
        self.sep = sep
        self.space = space
        self.decimal = decimal
        self.rescale = rescale

    def transform(self, values):
        """Convert a signal to a string.

        Convert a 1-dimensional time series into text by casting and rescaling it
        to nonnegative integer values then into a string.

        Args:
            values (numpy.ndarray):
                A sequence of signal values.

        Returns:
            str:
                Text containing the elements of `values`.
        """
        sign = 1 * (values >= 0) - 1 * (values < 0)
        values = np.abs(values)

        sequence = sign * (values * 10**self.decimal).astype(int)

        # rescale all elements to be nonnegative
        if self.rescale:
            sequence = sequence - min(sequence)

        res = self.sep.join(list(map(str, sequence)))
        if self.space:
            res = ' '.join(res)

        return res

    def reverse_transform(self, text, trunc=None):
        """Convert a text string to a signal.

        Convert a string containing digits into an array of numbers.

        Args:
            text (str):
                A string containing signal values.
            trunc (int):
                Whether to truncate the text to a specific length. Default `None`.

        Returns:
            numpy.ndarray:
                A 1-dimensional array containing parsed elements in `text`.
        """
        if self.space:
            text = text.replace(" ", "")
            
        values = list(filter(None, text.split(self.sep)))
        if trunc:
            values = values[:trunc]

        values = np.array(list(map(float, values)))
        return values * 10**(-self.decimal)
