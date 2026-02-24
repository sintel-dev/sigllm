#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re

import numpy as np
from sklearn.cluster import KMeans


def format_as_string(X, sep=',', space=False, single=False):
    """Format X to a list of string.

    Transform an array of integers to string(s), separated by the
    indicated separator and space. Handles two cases:
    - If single=True, treats X as a single time series (window_size, 1)
    - If single=False, treats X as multiple windows (num_windows, window_size, 1)

    Args:
        sep (str):
            String to separate each element in X. Default to `','`.
        space (bool):
            Whether to add space between each digit in the result. Default to `False`.
        single (bool):
            Whether to treat X as a single time series. If True, expects (window_size, 1)
            and returns a single string. If False, expects (num_windows, window_size, 1)
            and returns a list of strings. Default to `False`.

    Returns:
        ndarray or str:
            If single=True, returns one string representation. If single=False,
            returns a list of string representations for each window.
    """

    def _as_string(x):
        text = sep.join(list(map(str, x.flatten())))
        if space:
            text = ' '.join(text)
        return text

    if single:
        # single time series (window_size, 1)
        return _as_string(X)
    else:
        # multiple windows (num_windows, window_size, 1)
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
            if not text:  # empty string
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

        values = sign * (values * 10**self.decimal)
        values = np.where(
            np.abs(values - np.rint(values)) < 1e-8, np.rint(values), np.floor(values)
        )
        values = values.astype(int)

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


class Scalar2Cluster:
    """Convert an array of float values to cluster indices using K-means.

    Fits K-means on the input data and maps each value to the index of
    its nearest centroid. Centroids are sorted in ascending order so that
    cluster index 0 corresponds to the smallest centroid value.

    Args:
        n_clusters (int):
            Number of K-means clusters. Default to ``100``.
    """

    def __init__(self, n_clusters=100):
        self.n_clusters = n_clusters
        self.centroids = None

    def fit(self, X):
        """Fit K-means on the data and store sorted centroids.

        Args:
            X (ndarray):
                2-D array of shape ``(n_samples, n_features)``.
        """
        centroids_list = []
        for col in X.T:
            n_unique = len(np.unique(col))
            if self.n_clusters >= n_unique:
                centroids = np.sort(np.unique(col))
            else:
                kmeans = KMeans(n_clusters=self.n_clusters, random_state=0, n_init=10)
                kmeans.fit(col.reshape(-1, 1))
                centroids = np.sort(kmeans.cluster_centers_.ravel())
            centroids_list.append(centroids)

        self.centroids = centroids_list

    def transform(self, X):
        """Map each value to its nearest centroid index.

        Args:
            X (ndarray):
                2-D array of shape ``(n_samples, n_features)``.

        Returns:
            tuple:
                * **X** (ndarray) - Integer cluster labels with the same shape as input.
                * **centroids** (list of ndarray) - Sorted centroid arrays, one per column.
        """
        labels_list = []
        for i, col in enumerate(X.T):
            centroids = self.centroids[i]
            col_labels = np.argmin(np.abs(col[:, None] - centroids[None, :]), axis=1)
            labels_list.append(col_labels)

        labels = (
            np.column_stack(labels_list) if len(labels_list) > 1 else labels_list[0].reshape(-1, 1)
        )
        return labels, self.centroids


class Cluster2Scalar:
    """Convert cluster indices back to float values using centroids.

    Maps an array of integer cluster indices to the corresponding
    centroid values produced by :class:`Scalar2Cluster`.
    """

    def transform(self, X, centroids):
        """Convert cluster indices to centroid float values.

        Args:
            X (ndarray):
                Integer cluster labels.
            centroids (list of ndarray):
                Sorted centroid arrays from :class:`Scalar2Cluster`.

        Returns:
            ndarray:
                Float values corresponding to the centroid of each label.
        """
        base_centroids = np.asarray(centroids[0])
        idx = np.clip(X.astype(int), 0, len(base_centroids) - 1)
        return np.take(base_centroids, idx)
