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

    def __init__(self, strategy='scaling', n_clusters=100, decimal=2, rescale=True):
        self.strategy = strategy
        self.n_clusters = n_clusters
        self.decimal = decimal
        self.rescale = rescale
        
        # State variables
        self.minimum = None
        self.centroids = None
        self.labels = None

    def fit(self, X):
        """Learn parameters from data.
        
        For scaling: learns the minimum value.
        For binning: learns K-means cluster centroids.
        """
        if self.strategy == 'scaling':
            self.minimum = np.min(X)
        elif self.strategy == 'binning':
            centroids_list = []
            labels = []
            for col in X.T:
                if self.n_clusters >= len(np.unique(col)):
                    centroids = np.unique(col)
                else:     
                    kmeans = KMeans(n_clusters=self.n_clusters, random_state=0)
                    kmeans.fit(col.reshape(-1, 1))
                    centroids = np.sort(kmeans.cluster_centers_.ravel())
                    
                col_labels = np.argmin(np.abs(col[:, None] - centroids[None, :]), axis=1)

                labels.append(col_labels)
                centroids_list.append(centroids)
            
            self.labels = np.column_stack(labels)
            self.centroids = centroids_list
        else:
            raise ValueError(f"Unknown strategy '{self.strategy}'. Use 'scaling' or 'binning'.")

    def transform(self, X):
        """Transform data to integer representation.
        
        Returns:
            tuple: (values, metadata) where metadata is a dict containing:
                - For scaling: {'strategy': 'scaling', 'minimum': float, 'decimal': int}
                - For binning: {'strategy': 'binning', 'centroids': list}
        """
        print(f"[Float2Scalar] Using strategy: {self.strategy}")
        if self.strategy == 'scaling':
            if self.rescale:
                X = X - self.minimum

            sign = 1 * (X >= 0) - 1 * (X < 0)
            values = np.abs(X)

            values = sign * np.round(values * 10**self.decimal).astype(int)

            metadata = {
                'strategy': 'scaling',
                'minimum': self.minimum,
                'decimal': self.decimal
            }
            return values, metadata
        
        elif self.strategy == 'binning':
            # Re-fit to get labels for this X (transform is same as fit for binning)
            self.fit(X)
            metadata = {
                'strategy': 'binning',
                'centroids': self.centroids
            }
            return self.labels, metadata
        
        else:
            raise ValueError(f"Unknown strategy '{self.strategy}'. Use 'scaling' or 'binning'.")


class Scalar2Float:
    """Convert an array of integer values to float.

    Transforms an array of integers back to floats using the metadata from Float2Scalar.
    
    - 'scaling': Divide by 10^decimal and add minimum offset.
        Example: 105, 200, 310, 483, 500, 0 -> 1.05, 2., 3.1, 4.83, 5, 0
    
    - 'binning': Map cluster indices back to centroid values.
    """

    def transform(self, X, metadata):
        """Convert data from integer back to float.
        
        Args:
            X (ndarray): Integer values to convert.
            metadata (dict): Metadata from Float2Scalar containing strategy and parameters.
        
        Returns:
            ndarray: Float values.
        """
        strategy = metadata.get('strategy', 'binning')
        print(f"[Scalar2Float] Using strategy: {strategy}")
        print(f"[Scalar2Float] Full metadata: {metadata}")
        
        if strategy == 'scaling':
            minimum = metadata.get('minimum', 0)
            decimal = metadata.get('decimal', 2)
            values = X * 10 ** (-decimal)
            return values + minimum
        
        elif strategy == 'binning':
            centroids = metadata.get('centroids')
            if centroids is None:
                raise ValueError("centroids must be provided in metadata for binning strategy")
            base_centroids = np.asarray(centroids[0]) 
            idx = np.clip(X.astype(int), 0, len(base_centroids) - 1)
            X_pred = np.take(base_centroids, idx)
            return X_pred
        
        else:
            raise ValueError(f"Unknown strategy '{strategy}'. Use 'scaling' or 'binning'.")
