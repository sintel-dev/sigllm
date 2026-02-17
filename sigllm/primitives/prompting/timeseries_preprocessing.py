# -*- coding: utf-8 -*-

"""Data preprocessing module.

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
    if X.ndim == 1:
        dim = 1
    else:
        dim = X.shape[1]

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
    return np.asarray(out_X), np.asarray(X_index), window_size, step_size, dim


def long_short_term_context(X, first_index, step_size, L, S=None, W=None, aggregation_method='mean', window_size=None, dim=None):
    """Process rolling window sequences to create long-term aggregated and short-term raw context windows.

    This function processes already-windowed sequences from rolling_window_sequences to create
    context windows with long-term aggregated segments (rounded to nearest integer) and
    short-term raw segments.

    Args:
        X (ndarray):
            Input windows from rolling_window_sequences with shape (num_windows, window_size, dim).
        first_index (ndarray):
            First index value of each input sequence.
        step_size (int):
            Step size used in rolling_window_sequences.
        L (int):
            Number of long-term windows to extract.
        S (int, optional):
            Number of short-term values to extract (last S values). If None, set to L.
            Defaults to None.
        W (int, optional):
            Size of each long-term window. If None, computed automatically from window_size, L, and S
            Defaults to None.
        aggregation_method (str):
            Aggregation method for long-term windows. Currently only 'mean' is supported.
            Defaults to 'mean'.
        window_size (int, optional):
            Size of each input window. If None, computed from X.shape[1].
        dim (int, optional):
            Dimensionality of the data. If None, computed from X.shape.

    Returns:
        tuple:
            * processed_windows_array (ndarray): Processed windows with shape (num_windows, S + L, dim)
            * first_index (ndarray): First index array (passed through)
            * new_window_size (int): New window size (S + L)
            * step_size (int): Step size (passed through)
            * dim (int): Dimensionality (passed through)
    """
    if aggregation_method != 'mean':
        raise ValueError(f"Aggregation method '{aggregation_method}' not yet supported. Only 'mean' is currently supported.")

    if window_size is None:
        window_size = X.shape[1]
    
    if dim is None:
        if X.ndim == 2:
            dim = 1
        else:
            dim = X.shape[2]

    if S is None:
        S = L  
    
    if W is None:
        if window_size < S:
            raise ValueError(f"window_size ({window_size}) must be at least S ({S})")
        W = (window_size - S) // L
        if W <= 0:
            raise ValueError(f"Cannot compute W: window_size ({window_size}) must be greater than S ({S}) + L ({L})")
    
    required_size = W * L + S
    remainder = window_size - required_size
    
    if remainder < 0:
        raise ValueError(f"window_size ({window_size}) is too small. Need at least W*L + S = {W}*{L} + {S} = {required_size}")
    

    num_windows = X.shape[0]
    processed_windows = []

    is_2d_input = X.ndim == 2

    for i in range(num_windows):
        window = X[i]  

        if remainder > 0:
            window = window[remainder:]  

        long_term_aggregated = []
        for j in range(L):
            start_idx = j * W
            end_idx = (j + 1) * W
            long_term_window = window[start_idx:end_idx]  

            if is_2d_input:
                aggregated_value = np.mean(long_term_window)
            else:
                if dim == 1:
                    aggregated_value = np.mean(long_term_window)
                else:
                    aggregated_value = np.mean(long_term_window, axis=0)  

            aggregated_value = np.round(aggregated_value).astype(int)
            long_term_aggregated.append(aggregated_value)

        short_term = window[required_size - S:required_size]  

        if short_term.dtype != np.int64:
            short_term = short_term.astype(int)

        if is_2d_input:
            long_term_array = np.array(long_term_aggregated)  
            combined = np.concatenate([long_term_array, short_term])  
            processed_windows.append(combined)
        else:
            long_term_array = np.array(long_term_aggregated)  
            if dim == 1:
                combined = np.concatenate([long_term_array, short_term.flatten()])  
                processed_windows.append(combined.reshape(-1, 1))  
            else:
                combined = np.concatenate([long_term_array, short_term], axis=0)  
                processed_windows.append(combined)

    processed_windows_array = np.asarray(processed_windows)  
    new_window_size = S + L

    return processed_windows_array, first_index, new_window_size, step_size, dim
