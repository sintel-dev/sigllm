# -*- coding: utf-8 -*-

"""
Prompter post-processing module

This module contains functions that help filter LLMs results to get the final anomalies.
"""

import numpy as np
import pandas as pd


def val2idx(y, X):
    """Convert detected anomalies values into indices.

    Convert windows of detected anomalies values into an array of all indices
    in the input sequence that have those values.

    Args:
        y (ndarray):
            A 3d array containing detected anomalous values from different
            responses of each window.
        X (ndarray):
            rolling window sequences.
    Returns:
        List([ndarray]):
            A 3d array containing detected anomalous indices from different
            responses of each window.
    """

    idx_list = []
    for anomalies_list, seq in zip(y, X):
        idx_win_list = []
        for anomalies in anomalies_list:
            mask = np.isin(seq, anomalies)
            indices = np.where(mask)[0]
            idx_win_list.append(indices)
        idx_list.append(idx_win_list)
    idx_list = np.array(idx_list, dtype=object)
    return idx_list


def ano_within_windows(y, alpha=0.5):
    """Get the final list of anomalous indices of each window

    Choose anomalous index in the sequence based on multiple LLM responses

    Args:
        y (ndarray):
            A 3d array containing detected anomalous values from different
            responses of each window.
        alpha (float):
            Percent of votes needed for an index to be anomalous. Default to `0.5`.

    Returns:
        ndarray:
            A 2-dimensional array containing final anomalous indices of each windows.
    """

    idx_list = []
    for samples in y:
        min_vote = np.ceil(alpha * len(samples))
        # print(type(samples.tolist()))

        flattened_res = np.concatenate(samples.tolist())

        unique_elements, counts = np.unique(flattened_res, return_counts=True)

        final_list = unique_elements[counts >= min_vote]

        idx_list.append(final_list)
    idx_list = np.array(idx_list, dtype=object)
    return idx_list


def merge_anomaly_seq(y, first_index, window_size, step_size, beta=0.5):
    """Get the final list of anomalous indices of a sequence when merging all rolling windows

    Args:
        y (ndarray):
            A 2-dimensional array containing anomalous indices of each window.
        first_index (ndarray):
            A 1-dimensional array contaning the first index of each window.
        window_size (int):
            Length of each window
        step_size (int):
            Indicating the number of steps the window moves forward each round.
        beta (float):
            Percent of windows needed for index to be anomalous. Default to `0.5`.

    Return:
        ndarray:
            A 1-dimensional array containing final anomalous indices.
    """
    anomalies = [arr + first_idx for (arr, first_idx) in zip(y, first_index)]

    min_vote = np.ceil(beta * window_size / step_size)

    flattened_res = np.concatenate(anomalies)

    unique_elements, counts = np.unique(flattened_res, return_counts=True)

    final_list = unique_elements[counts >= min_vote]

    return np.sort(final_list)


def idx2time(timestamp, y):
    """Convert list of indices into list of timestamp

    Args:
        sequence (DataFrame):
            Signal with timestamps and values.
        y (ndarray):
            A 1-dimensional array of indices.

    Returns:
        ndarray:
            A 1-dimensional array containing timestamps.
    """
    timestamp_list = timestamp[y]
    return timestamp_list


def timestamp2interval(y, timestamp, padding_size=50):
    """Convert list of timestamps to list of intervals by padding to both sides
    and merge overlapping

    Args:
        y (ndarray):
            A 1d array of point timestamps.
        timestamp (ndarray):
            List of full timestamp of the signal
        padding_size (int):
            Number of steps to pad on both sides of a timestamp point. Default to `50`.

    Returns:
        Dataframe:
            Dataframe of interval (start, end, score).
    """
    start, end = timestamp[0], timestamp[-1]
    interval = timestamp[1] - timestamp[0]
    intervals = []
    for timestamp in y:
        intervals.append((max(start, timestamp - padding_size * interval),
                         min(end, timestamp + padding_size * interval)))
    if not intervals:
        return []

    intervals.sort(key=lambda x: x[0])  # Sort intervals based on start time
    merged_intervals = [intervals[0]]  # Initialize merged intervals with the first interval

    for current_interval in intervals[1:]:
        previous_interval = merged_intervals[-1]

        # If the current interval overlaps with the previous one, merge them
        if current_interval[0] <= previous_interval[1]:
            previous_interval = (
                previous_interval[0], max(
                    previous_interval[1], current_interval[1]))
            merged_intervals[-1] = previous_interval
        else:
            merged_intervals.append(current_interval)  # Append the current interval if no overlap

    df = pd.DataFrame(merged_intervals, columns=['start', 'end'])
    df['score'] = 0
    return df
