# -*- coding: utf-8 -*-

"""
Prompter post-processing module

This module contains functions that help filter LLMs results to get the final anomalies.
"""

import numpy as np


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


def find_anomalies_in_windows(y, alpha=0.5):
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


def merge_anomalous_sequences(y, first_index, window_size, step_size, beta=0.5):
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


def format_anomalies(y, timestamp, padding_size=50):
    """Convert list of anomalous indices to list of intervals by padding to both sides
    and merge overlapping

    Args:
        y (ndarray):
            A 1-dimensional array of indices.
        timestamp (ndarray):
            List of full timestamp of the signal
        padding_size (int):
            Number of steps to pad on both sides of a timestamp point. Default to `50`.

    Returns:
        List[Tuple]:
            List of intervals (start, end, score).
    """
    y = timestamp[y]  # Convert list of indices into list of timestamps
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

    merged_intervals = [(interval[0], interval[1], 0) for interval in merged_intervals]
    return merged_intervals
