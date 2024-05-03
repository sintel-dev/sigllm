# -*- coding: utf-8 -*-

"""
Prompter post-processing module

This module contains functions that help filter LLMs results to get the final anomalies.
"""

import numpy as np

def val2idx(vals, windows): 
    """Convert detected anomalies values into indices.
    
    Convert windows of detected anomalies values into an array of all indices
    in the input sequence that have those values. 
    
    Args: 
        vals (List[ndarray]]): 
            A list nd array containing detected anomalous values from different
            responses of one window in one sample response. 
        windows (ndarray):
            rolling window sequences.      
    Returns: 
        List([ndarray]):
            A list of nd array containing detected anomalous indices from different
            responses of one window in one sample response.
    """

    idx_list = []
    for anomalies_list, seq in zip(vals, windows): 
        idx_win_list = []
        for anomalies in anomalies_list:
            mask = np.isin(seq, anomalies)
            indices = np.where(mask)[0]
            idx_win_list.append(indices)
        idx_win_list = np.array(idx_win_list)
        idx_list.append(idx_win_list)
    return idx_list

def ano_within_windows(idx_win_list, alpha=0.5):
    """Get the final list of anomalous indices of a sequence

    Choose anomalous index in the sequence based on multiple LLM responses

    Args:
        idx_win_list (List[List[numpy.ndarray]]):
            A list of lists of 1d array containing detected anomalous indices of
            one window in one sample response.
        alpha (float):
            Percentage of votes needed for an index to be deemed anomalous. Default to `0.5`.

    Returns:
        List[numpy.ndarray]:
            A list of 1-dimensional array containing final anomalous indices of each windows.
    """
    
    idx_list = []
    for samples in idx_win_list:
        min_vote = np.ceil(alpha * len(samples))

        flattened_res = np.flatten(samples)

        unique_elements, counts = np.unique(flattened_res, return_counts=True)

        final_list = unique_elements[counts >= min_vote]

        idx_list.append(final_list)

    return idx_list

def merge_anomaly_seq(anomalies, start_indices, window_size, step_size, beta=0.5):
    """Get the final list of anomalous indices of a sequence when merging all rolling windows

    Args:
        anomalies (List[numpy.ndarray]):
            A list of 1-dimensional array containing anomous indices of each window.
        start_indices (numpy.ndarray):
            A 1-dimensional array contaning the first index of each window.
        window_size (int):
            Length of each window.
        step_size (int):
            Indicating the number of steps the window moves forward each round.
        beta (float):
            Percentage of containing windows needed for index to be deemed anomalous. Default to `0.5`.

    Return:
        numpy.ndarray:
            A 1-dimensional array containing final anomalous indices.
    """
    anomalies = [arr + first_idx for (arr, first_idx) in zip(anomalies, start_indices)]

    min_vote = np.ceil(beta * window_size / step_size)

    flattened_res = np.concatenate(anomalies)

    unique_elements, counts = np.unique(flattened_res, return_counts=True)

    final_list = unique_elements[counts >= min_vote]

    return np.sort(final_list)

def idx2time(sequence, idx_list):
    """Convert list of indices into list of timestamp

    Args:
        sequence (pandas.Dataframe):
            Signal with timestamps and values.
        idx_list (numpy.ndarray):
            A 1-dimensional array of indices.

    Returns:
        numpy.ndarray:
            A 1-dimensional array containing timestamps.
    """
    return sequence.iloc[idx_list].timestamp.to_numpy()

def timestamp2interval(timestamp_list, interval, start, end, padding_size = 50): 
    """Convert list of timestamps to list of intervals by padding to both sides
    and merge overlapping 
    
    Args: 
        timestamp_list (List[timestamp]): 
            A list of point timestamps.
        interval (int):
            The fixed gap between two consecutive timestamps of the time series.
        start (timestamp): 
            The start timestamp of the time series.
        end (timestamp): 
            The end timestamp of the time series.
        padding_size (int): 
            Number of steps to pad on both sides of a timestamp point.
    """
    intervals = []
    for timestamp in timestamp_list: 
        intervals.append((max(start, timestamp-padding_size*interval), min(end, timestamp+padding_size*interval)))

    if not intervals:
        return []

    intervals.sort(key=lambda x: x[0])  # Sort intervals based on start time
    merged_intervals = [intervals[0]]  # Initialize merged intervals with the first interval

    for current_interval in intervals[1:]:
        previous_interval = merged_intervals[-1]
        
        # If the current interval overlaps with the previous one, merge them
        if current_interval[0] <= previous_interval[1]:
            previous_interval = (previous_interval[0], max(previous_interval[1], current_interval[1]))
            merged_intervals[-1] = previous_interval
        else:
            merged_intervals.append(current_interval)  # Append the current interval if no overlap

    return merged_intervals