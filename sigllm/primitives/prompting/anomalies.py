# -*- coding: utf-8 -*-

"""Prompter post-processing module.

This module contains functions that help filter LLMs results to get the final anomalies.
"""

import ast
import re

import numpy as np

PATTERN = r'\[([\d\s,]+)\]'


def _clean_response(text):
    text = text.strip().lower()
    text = re.sub(r',+', ',', text)

    if 'no anomalies' in text or 'no anomaly' in text:
        return ''

    return text


def _parse_list_response(text):
    clean = _clean_response(text)

    # match anything that consists of digits and commas
    match = re.search(PATTERN, clean)

    if match:
        values = match.group(1)
        values = [val.strip() for val in values.split(',') if val.strip()]
        return ','.join(values)

    return ''


def _parse_interval_response(text):
    clean = _clean_response(text)
    match = re.finditer(PATTERN, clean)

    if match:
        values = list()
        for m in match:
            interval = ast.literal_eval(m.group())
            if len(interval) == 2:
                start, end = ast.literal_eval(m.group())
                values.extend(list(range(start, end + 1)))

        return values

    return []


def parse_anomaly_response(X, interval=False):
    """Parse a list of lists of LLM responses to extract anomaly values and format them as strings.

    Args:
        X (List[List[str]]):
            List of lists of response texts from the LLM in the format
            "Answer: no anomalies" or "Answer: [val1, val2, ..., valN]."
            values must be within brackets.
        interval (bool):
            Whether to parse the response as a list "Answer: [val1, val2, ..., valN]."
            or list of intervals "Answer: [[s1, e1], [s2, e2], ..., [sn, en]]."

    Returns:
        List[List[str]]:
            List of lists of parsed responses where each element is either
            "val1,val2,...,valN" if anomalies are found, or empty string if
            no anomalies are present.
    """
    method = _parse_list_response
    if interval:
        method = _parse_interval_response

    result = []
    for response_list in X:
        parsed_list = [method(response) for response in response_list]
        result.append(parsed_list)

    return result


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
    """Get the final list of anomalous indices of each window.

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

        flattened_res = np.concatenate(samples.tolist())

        unique_elements, counts = np.unique(flattened_res, return_counts=True)

        final_list = unique_elements[counts >= min_vote]

        idx_list.append(final_list)
    idx_list = np.array(idx_list, dtype=object)

    return idx_list


def merge_anomalous_sequences(y, first_index, window_size, step_size, beta=0.5):
    """Merge sequences.

    Get the final list of anomalous indices of a sequence when merging all rolling windows.

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
    """Format anomalies into intervals.

    Convert list of anomalous indices to list of intervals
    by padding to both sides and merge overlapping.

    Args:
        y (ndarray):
            A 1-dimensional array of indices. Can be empty if no anomalies are found.
        timestamp (ndarray):
            List of full timestamp of the signal.
        padding_size (int):
            Number of steps to pad on both sides of a timestamp point. Default to `50`.

    Returns:
        List[Tuple]:
            List of intervals (start, end, score). Empty list if no anomalies are found.
    """
    # Handle empty array case
    if len(y) == 0:
        return []

    y = timestamp[y]  # Convert list of indices into list of timestamps
    start, end = timestamp[0], timestamp[-1]
    interval = timestamp[1] - timestamp[0]
    intervals = []
    for timestamp in y:
        intervals.append((
            max(start, timestamp - padding_size * interval),
            min(end, timestamp + padding_size * interval),
        ))
    if not intervals:
        return []

    intervals.sort(key=lambda x: x[0])  # Sort intervals based on start time
    merged_intervals = [intervals[0]]  # Initialize merged intervals with the first interval

    for current_interval in intervals[1:]:
        previous_interval = merged_intervals[-1]

        # If the current interval overlaps with the previous one, merge them
        if current_interval[0] <= previous_interval[1]:
            previous_interval = (
                previous_interval[0],
                max(previous_interval[1], current_interval[1]),
            )
            merged_intervals[-1] = previous_interval
        else:
            merged_intervals.append(current_interval)  # Append the current interval if no overlap

    merged_intervals = [(interval[0], interval[1], 0) for interval in merged_intervals]

    return merged_intervals
