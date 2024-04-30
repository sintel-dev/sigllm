# -*- coding: utf-8 -*-

"""
Result post-processing module.

This module contains functions that help convert model responses back to indices and timestamps.
"""
import numpy as np


def str2sig(text, sep=',', decimal=0):
    """Convert a text string to a signal.

    Convert a string containing digits into an array of numbers.

    Args:
        text (str):
            A string containing signal values.
        sep (str):
            String that was used to separate each element in text, Default to `","`.
        decimal (int):
            Number of decimal points to shift each element in text to. Default to `0`.

    Returns:
        numpy.ndarray:
            A 1-dimensional array containing parsed elements in `text`.
    """
    # Remove all characters from text except the digits and sep and decimal point
    text = ''.join(i for i in text if (i.isdigit() or i == sep or i == '.'))
    values = np.fromstring(text, dtype=float, sep=sep)
    return values * 10**(-decimal)


def str2idx(text, len_seq, sep=','):
    """Convert a text string to indices.

    Convert a string containing digits into an array of indices.

    Args:
        text (str):
            A string containing indices values.
        len_seq (int):
            The length of processed sequence
        sep (str):
            String that was used to separate each element in text, Default to `","`.

    Returns:
        numpy.ndarray:
            A 1-dimensional array containing parsed elements in `text`.
    """
    # Remove all characters from text except the digits and sep
    text = ''.join(i for i in text if (i.isdigit() or i == sep))

    values = np.fromstring(text, dtype=int, sep=sep)

    # Remove indices that exceed the length of sequence
    values = values[values < len_seq]
    return values


def get_anomaly_list_within_seq(res_list, alpha=0.5):
    """Get the final list of anomalous indices of a sequence

    Choose anomalous index in the sequence based on multiple LLM responses

    Args:
        res_list (List[numpy.ndarray]):
            A list of 1-dimensional array containing anomous indices output by LLM
        alpha (float):
            Percentage of votes needed for an index to be deemed anomalous. Default: 0.5

    Returns:
        numpy.ndarray:
            A 1-dimensional array containing final anomalous indices
    """
    min_vote = np.ceil(alpha * len(res_list))

    flattened_res = np.concatenate(res_list)

    unique_elements, counts = np.unique(flattened_res, return_counts=True)

    final_list = unique_elements[counts >= min_vote]

    return final_list


def merge_anomaly_seq(anomalies, start_indices, window_size, step_size, beta=0.5):
    """Get the final list of anomalous indices of a sequence when merging all rolling windows

    Args:
        anomalies (List[numpy.ndarray]):
            A list of 1-dimensional array containing anomous indices of each window
        start_indices (numpy.ndarray):
            A 1-dimensional array contaning the first index of each window
        window_size (int):
            Length of each window
        step_size (int):
            Indicating the number of steps the window moves forward each round.
        beta (float):
            Percentage of containing windows needed for index to be deemed anomalous. Default: 0.5

    Return:
        numpy.ndarray:
            A 1-dimensional array containing final anomalous indices
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
            Signal with timestamps and values
        idx_list (numpy.ndarray):
            A 1-dimensional array of indices

    Returns:
        numpy.ndarray:
            A 1-dimensional array containing timestamps
    """
    return sequence.iloc[idx_list].timestamp.to_numpy()
