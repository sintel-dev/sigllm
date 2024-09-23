# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from pytest import fixture

from sigllm.primitives.prompting.anomalies import (
    ano_within_windows, idx2time, merge_anomaly_seq, timestamp2interval, val2idx,)


@fixture
def anomaly_list_within_seq():
    return np.array([[np.array([0, 3]), np.array([1]), np.array([1, 2])],
                     [np.array([0]), np.array([1, 4]), np.array([2, 3])],
                     [np.array([0, 2]), np.array([]), np.array([0, 1])]], dtype=object)


@fixture
def anomaly_list_across_seq():
    return np.array([np.array([0]),
                     np.array([1, 2]),
                     np.array([0, 2]),
                     np.array([1, 2]),
                     np.array([1])], dtype=object)


@fixture
def first_indices():
    return np.array([0, 1, 2, 3, 4])


@fixture
def window_size():
    return 3


@fixture
def step_size():
    return 1


@fixture
def timestamp():
    return np.array([1222819200, 1222840800, 1222862400, 1222884000, 1222905600])


@fixture
def idx_list():
    return np.array([0, 1, 3])


@fixture
def anomalous_val():
    return np.array([[np.array([0, 3]), np.array([])],
                    [np.array([2]), np.array([4])]], dtype=object)


@fixture
def windows():
    return np.array([[0, 1, 0, 3],
                     [3, 2, 6, 2]])


@fixture
def point_timestamp():
    return np.array([1320, 6450, 7890, 12030, 12340])


@fixture
def timestamp1():
    return np.array(range(1000, 13000, 10))


def test_ano_within_windows(anomaly_list_within_seq):
    expected = np.array([np.array([1]),
                         np.array([]),
                         np.array([0])], dtype=object)

    result = ano_within_windows(anomaly_list_within_seq)

    for r, e in zip(result, expected):
        np.testing.assert_equal(r, e)


def test_merge_anomaly_seq(anomaly_list_across_seq, first_indices, window_size, step_size):
    expected = np.array([2, 4, 5])

    result = merge_anomaly_seq(anomaly_list_across_seq, first_indices, window_size, step_size)

    np.testing.assert_equal(result, expected)


def test_idx2time(timestamp, idx_list):
    expected = np.array([1222819200, 1222840800, 1222884000])

    result = idx2time(timestamp, idx_list)

    np.testing.assert_equal(result, expected)


# val2idx
def test_val2idx(anomalous_val, windows):
    expected = np.array([[np.array([0, 2, 3]), np.array([])],
                         [np.array([1, 3]), np.array([])]], dtype=object)
    result = val2idx(anomalous_val, windows)

    for r_list, e_list in zip(result, expected):
        for r, e in zip(r_list, e_list):
            np.testing.assert_equal(r, e)

# timestamp2interval


def test_timestamp2interval(point_timestamp, timestamp1):
    expected = pd.DataFrame([(1000, 1820, 0), (5950, 6950, 0), (7390, 8390, 0),
                            (11530, 12840, 0)], columns=['start', 'end', 'score'])
    result = timestamp2interval(point_timestamp, timestamp1)

    pd.testing.assert_frame_equal(expected, result, check_dtype = False)
