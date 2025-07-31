# -*- coding: utf-8 -*-
import unittest

import numpy as np
from pytest import fixture

from sigllm.primitives.prompting.anomalies import (
    _clean_response,
    _parse_interval_response,
    _parse_list_response,
    find_anomalies_in_windows,
    format_anomalies,
    merge_anomalous_sequences,
    parse_anomaly_response,
    val2idx,
)


@fixture
def anomaly_list_within_seq():
    return np.array(
        [
            [np.array([0, 3]), np.array([1]), np.array([1, 2])],
            [np.array([0]), np.array([1, 4]), np.array([2, 3])],
            [np.array([0, 2]), np.array([]), np.array([0, 1])],
        ],
        dtype=object,
    )


@fixture
def anomaly_list_across_seq():
    return np.array(
        [np.array([0]), np.array([1, 2]), np.array([0, 2]), np.array([1, 2]), np.array([1])],
        dtype=object,
    )


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
def idx_list():
    return np.array([32, 545, 689, 1103, 1134])


@fixture
def anomalous_val():
    return np.array(
        [[np.array([0, 3]), np.array([])], [np.array([2]), np.array([4])]], dtype=object
    )


@fixture
def windows():
    return np.array([[0, 1, 0, 3], [3, 2, 6, 2]])


@fixture
def timestamp():
    return np.array(range(1000, 13000, 10))


def test_ano_within_windows(anomaly_list_within_seq):
    expected = np.array([np.array([1]), np.array([]), np.array([0])], dtype=object)

    result = find_anomalies_in_windows(anomaly_list_within_seq)

    for r, e in zip(result, expected):
        np.testing.assert_equal(r, e)


def test_merge_anomaly_seq(anomaly_list_across_seq, first_indices, window_size, step_size):
    expected = np.array([2, 4, 5])

    result = merge_anomalous_sequences(
        anomaly_list_across_seq, first_indices, window_size, step_size
    )

    np.testing.assert_equal(result, expected)


# val2idx
def test_val2idx(anomalous_val, windows):
    expected = np.array(
        [[np.array([0, 2, 3]), np.array([])], [np.array([1, 3]), np.array([])]], dtype=object
    )
    result = val2idx(anomalous_val, windows)

    for r_list, e_list in zip(result, expected):
        for r, e in zip(r_list, e_list):
            np.testing.assert_equal(r, e)


# timestamp2interval
def test_format_anomalies(idx_list, timestamp):
    expected = [(1000, 1820, 0), (5950, 6950, 0), (7390, 8390, 0), (11530, 12840, 0)]
    result = format_anomalies(idx_list, timestamp)

    assert expected == result


def test_clean_response_no_anomalies():
    test_cases = [
        'no anomalies',
        'NO ANOMALIES',
        '  no anomalies  ',
        'There are no anomalies in this data',
        'No anomaly detected',
        '  No anomaly  ',
    ]
    for text in test_cases:
        assert _clean_response(text) == ''


def test_clean_response_with_anomalies():
    test_cases = [
        ('[1, 2, 3]', '[1, 2, 3]'),
        ('  [1, 2, 3]  ', '[1, 2, 3]'),
        ('Anomalies found at [1, 2, 3]', 'anomalies found at [1, 2, 3]'),
        ('ANOMALIES AT [1, 2, 3]', 'anomalies at [1, 2, 3]'),
    ]
    for input_text, expected in test_cases:
        assert _clean_response(input_text) == expected


def test_parse_list_response_valid_cases():
    test_cases = [
        ('[1, 2, 3]', '1,2,3'),
        ('  [1, 2, 3]  ', '1,2,3'),
        ('Anomalies found at [1, 2, 3]', '1,2,3'),
        ('[1,2,3]', '1,2,3'),
        ('[1, 2, 3, 4, 5]', '1,2,3,4,5'),
    ]
    for input_text, expected in test_cases:
        assert _parse_list_response(input_text) == expected


def test_parse_list_response_invalid_cases():
    test_cases = [
        'no anomalies',
        '[]',
        '[ ]',
        'text with [no numbers]',
        'text with [letters, and, symbols]',
        '   ',
    ]
    for text in test_cases:
        assert _parse_list_response(text) == ''


def test_parse_list_response_edge_cases():
    test_cases = [
        ('[1,2,3,]', '1,2,3'),  # trailing comma
        ('[1,,2,3]', '1,2,3'),  # double comma
        ('[1, 2, 3], [5]', '1,2,3'),  # two lists
    ]
    for input_text, expected in test_cases:
        assert _parse_list_response(input_text) == expected


def test_parse_interval_response_valid_cases():
    test_cases = [
        ('[[1, 3]]', [1, 2, 3]),
        ('  [[1, 3]]  ', [1, 2, 3]),
        ('Anomalies found at [[1, 3]]', [1, 2, 3]),
        ('[[1, 3], [5, 7]]', [1, 2, 3, 5, 6, 7]),
        ('[[1, 3], [5, 7], [8, 9]]', [1, 2, 3, 5, 6, 7, 8, 9]),
        ('[[1, 3], [4, 6],]', [1, 2, 3, 4, 5, 6]),
        ('[[1, 2], [3]]', [1, 2]),
        ('[[1,,3]]', [1, 2, 3]),
        ('[[0, 10]]', list(range(11))),
    ]
    for input_text, expected in test_cases:
        assert _parse_interval_response(input_text) == expected


def test_parse_interval_response_invalid_cases():
    test_cases = [
        '[]',
        '[[]]',
        'text with [no numbers]',
        '[[1]]',  # single number instead of pair
        '[[1, 2, 3]]',  # triple instead of pair
    ]
    for text in test_cases:
        assert _parse_interval_response(text) == []


def test_parse_interval_response_multiple_matches():
    test_cases = [
        ('Found [[1, 3]] and [[5, 7]]', [1, 2, 3, 5, 6, 7]),
        ('[[1, 2]] in first part and [[3, 4]] in second', [1, 2, 3, 4]),
        ('Multiple intervals: [[1, 3]], [[4, 6]], [[7, 9]]', [1, 2, 3, 4, 5, 6, 7, 8, 9]),
        ('[[1, 2]] and [[1, 2]] and [[1, 2]]', [1, 2, 1, 2, 1, 2]),
    ]
    for input_text, expected in test_cases:
        assert _parse_interval_response(input_text) == expected


class ParseAnomalyResponseTest(unittest.TestCase):
    def test_no_anomalies(self):
        data = [['Answer: no anomalies'], ['Answer: no anomaly'], ['no anomaly, with extra']]
        expected = [[''], [''], ['']]

        output = parse_anomaly_response(data)
        self.assertEqual(output, expected)

    def test_single_anomaly(self):
        data = [['Answer: [123]'], ['Answer: [456]', 'answer: [789]']]
        expected = [['123'], ['456', '789']]

        output = parse_anomaly_response(data)
        self.assertEqual(output, expected)

    def test_multiple_anomalies(self):
        data = [['Answer: [123, 456, 789]'], ['Answer: [111, 222, 333]']]
        expected = [['123,456,789'], ['111,222,333']]

        output = parse_anomaly_response(data)
        self.assertEqual(output, expected)

    def test_mixed_responses(self):
        data = [['Answer: no anomalies', 'Answer: [123, 456]'], ['Answer: [789]', 'no anomaly']]
        expected = [['', '123,456'], ['789', '']]

        output = parse_anomaly_response(data)
        self.assertEqual(output, expected)

    def test_different_formats(self):
        data = [
            ['Answer: [123, 456]', 'Answer: [ 789 , 101 ]'],
            ['Answer: [1,2,3]', 'Answer: [ 4 , 5 , 6 ]'],
        ]
        expected = [['123,456', '789,101'], ['1,2,3', '4,5,6']]

        output = parse_anomaly_response(data)
        self.assertEqual(output, expected)

    def test_empty_responses(self):
        data = [[''], ['Answer: no anomalies'], ['answer'], ['no anomly']]
        expected = [[''], [''], [''], ['']]

        output = parse_anomaly_response(data)
        self.assertEqual(output, expected)

    def test_invalid_format(self):
        data = [['Answer: invalid format'], ['Answer: [123, abc]']]
        expected = [[''], ['']]

        output = parse_anomaly_response(data)
        self.assertEqual(output, expected)
