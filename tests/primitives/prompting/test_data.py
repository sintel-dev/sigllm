# -*- coding: utf-8 -*-

import numpy as np
from pytest import fixture

from sigllm.primitives.prompting.data import rolling_window_sequences, sig2str


@fixture
def integers():
    return np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])


@fixture
def floats():
    return np.array([
        1.283,
        2.424,
        3.213,
        4.583,
        5.486,
        6.284,
        7.297,
        8.023,
        9.786
    ])


@fixture
def negatives():
    return np.array([
        -2.5,
        -1.5,
        0,
        1.5,
        2.5,
    ])


@fixture
def indices():
    return np.array([0, 1, 2, 3, 4, 5, 6])


@fixture
def values():
    return np.array([0.555, 2.345, 1.501, 5.903, 9.116, 3.068, 4.678])


@fixture
def window_size():
    return 3


@fixture
def step_size():
    return 1


def test_sig2str(integers):
    expected = '0,1,2,3,4,5,6,7,8'

    result = sig2str(integers)

    assert result == expected


def test_sig2str_noscale(integers):
    expected = '1,2,3,4,5,6,7,8,9'

    result = sig2str(integers, rescale=False)

    assert result == expected


def test_sig2str_decimal(integers):
    expected = '0,100,200,300,400,500,600,700,800'

    result = sig2str(integers, decimal=2)

    assert result == expected


def test_sig2str_sep(integers):
    expected = '0|1|2|3|4|5|6|7|8'

    result = sig2str(integers, sep='|')

    assert result == expected


def test_sig2str_space(integers):
    expected = '0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8'

    result = sig2str(integers, space=True)

    assert result == expected


def test_sig2str_float(floats):
    expected = '0,1,2,3,4,5,6,7,8'

    result = sig2str(floats)

    assert result == expected


def test_sig2str_float_decimal(floats):
    expected = '0,114,193,330,420,500,601,674,850'

    result = sig2str(floats, decimal=2)

    assert result == expected


def test_sig2str_negative_decimal(negatives):
    expected = '0,10,25,40,50'

    result = sig2str(negatives, decimal=1)

    assert result == expected


def test_rolling_window_sequences(values, indices, window_size, step_size):
    expected = (np.array([[0.555, 2.345, 1.501],
                          [2.345, 1.501, 5.903],
                          [1.501, 5.903, 9.116],
                          [5.903, 9.116, 3.068],
                          [9.116, 3.068, 4.678], ]),
                np.array([0, 1, 2, 3, 4]))

    result = rolling_window_sequences(values, indices, window_size, step_size)

    if len(result) != len(expected):
        raise AssertionError("Tuples has different length")

    for arr1, arr2 in zip(result, expected):
        np.testing.assert_equal(arr1, arr2)
