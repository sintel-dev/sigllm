# -*- coding: utf-8 -*-

import numpy as np
from pytest import fixture

from sigllm.data import sig2str, str2sig


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
def text():
    return '1,2,3,4,5,6,7,8,9'


def test_sig2str(integers):
    expected = '1,2,3,4,5,6,7,8,9'

    result = sig2str(integers)

    assert result == expected


def test_sig2str_decimal(integers):
    expected = '100,200,300,400,500,600,700,800,900'

    result = sig2str(integers, decimal=2)

    assert result == expected


def test_sig2str_sep(integers):
    expected = '1|2|3|4|5|6|7|8|9'

    result = sig2str(integers, sep='|')

    assert result == expected


def test_sig2str_space(integers):
    expected = '1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9'

    result = sig2str(integers, space=True)

    assert result == expected


def test_sig2str_float(floats):
    expected = '1,2,3,4,5,6,7,8,9'

    result = sig2str(floats)

    assert result == expected


def test_sig2str_float_decimal(floats):
    expected = '128,242,321,458,548,628,729,802,978'

    result = sig2str(floats, decimal=2)

    assert result == expected


def test_str2sig(text):
    expected = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

    result = str2sig(text)

    np.testing.assert_equal(result, expected)


def test_str2sig_decimal(text):
    expected = np.array([.01, .02, .03, .04, .05, .06, .07, .08, .09])

    result = str2sig(text, decimal=2)

    np.testing.assert_equal(result, expected)
