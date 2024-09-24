import numpy as np
from pytest import fixture

from sigllm.primitives.prompting.timeseries_preprocessing import rolling_window_sequences


@fixture
def values():
    return np.array([0.555, 2.345, 1.501, 5.903, 9.116, 3.068, 4.678])


@fixture
def window_size():
    return 3


@fixture
def step_size():
    return 1


def test_rolling_window_sequences(values, window_size, step_size):
    expected = (np.array([[0.555, 2.345, 1.501],
                          [2.345, 1.501, 5.903],
                          [1.501, 5.903, 9.116],
                          [5.903, 9.116, 3.068],
                          [9.116, 3.068, 4.678]]),
                np.array([0, 1, 2, 3, 4]),
                3,
                1)

    result = rolling_window_sequences(values, window_size, step_size)

    if len(result) != len(expected):
        raise AssertionError("Tuples has different length")

    for arr1, arr2 in zip(result, expected):
        np.testing.assert_equal(arr1, arr2)
