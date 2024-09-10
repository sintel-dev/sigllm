import unittest

import numpy as np

from sigllm.primitives.postprocessing import aggregate_rolling_window


class FormatAsStringTest(unittest.TestCase):

    def test_aggregate_rolling_window_shape_1_1_5(self):

        data = np.array([[
            [1, 2, 3, 4, 5]
        ]])
        expected = np.array(
            [1, 2, 3, 4, 5]
        )

        output = aggregate_rolling_window(data)

        np.testing.assert_array_equal(output, expected)

    def test_aggregate_rolling_window_shape_1_3_5(self):

        data = np.array([[
            [1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6],
            [3, 4, 5, 6, 7]
        ]])
        expected = np.array(
            [2, 3, 4, 5, 6]
        )

        output = aggregate_rolling_window(data)

        np.testing.assert_array_equal(output, expected)

    def test_aggregate_rolling_window_shape_1_5_1(self):

        data = np.array([[
            [1],
            [2],
            [3],
            [4],
            [5]
        ]])
        expected = np.array(
            [3]
        )

        output = aggregate_rolling_window(data)

        np.testing.assert_array_equal(output, expected)

    def test_aggregate_rolling_window_shape_3_1_5(self):

        data = np.array([
            [[1, 2, 3, 4, 5]],
            [[2, 3, 4, 5, 6]],
            [[3, 4, 5, 6, 7]]
        ])
        expected = np.array(
            [1, 2, 3, 4, 5, 6, 7]
        )

        output = aggregate_rolling_window(data)

        np.testing.assert_array_equal(output, expected)

    def test_aggregate_rolling_window_shape_3_2_5(self):

        data = np.array([
            [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]],
            [[2, 3, 4, 5, 6], [3, 4, 5, 6, 7]],
            [[3, 4, 5, 6, 7], [4, 5, 6, 7, 8]]
        ])
        expected = np.array(
            [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
        )

        output = aggregate_rolling_window(data)

        np.testing.assert_array_equal(output, expected)
