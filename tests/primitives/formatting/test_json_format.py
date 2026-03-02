import unittest

import numpy as np

from sigllm.primitives.formatting.json_format import JSONFormat


class JSONFormatFormatAsStringTest(unittest.TestCase):
    """Tests for JSONFormat.format_as_string."""

    def setUp(self):
        self.formatter = JSONFormat(trunc=5)

    def test_single_window_single_row_to_string(self):
        X = np.array([[[1, 2, 3]]])
        out = self.formatter.format_as_string(X)
        self.assertEqual(out, ['d0:1,d1:2,d2:3'])

    def test_single_window_multiple_rows_to_string(self):
        X = np.array([[[1, 2], [3, 4]]])
        out = self.formatter.format_as_string(X)
        self.assertEqual(out, ['d0:1,d1:2,d0:3,d1:4'])

    def test_multiple_windows_to_string(self):
        X = np.array([[[10, 20]], [[30, 40]]])
        out = self.formatter.format_as_string(X)
        self.assertEqual(out, ['d0:10,d1:20', 'd0:30,d1:40'])

    def test_multiple_windows_multiple_rows_to_string(self):
        X = np.array([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ])
        out = self.formatter.format_as_string(X)
        self.assertEqual(out, ['d0:1,d1:2,d0:3,d1:4', 'd0:5,d1:6,d0:7,d1:8'])

    def test_separator_kwarg_accepted(self):
        X = np.array([[[1, 2]]])
        out = self.formatter.format_as_string(X, separator=';')
        self.assertEqual(out, ['d0:1,d1:2'])


class JSONFormatFormatAsIntegerLegacyTest(unittest.TestCase):
    """Tests for JSONFormat.format_as_integer with trunc (legacy)."""

    def setUp(self):
        self.formatter = JSONFormat(trunc=2)

    def test_trunc_none_single_sample_to_integer(self):
        X = np.array([['d0:1,d1:2,d0:3,d1:4']])
        out = self.formatter.format_as_integer(X, trunc=None)
        self.assertEqual(out.shape, (1, 1, 2))
        np.testing.assert_array_equal(out[0, 0], [1, 3])

    def test_trunc_none_multiple_samples_to_integer(self):
        X = np.array([['d0:10,d1:11,d0:12', 'd0:20,d1:21']])
        out = self.formatter.format_as_integer(X, trunc=None)
        self.assertEqual(out.shape, (1, 2, 2))
        np.testing.assert_array_equal(out[0, 0], [10, 12])
        np.testing.assert_array_equal(out[0, 1], [20, None])

    def test_trunc_int_single_window_to_integer(self):
        X = np.array([['d0:1,d1:2,d0:3,d1:4,d0:5']])
        out = self.formatter.format_as_integer(X, trunc=3)
        np.testing.assert_array_equal(out, np.array([[[1, 3, 5]]]))

    def test_trunc_int_multiple_windows_to_integer(self):
        X = np.array([
            ['d0:1,d0:2,d0:3'],
            ['d0:4,d0:5,d0:6'],
        ])
        out = self.formatter.format_as_integer(X, trunc=2)
        expected = np.array([[[1, 2]], [[4, 5]]])
        np.testing.assert_array_equal(out, expected)

    def test_trunc_larger_than_values_fills_with_none(self):
        X = np.array([['d0:7,d1:8,d0:9']])
        out = self.formatter.format_as_integer(X, trunc=5)
        np.testing.assert_array_equal(out[0, 0], [7, 9, None, None, None])


class JSONFormatFormatAsIntegerStepsAheadTest(unittest.TestCase):
    """Tests for JSONFormat.format_as_integer with steps_ahead."""

    def setUp(self):
        self.formatter = JSONFormat(trunc=5)

    def test_steps_ahead_single_step(self):
        X = np.array([['d0:10,d1:11,d0:20,d1:21,d0:30']])
        out = self.formatter.format_as_integer(X, steps_ahead=[1, 2, 3])
        self.assertIn(1, out)
        self.assertIn(2, out)
        self.assertIn(3, out)
        np.testing.assert_array_equal(out[1], np.array([[10]]))
        np.testing.assert_array_equal(out[2], np.array([[20]]))
        np.testing.assert_array_equal(out[3], np.array([[30]]))

    def test_steps_ahead_missing_step_is_none(self):
        X = np.array([['d0:10,d1:11,d0:20']])
        out = self.formatter.format_as_integer(X, steps_ahead=[1, 2, 5])
        self.assertEqual(out[1][0, 0], 10)
        self.assertEqual(out[2][0, 0], 20)
        self.assertIsNone(out[5][0, 0])

    def test_steps_ahead_multiple_samples(self):
        X = np.array([['d0:1,d0:2,d0:3', 'd0:4,d0:5']])
        out = self.formatter.format_as_integer(X, steps_ahead=[2])
        np.testing.assert_array_equal(out[2], np.array([[2, 5]]))

    def test_steps_ahead_from_config(self):
        formatter = JSONFormat(trunc=1, steps_ahead=[1, 2])
        X = np.array([['d0:100,d0:200']])
        out = formatter.format_as_integer(X)
        np.testing.assert_array_equal(out[1], np.array([[100]]))
        np.testing.assert_array_equal(out[2], np.array([[200]]))


class JSONFormatExtractD0ValuesTest(unittest.TestCase):
    """Tests for d0 extraction (via format_as_integer)."""

    def setUp(self):
        self.formatter = JSONFormat(trunc=2)

    def test_only_d0_extracted(self):
        X = np.array([['d1:5,d0:1,d2:3,d0:2,d1:9']])
        out = self.formatter.format_as_integer(X)
        np.testing.assert_array_equal(out[0, 0], [1, 2])

    def test_no_d0_fills_with_none(self):
        X = np.array([['d1:1,d2:2,d1:3']])
        out = self.formatter.format_as_integer(X)
        np.testing.assert_array_equal(out[0, 0], [None, None])

    def test_no_d0_steps_ahead_returns_none(self):
        X = np.array([['d1:1,d2:2']])
        out = self.formatter.format_as_integer(X, steps_ahead=[1])
        np.testing.assert_array_equal(out[1][0, 0], [None])


class JSONFormatTargetDimTest(unittest.TestCase):
    """Tests for target_column parameter in format_as_integer."""

    def setUp(self):
        self.formatter = JSONFormat()

    def test_target_column_one(self):
        X = np.array([['d0:1,d1:10,d0:2,d1:20']])
        out = self.formatter.format_as_integer(X, trunc=None, target_column=1)
        np.testing.assert_array_equal(out[0, 0], [10, 20])

    def test_target_column_with_trunc(self):
        X = np.array([['d0:1,d1:10,d0:2,d1:20,d0:3,d1:30']])
        out = self.formatter.format_as_integer(X, trunc=2, target_column=1)
        np.testing.assert_array_equal(out[0, 0], [10, 20])

    def test_target_column_with_steps_ahead(self):
        X = np.array([['d0:1,d1:10,d0:2,d1:20,d0:3,d1:30']])
        out = self.formatter.format_as_integer(X, steps_ahead=[1, 2], target_column=1)
        self.assertEqual(out[1][0, 0], 10)
        self.assertEqual(out[2][0, 0], 20)

    def test_target_column_from_config(self):
        formatter = JSONFormat(target_column=1)
        X = np.array([['d0:1,d1:10,d0:2,d1:20']])
        out = formatter.format_as_integer(X, trunc=None)
        np.testing.assert_array_equal(out[0, 0], [10, 20])
