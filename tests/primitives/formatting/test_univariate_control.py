import unittest

import numpy as np

from sigllm.primitives.formatting.univariate_control import UnivariateControl


class UnivariateControlFormatAsStringTest(unittest.TestCase):
    """Tests for UnivariateControl.format_as_string."""

    def setUp(self):
        self.formatter = UnivariateControl()

    def test_single_window_single_row_to_string(self):
        X = np.array([[[10, 20]]])
        out = self.formatter.format_as_string(X)
        self.assertEqual(out, ['10'])

    def test_single_window_multiple_rows_to_string(self):
        X = np.array([[[1, 100], [2, 200], [3, 300]]])
        out = self.formatter.format_as_string(X)
        self.assertEqual(out, ['1,2,3'])

    def test_multiple_windows_to_string(self):
        X = np.array([[[1, 10], [2, 20]], [[3, 30], [4, 40]]])
        out = self.formatter.format_as_string(X)
        self.assertEqual(out, ['1,2', '3,4'])

    def test_custom_separator(self):
        X = np.array([[[1, 10], [2, 20], [3, 30]]])
        out = self.formatter.format_as_string(X, separator=';')
        self.assertEqual(out, ['1;2;3'])

    def test_target_column_one(self):
        X = np.array([[[1, 10], [2, 20]]])
        out = self.formatter.format_as_string(X, target_column=1)
        self.assertEqual(out, ['10,20'])

    def test_target_column_from_config(self):
        formatter = UnivariateControl(target_column=1)
        X = np.array([[[1, 10], [2, 20]]])
        out = formatter.format_as_string(X)
        self.assertEqual(out, ['10,20'])


class UnivariateControlFormatAsIntegerTest(unittest.TestCase):
    """Tests for UnivariateControl.format_as_integer."""

    def setUp(self):
        self.formatter = UnivariateControl()

    def test_single_entry_single_value_to_integer(self):
        X = [['42']]
        out = self.formatter.format_as_integer(X)
        np.testing.assert_array_equal(out[0, 0], np.array([42]))

    def test_single_entry_multiple_values_to_integer(self):
        X = [['1,2,3']]
        out = self.formatter.format_as_integer(X)
        np.testing.assert_array_equal(out[0, 0], np.array([1, 2, 3]))

    def test_multiple_entries_to_integer(self):
        X = [['1,2,3', '4,5,6'], ['11,12,13', '14,15,16']]
        out = self.formatter.format_as_integer(X)
        np.testing.assert_array_equal(out[0, 0], np.array([1, 2, 3]))
        np.testing.assert_array_equal(out[0, 1], np.array([4, 5, 6]))
        np.testing.assert_array_equal(out[1, 0], np.array([11, 12, 13]))
        np.testing.assert_array_equal(out[1, 1], np.array([14, 15, 16]))

    def test_trunc_limits_values(self):
        X = [['10,20,30,40']]
        out = self.formatter.format_as_integer(X, trunc=2)
        np.testing.assert_array_equal(out[0, 0], np.array([10, 20]))

    def test_custom_separator(self):
        X = [['10;20;30']]
        out = self.formatter.format_as_integer(X, separator=';')
        np.testing.assert_array_equal(out[0, 0], np.array([10, 20, 30]))


class UnivariateControlRoundTripTest(unittest.TestCase):
    """Round-trip: format_as_string then format_as_integer."""

    def setUp(self):
        self.formatter = UnivariateControl()

    def test_round_trip_default_target_column(self):
        X = np.array([[[1, 10], [2, 20]]])
        strings = self.formatter.format_as_string(X)
        X_in = [[s] for s in strings]
        out = self.formatter.format_as_integer(X_in)
        np.testing.assert_array_equal(out[0][0], np.array([1, 2]))

    def test_round_trip_target_column_one(self):
        X = np.array([[[1, 10], [2, 20]]])
        strings = self.formatter.format_as_string(X, target_column=1)
        X_in = [[s] for s in strings]
        out = self.formatter.format_as_integer(X_in)
        np.testing.assert_array_equal(out[0][0], np.array([10, 20]))

    def test_round_trip_multiple_windows(self):
        X = np.array([[[1, 10], [2, 20]], [[3, 30], [4, 40]]])
        strings = self.formatter.format_as_string(X, target_column=1)
        X_in = [[s] for s in strings]
        out = self.formatter.format_as_integer(X_in)
        np.testing.assert_array_equal(out[0][0], np.array([10, 20]))
        np.testing.assert_array_equal(out[1][0], np.array([30, 40]))
