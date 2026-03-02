import unittest

import numpy as np

from sigllm.primitives.formatting.value_concatenation import ValueConcatenation


class ValueConcatenationFormatAsStringTest(unittest.TestCase):
    """Tests for ValueConcatenation.format_as_string (value concatenation)."""

    def setUp(self):
        self.formatter = ValueConcatenation()

    def test_single_window_single_row(self):
        X = np.array([[[10]]])
        out = self.formatter.format_as_string(X)
        self.assertEqual(out, ["10"])

    def test_single_window_multiple_rows(self):
        X = np.array([[[1], [2], [3]]])
        out = self.formatter.format_as_string(X)
        self.assertEqual(out, ["1,2,3"])

    def test_single_window_multiple_rows_multiple_dims(self):
        X = np.array([[[1, 2], [3, 4]]])
        out = self.formatter.format_as_string(X)
        self.assertEqual(out, ["1,2,3,4"])

    def test_multiple_windows(self):
        X = np.array([[[1], [2]], [[3], [4]]])
        out = self.formatter.format_as_string(X)
        self.assertEqual(out, ["1,2", "3,4"])

    def test_custom_separator(self):
        X = np.array([[[1], [2], [3]]])
        out = self.formatter.format_as_string(X, separator=";")
        self.assertEqual(out, ["1;2;3"])

    def test_flattens_all_values_in_window(self):
        X = np.array([[[100, 200], [300, 400]]])
        out = self.formatter.format_as_string(X)
        self.assertEqual(out, ["100,200,300,400"])


class ValueConcatenationFormatAsIntegerTest(unittest.TestCase):
    """Tests for ValueConcatenation.format_as_integer (parsing concatenated values)."""

    def setUp(self):
        self.formatter = ValueConcatenation(num_dims=1)
        print(self.formatter.config)

    def test_single_entry_single_value(self):
        X = [["42"]]
        out = self.formatter.format_as_integer(X)
        self.assertEqual(out.shape, (1, 1, 1))
        np.testing.assert_array_equal(out[0, 0], np.array([42]))

    def test_single_entry_multiple_values(self):
        X = [["1,2,3"]]
        out = self.formatter.format_as_integer(X)
        self.assertEqual(out.shape, (1, 1, 3))
        np.testing.assert_array_equal(out[0, 0], np.array([1, 2, 3]))

    def test_multiple_entries_equal_length(self):
        X = [["1,2,3", "4,5,6"], ["11,12,13", "14,15,16"]]
        out = self.formatter.format_as_integer(X)
        self.assertEqual(out.shape, (2, 2, 3))
        np.testing.assert_array_equal(out[0, 0], np.array([1, 2, 3]))
        np.testing.assert_array_equal(out[0, 1], np.array([4, 5, 6]))
        np.testing.assert_array_equal(out[1, 0], np.array([11, 12, 13]))
        np.testing.assert_array_equal(out[1, 1], np.array([14, 15, 16]))

    def test_multiple_entries_unequal_length(self):
        X = [["1,2,3", "4,5"], ["6", "7,8"]]
        out = self.formatter.format_as_integer(X)
        self.assertEqual(out.shape, (2, 2))
        np.testing.assert_array_equal(out[0, 0], np.array([1, 2, 3]))
        np.testing.assert_array_equal(out[0, 1], np.array([4, 5]))
        np.testing.assert_array_equal(out[1, 0], np.array([6]))
        np.testing.assert_array_equal(out[1, 1], np.array([7, 8]))

    def test_trunc_limits_values(self):
        X = [["10,20,30,40"]]
        out = self.formatter.format_as_integer(X, trunc=2)
        self.assertEqual(out.shape, (1, 1, 2))
        np.testing.assert_array_equal(out[0, 0], np.array([10, 20]))

    def test_trunc_none_keeps_all(self):
        X = [["7,8,9"]]
        out = self.formatter.format_as_integer(X, trunc=None)
        np.testing.assert_array_equal(out[0, 0], np.array([7, 8, 9]))

    def test_custom_separator(self):
        X = [["10;20;30"]]
        out = self.formatter.format_as_integer(X, separator=";")
        np.testing.assert_array_equal(out[0, 0], np.array([10, 20, 30]))

    def test_leading_separator_stripped(self):
        X = [[",7,8,9"]]
        out = self.formatter.format_as_integer(X)
        np.testing.assert_array_equal(out[0, 0], np.array([7, 8, 9]))

    def test_empty_after_split_filtered(self):
        X = [["1,2,"]]
        out = self.formatter.format_as_integer(X)
        np.testing.assert_array_equal(out[0, 0], np.array([1, 2]))

    def test_multiple_dimensions(self):
        X = [["10,20,30,40"]]
        out = self.formatter.format_as_integer(X, num_dims=2)
        self.assertEqual(out.shape, (1, 1, 2))
        np.testing.assert_array_equal(out[0, 0], np.array([10, 30]))

    def test_target_column_one(self):
        X = [["10,20,30,40"]]
        out = self.formatter.format_as_integer(X, num_dims=2, target_column=1)
        self.assertEqual(out.shape, (1, 1, 2))
        np.testing.assert_array_equal(out[0, 0], np.array([20, 40]))
