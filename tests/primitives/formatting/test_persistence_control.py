import unittest

import numpy as np

from sigllm.primitives.formatting.persistence_control import PersistenceControl


class PersistenceControlFormatAsStringTest(unittest.TestCase):
    """Tests for PersistenceControl.format_as_string."""

    def setUp(self):
        self.formatter = PersistenceControl()

    def test_single_window_single_row(self):
        X = np.array([[[10]]])
        out = self.formatter.format_as_string(X)
        self.assertEqual(out, ["10"])

    def test_single_window_multiple_rows(self):
        X = np.array([[[1], [2], [3]]])
        out = self.formatter.format_as_string(X)
        self.assertEqual(out, ["1,2,3"])

    def test_multiple_windows(self):
        # X: (2 windows, 2 rows each, 1 dim)
        X = np.array([[[1], [2]], [[3], [4]]])
        out = self.formatter.format_as_string(X)
        self.assertEqual(out, ["1,2", "3,4"])

    def test_custom_separator(self):
        X = np.array([[[1], [2], [3]]])
        out = self.formatter.format_as_string(X, separator=";")
        self.assertEqual(out, ["1;2;3"])

    def test_uses_first_dimension_only(self):
        X = np.array([[[100, 200, 300], [400, 500, 600]]])
        out = self.formatter.format_as_string(X)
        self.assertEqual(out, ["100,400"])


class PersistenceControlFormatAsIntegerTest(unittest.TestCase):
    """Tests for PersistenceControl.format_as_integer (last value only)."""

    def setUp(self):
        self.formatter = PersistenceControl()

    def test_single_entry_takes_last(self):
        X = ["1,2,3"]
        out = self.formatter.format_as_integer(X)
        self.assertEqual(out.shape, (1, 1, 1))
        np.testing.assert_array_equal(out[0, 0], np.array([3]))

    def test_multiple_entries(self):
        X = ["1,2,3", "4,5", "99"]
        out = self.formatter.format_as_integer(X)
        self.assertEqual(out.shape, (3, 1, 1))
        np.testing.assert_array_equal(out[0, 0], np.array([3]))
        np.testing.assert_array_equal(out[1, 0], np.array([5]))
        np.testing.assert_array_equal(out[2, 0], np.array([99]))

    def test_single_value(self):
        X = ["42"]
        out = self.formatter.format_as_integer(X)
        np.testing.assert_array_equal(out[0, 0], np.array([42]))

    def test_custom_separator(self):
        X = ["10;20;30"]
        out = self.formatter.format_as_integer(X, separator=";")
        np.testing.assert_array_equal(out[0, 0], np.array([30]))

    def test_leading_separator_stripped(self):
        X = [",7,8,9"]
        out = self.formatter.format_as_integer(X)
        np.testing.assert_array_equal(out[0, 0], np.array([9]))

    def test_empty_after_split_filtered(self):
        X = ["1,2,"]
        out = self.formatter.format_as_integer(X)
        np.testing.assert_array_equal(out[0, 0], np.array([2]))


class PersistenceControlTargetDimTest(unittest.TestCase):
    """Tests for target_column parameter in format_as_string."""

    def setUp(self):
        self.formatter = PersistenceControl()

    def test_target_column_zero_default(self):
        X = np.array([[[100, 200], [300, 400]]])
        out = self.formatter.format_as_string(X, target_column=0)
        self.assertEqual(out, ["100,300"])

    def test_target_column_one(self):
        X = np.array([[[100, 200], [300, 400]]])
        out = self.formatter.format_as_string(X, target_column=1)
        self.assertEqual(out, ["200,400"])

    def test_target_column_from_config(self):
        formatter = PersistenceControl(target_column=1)
        X = np.array([[[100, 200], [300, 400]]])
        out = formatter.format_as_string(X)
        self.assertEqual(out, ["200,400"])

    def test_round_trip_target_column_one(self):
        X = np.array([[[100, 200], [300, 400]]])
        strings = self.formatter.format_as_string(X, target_column=1)
        out = self.formatter.format_as_integer(strings)
        np.testing.assert_array_equal(out[0, 0], np.array([400]))
