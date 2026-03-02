import unittest

import numpy as np

from sigllm.primitives.formatting.digit_interleave import DigitInterleave


class DigitInterleaveFormatAsStringTest(unittest.TestCase):
    """Tests for DigitInterleave.format_as_string."""

    def setUp(self):
        self.formatter = DigitInterleave()

    def test_single_window_single_timestamp_one_value(self):
        X = np.array([[[5]]])
        out = self.formatter.format_as_string(X)
        self.assertEqual(out, ["005,"])
        self.assertEqual(self.formatter.metadata["width_used"], 3)

    def test_single_window_single_timestamp_two_values(self):
        X = np.array([[[1, 23]]])
        out = self.formatter.format_as_string(X)
        self.assertEqual(out, ["000213,"])
        self.assertEqual(self.formatter.metadata["width_used"], 3)

    def test_single_window_multiple_timestamps(self):
        X = np.array([[[100, 2], [3, 4]]])
        out = self.formatter.format_as_string(X)
        self.assertEqual(out, ["100002,000034,"])
        self.assertEqual(self.formatter.metadata["width_used"], 3)

    def test_multiple_windows(self):
        X = np.array([[[1, 2]], [[3, 4]]])
        out = self.formatter.format_as_string(X)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0], "000012,")
        self.assertEqual(out[1], "000034,")

    def test_digits_per_timestamp_wider_than_values(self):
        X = np.array([[[7]]]) 
        out = self.formatter.format_as_string(X, digits_per_timestamp=3)
        self.assertEqual(out, ["007,"])
        self.assertEqual(self.formatter.metadata["width_used"], 3)

    def test_values_wider_than_digits_per_timestamp(self):
        X = np.array([[[1234, 500], [101,500]], [[30, 10], [32, 14]]])
        out = self.formatter.format_as_string(X, digits_per_timestamp=2)
        self.assertEqual(out, ["10253040,00150010,", "00003100,00003124,"])
        self.assertEqual(self.formatter.metadata["width_used"], 4)

    def test_custom_separator(self):
        X = np.array([[[1], [2]]])
        out = self.formatter.format_as_string(X, separator=";")
        self.assertEqual(out, ["001;002;"])

    def test_custom_digits_per_timestamp(self):
        X = np.array([[[1], [2]]])
        out = self.formatter.format_as_string(X, digits_per_timestamp=2)
        self.assertEqual(out, ["01,02,"])
        self.assertEqual(self.formatter.metadata["width_used"], 2)


class DigitInterleaveFormatAsIntegerTest(unittest.TestCase):
    """Tests for DigitInterleave.format_as_integer (requires width_used in metadata)."""

    def setUp(self):
        self.formatter = DigitInterleave()
        self.formatter.metadata["width_used"] = 3

    def test_single_timestamp_single_value(self):
        X = [["005,"]]
        out = self.formatter.format_as_integer(X)
        self.assertEqual(len(out), 1)
        self.assertEqual(len(out[0]), 1)
        np.testing.assert_array_equal(out[0][0], np.array([5]))

    def test_single_timestamp_two_values(self):
        X = [["000213,"]]
        out = self.formatter.format_as_integer(X)
        self.assertEqual(len(out), 1)
        self.assertEqual(len(out[0]), 1)
        np.testing.assert_array_equal(out[0][0], np.array([1]))

    def test_multiple_timestamps_in_one_sample(self):
        X = [["000012,000034,"]]
        out = self.formatter.format_as_integer(X)
        self.assertEqual(len(out), 1)
        self.assertEqual(len(out[0]), 2)
        np.testing.assert_array_equal(out[0][0], np.array([1]))
        np.testing.assert_array_equal(out[0][1], np.array([3]))

    def test_multiple_entries(self):
        X = [["005,"], ["012,"]]
        out = self.formatter.format_as_integer(X)
        self.assertEqual(len(out), 2)
        np.testing.assert_array_equal(out[0][0], np.array([5]))
        np.testing.assert_array_equal(out[1][0], np.array([12]))

    def test_trunc_limits_timestamps(self):
        X = [["000012,000034,000056,"]]
        out = self.formatter.format_as_integer(X, trunc=2)
        self.assertEqual(len(out[0]), 2)
        np.testing.assert_array_equal(out[0][0], np.array([1]))
        np.testing.assert_array_equal(out[0][1], np.array([3]))

    def test_trunc_limits_values_per_timestamp(self):
        X = [["000000123,"]]
        out = self.formatter.format_as_integer(X, trunc=2)
        self.assertEqual(len(out[0]), 1)
        np.testing.assert_array_equal(out[0][0], np.array([1]))

    def test_custom_separator(self):
        X = [["001;002;"]]
        out = self.formatter.format_as_integer(X, separator=";")
        np.testing.assert_array_equal(out[0][0], np.array([1]))
        np.testing.assert_array_equal(out[0][1], np.array([2]))

    def test_target_column_one(self):
        X = [["000012,000034,"]]
        out = self.formatter.format_as_integer(X, target_column=1)
        self.assertEqual(len(out), 1)
        self.assertEqual(len(out[0]), 2)
        np.testing.assert_array_equal(out[0][0], np.array([2]))
        np.testing.assert_array_equal(out[0][1], np.array([4]))

    def test_target_column_with_trunc(self):
        X = [["000012,000034,000056,"]]
        out = self.formatter.format_as_integer(X, target_column=1, trunc=2)
        self.assertEqual(len(out[0]), 2)
        np.testing.assert_array_equal(out[0][0], np.array([2]))
        np.testing.assert_array_equal(out[0][1], np.array([4]))

    def test_target_column_from_config(self):
        formatter = DigitInterleave(target_column=1)
        formatter.metadata["width_used"] = 3
        X = [["000012,000034,"]]
        out = formatter.format_as_integer(X)
        np.testing.assert_array_equal(out[0][0], np.array([2]))
        np.testing.assert_array_equal(out[0][1], np.array([4]))


class DigitInterleaveRoundTripTest(unittest.TestCase):
    """Round-trip: format_as_string then format_as_integer."""

    def setUp(self):
        self.formatter = DigitInterleave()

    def test_round_trip_single_window(self):
        X = np.array([[[1, 23], [45, 6]]])
        strings = self.formatter.format_as_string(X)
        X_in = [[s] for s in strings]
        out = self.formatter.format_as_integer(X_in)
        self.assertEqual(len(out), 1)
        self.assertEqual(len(out[0]), 2)
        np.testing.assert_array_equal(out[0][0], np.array([1]))
        np.testing.assert_array_equal(out[0][1], np.array([45]))

    def test_round_trip_multiple_windows(self):
        X = np.array([[[10, 20]], [[30, 40]]])
        strings = self.formatter.format_as_string(X)
        X_in = [[s] for s in strings]
        out = self.formatter.format_as_integer(X_in)
        self.assertEqual(len(out), 2)
        np.testing.assert_array_equal(out[0][0], np.array([10]))
        np.testing.assert_array_equal(out[1][0], np.array([30]))
