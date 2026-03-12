import unittest

import numpy as np

from sigllm.primitives.formatting.value_interleave import ValueInterleave


class ValueInterleaveFormatAsStringTest(unittest.TestCase):
    """Tests for ValueInterleave.format_as_string."""

    def setUp(self):
        self.formatter = ValueInterleave()

    def test_single_window_single_timestamp_one_value_to_string(self):
        X = np.array([[[512]]])
        out = self.formatter.format_as_string(X)
        self.assertEqual(out, ['512,'])
        self.assertEqual(self.formatter.metadata['width_used'], 3)

    def test_single_window_single_timestamp_two_values_to_string(self):
        X = np.array([[[1, 23]]])
        out = self.formatter.format_as_string(X)
        self.assertEqual(out, ['001023,'])
        self.assertEqual(self.formatter.metadata['width_used'], 3)

    def test_single_window_multiple_timestamps_to_string(self):
        X = np.array([[[101, 22], [35, 4]]])
        out = self.formatter.format_as_string(X)
        self.assertEqual(out, ['101022,035004,'])
        self.assertEqual(self.formatter.metadata['width_used'], 3)

    def test_multiple_windows_to_string(self):
        X = np.array([[[144, 254]], [[321, 456]]])
        out = self.formatter.format_as_string(X)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0], '144254,')
        self.assertEqual(out[1], '321456,')

    def test_digits_per_timestamp_wider_than_values(self):
        X = np.array([[[7]]])
        out = self.formatter.format_as_string(X, digits_per_timestamp=3)
        self.assertEqual(out, ['007,'])
        self.assertEqual(self.formatter.metadata['width_used'], 3)

    def test_values_wider_than_digits_per_timestamp(self):
        X = np.array([[[1234, 500], [101, 500]], [[30, 10], [32, 14]]])
        out = self.formatter.format_as_string(X, digits_per_timestamp=2)
        self.assertEqual(out, ['12340500,01010500,', '00300010,00320014,'])
        self.assertEqual(self.formatter.metadata['width_used'], 4)

    def test_custom_separator(self):
        X = np.array([[[1], [2]]])
        out = self.formatter.format_as_string(X, separator=';')
        self.assertEqual(out, ['001;002;'])

    def test_custom_digits_per_timestamp(self):
        X = np.array([[[1, 11, 40], [21, 50, 10]]])
        out = self.formatter.format_as_string(X, digits_per_timestamp=2)
        self.assertEqual(out, ['011140,215010,'])
        self.assertEqual(self.formatter.metadata['width_used'], 2)

    def test_multiple_windows_multiple_timestamps_to_string(self):
        X = np.array([[[144, 254], [104, 200]], [[321, 456], [101, 202]]])
        out = self.formatter.format_as_string(X)
        self.assertEqual(out, ['144254,104200,', '321456,101202,'])


class ValueInterleaveFormatAsIntegerTest(unittest.TestCase):
    """Tests for ValueInterleave.format_as_integer (requires width_used in metadata)."""

    def setUp(self):
        self.formatter = ValueInterleave()
        self.formatter.metadata['width_used'] = 3

    def test_single_timestamp_single_value_to_integer(self):
        X = [['005,']]
        out = self.formatter.format_as_integer(X)
        self.assertEqual(len(out), 1)
        self.assertEqual(len(out[0]), 1)
        np.testing.assert_array_equal(out[0][0], np.array([5]))

    def test_single_timestamp_two_values_to_integer(self):
        X = [['001023,']]
        out = self.formatter.format_as_integer(X)
        self.assertEqual(len(out), 1)
        self.assertEqual(len(out[0]), 1)
        np.testing.assert_array_equal(out[0][0], np.array([1]))

    def test_multiple_timestamps_in_one_sample_to_integer(self):
        X = [['001002,003004,']]
        out = self.formatter.format_as_integer(X)
        self.assertEqual(len(out), 1)
        self.assertEqual(len(out[0]), 1)
        np.testing.assert_array_equal(out[0][0], np.array([1, 3]))

    def test_multiple_entries_to_integer(self):
        X = [['005,'], ['012,']]
        out = self.formatter.format_as_integer(X)
        self.assertEqual(len(out), 2)
        np.testing.assert_array_equal(out[0][0], np.array([5]))
        np.testing.assert_array_equal(out[1][0], np.array([12]))

    def test_trunc_limits_timestamps(self):
        X = [['001002,003004,005006,']]
        out = self.formatter.format_as_integer(X, trunc=2)
        self.assertEqual(out.shape, (1, 1, 2))
        np.testing.assert_array_equal(out[0][0], np.array([1, 3]))

    def test_trunc_limits_values_per_timestamp(self):
        X = [['001002,003004,005006,']]
        out = self.formatter.format_as_integer(X, trunc=2)
        self.assertEqual(out.shape, (1, 1, 2))
        np.testing.assert_array_equal(out[0][0], np.array([1, 3]))

    def test_custom_separator(self):
        X = [['001;002;']]
        out = self.formatter.format_as_integer(X, separator=';')
        np.testing.assert_array_equal(out[0][0], np.array([1, 2]))

    def test_target_column_one(self):
        X = [['001023,045006,']]
        out = self.formatter.format_as_integer(X, target_column=1)
        self.assertEqual(len(out), 1)
        self.assertEqual(len(out[0]), 1)
        np.testing.assert_array_equal(out[0][0], np.array([23, 6]))

    def test_target_column_with_trunc(self):
        X = [['001023,045006,007008,']]
        out = self.formatter.format_as_integer(X, target_column=1, trunc=2)
        np.testing.assert_array_equal(out[0][0], np.array([23, 6]))

    def test_target_column_from_config(self):
        formatter = ValueInterleave(target_column=1)
        formatter.metadata['width_used'] = 3
        X = [['001023,045006,']]
        out = formatter.format_as_integer(X)
        np.testing.assert_array_equal(out[0][0], np.array([23, 6]))


class ValueInterleaveRoundTripTest(unittest.TestCase):
    """Round-trip: format_as_string then format_as_integer."""

    def setUp(self):
        self.formatter = ValueInterleave()

    def test_round_trip_single_window(self):
        X = np.array([[[1, 23], [45, 6]]])
        strings = self.formatter.format_as_string(X)
        X_in = [[s] for s in strings]
        out = self.formatter.format_as_integer(X_in)
        self.assertEqual(len(out), 1)
        self.assertEqual(len(out[0]), 1)
        np.testing.assert_array_equal(out[0][0], np.array([1, 45]))

    def test_round_trip_multiple_windows(self):
        X = np.array([[[10, 20]], [[30, 40]]])
        strings = self.formatter.format_as_string(X)
        X_in = [[s] for s in strings]
        out = self.formatter.format_as_integer(X_in)
        self.assertEqual(len(out), 2)
        np.testing.assert_array_equal(out[0][0], np.array([10]))
        np.testing.assert_array_equal(out[1][0], np.array([30]))
