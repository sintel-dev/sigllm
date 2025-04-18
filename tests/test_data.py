#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `sigllm.data` module."""

from datetime import datetime
from unittest.mock import patch

import pandas as pd
import pytest

from sigllm.data import load_normal


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=10, freq='D'),
        'value': range(10),
    })


@patch('sigllm.data.download_normal')
@patch('sigllm.data.format_csv')
def test_load_normal_without_start_end(mock_format_csv, mock_download, sample_data):
    mock_format_csv.return_value = sample_data
    mock_download.return_value = sample_data

    result = load_normal('test.csv')
    mock_download.assert_called_once()
    pd.testing.assert_frame_equal(result, sample_data)


@patch('sigllm.data.download_normal')
@patch('sigllm.data.format_csv')
def test_load_normal_with_index_based_start_end(mock_format_csv, mock_download, sample_data):
    mock_format_csv.return_value = sample_data
    mock_download.return_value = sample_data

    result = load_normal('test.csv', start=2, end=5)
    expected = sample_data.iloc[2:5]
    pd.testing.assert_frame_equal(result, expected)

    result = load_normal('test.csv', start=2)
    expected = sample_data.iloc[2:]
    pd.testing.assert_frame_equal(result, expected)

    result = load_normal('test.csv', end=5)
    expected = sample_data.iloc[:5]
    pd.testing.assert_frame_equal(result, expected)


@patch('sigllm.data.download_normal')
@patch('sigllm.data.format_csv')
def test_load_normal_with_timestamp_based_start_end(mock_format_csv, mock_download, sample_data):
    mock_format_csv.return_value = sample_data
    mock_download.return_value = sample_data

    start_date = datetime(2023, 1, 3)
    end_date = datetime(2023, 1, 6)
    result = load_normal('test.csv', timestamp_column='timestamp', start=start_date, end=end_date)

    expected = sample_data[
        (sample_data['timestamp'] >= start_date) & (sample_data['timestamp'] <= end_date)
    ]
    pd.testing.assert_frame_equal(result, expected)

    result = load_normal('test.csv', timestamp_column='timestamp', start=start_date)
    expected = sample_data[sample_data['timestamp'] >= start_date]
    pd.testing.assert_frame_equal(result, expected)

    result = load_normal('test.csv', timestamp_column='timestamp', end=end_date)
    expected = sample_data[sample_data['timestamp'] <= end_date]
    pd.testing.assert_frame_equal(result, expected)
