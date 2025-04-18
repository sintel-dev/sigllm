# -*- coding: utf-8 -*-

"""Data Management module.

This module contains functions that allow downloading demo data from Amazon S3,
as well as load and work with other data stored locally.
"""

import logging
import os

import pandas as pd
from orion.data import format_csv, load_csv

LOGGER = logging.getLogger(__name__)

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
BUCKET = 'sintel-sigllm'
S3_URL = 'https://{}.s3.amazonaws.com/{}'


def download_normal(name, data_path=DATA_PATH):
    """Load the CSV with the given name from S3.

    If the CSV has never been loaded before, it will be downloaded
    from the [sintel-sigllm bucket](https://sintel-sigllm.s3.amazonaws.com) or
    the S3 bucket specified following the `s3://{bucket}/path/to/the.csv` format,
    and then cached inside the `data` folder, within the `sigllm` package
    directory, and then returned.

    Otherwise, if it has been downloaded and cached before, it will be directly
    loaded from the `sigllm/data` folder without contacting S3.

    Args:
        name (str):
            Name of the CSV to load.
        data_path (str):
            Path to store data.

    Returns:
        pandas.DataFrame:
            A pandas.DataFrame is returned containing all the data.

    Raises:
        FileNotFoundError: If the normal file doesn't exist locally and can't
        be downloaded from S3.
    """
    try:
        url = None
        if name.startswith('s3://'):
            parts = name[5:].split('/', 1)
            bucket = parts[0]
            path = parts[1]
            url = S3_URL.format(bucket, path)
            filename = os.path.join(data_path, path.split('/')[-1])
        else:
            filename = os.path.join(data_path, name + '_normal.csv')
            data_path = os.path.join(data_path, os.path.dirname(name))

        if os.path.exists(filename):
            data = pd.read_csv(filename)
            return data

        url = url or S3_URL.format(BUCKET, '{}_normal.csv'.format(name))
        LOGGER.info('Downloading CSV %s from %s', name, url)

        try:
            data = pd.read_csv(url)
            os.makedirs(data_path, exist_ok=True)
            data.to_csv(filename, index=False)
            return data
        except Exception:
            error_msg = (
                f'Could not download or find normal file for {name}. '
                f'Please ensure the file exists at {filename} or can be '
                f'downloaded from {url}'
            )
            LOGGER.error(error_msg)
            raise FileNotFoundError(error_msg)

    except Exception as e:
        error_msg = f'Error processing normal file for {name}: {str(e)}'
        LOGGER.error(error_msg)
        raise FileNotFoundError(error_msg)


def load_normal(name, timestamp_column=None, value_column=None, start=None, end=None):
    """Load normal data from file or download if needed.

    Args:
        name (str):
            Name or path of the normal data.
        timestamp_column (str or int):
            Column index or name for timestamp.
        value_column (str or int):
            Column index or name for values.
        start (int or timestamp):
            Optional. If specified, this will be start of the sub-sequence.
        end (int or timestamp):
            Optional. If specified, this will be end of the sub-sequence.

    Returns:
        pandas.DataFrame:
            Loaded subsequence with `timestamp` and `value` columns.
    """
    if os.path.isfile(name):
        data = load_csv(name, timestamp_column, value_column)
    else:
        data = download_normal(name)

    data = format_csv(data)

    # handle start or end is specified
    if start or end:
        if any(data.index.isin([start, end])):
            data = data.iloc[start:end]
        else:
            mask = True
            if start is not None:
                mask &= data[timestamp_column] >= start
            if end is not None:
                mask &= data[timestamp_column] <= end
            data = data[mask]

    return data
