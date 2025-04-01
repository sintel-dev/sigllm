# -*- coding: utf-8 -*-

"""
Data Management module.

This module contains functions that allow downloading demo data from Amazon S3,
as well as load and work with other data stored locally.

The demo data is a modified version of the NASA data found here:

https://s3-us-west-2.amazonaws.com/telemanom/data.zip
"""

import json
import logging
import os

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'data'
)
BUCKET = 'sintel-sigllm'
S3_URL = 'https://{}.s3.amazonaws.com/{}'


def download_normal(name, test_size=None, data_path=DATA_PATH):
    """Load the CSV with the given name from S3.

    If the CSV has never been loaded before, it will be downloaded
    from the [d3-ai-orion bucket](https://d3-ai-orion.s3.amazonaws.com) or
    the S3 bucket specified following the `s3://{bucket}/path/to/the.csv` format,
    and then cached inside the `data` folder, within the `orion` package
    directory, and then returned.

    Otherwise, if it has been downloaded and cached before, it will be directly
    loaded from the `orion/data` folder without contacting S3.

    If a `test_size` value is given, the data will be split in two parts
    without altering its order, making the second one proportionally as
    big as the given value.

    Args:
        name (str): Name of the CSV to load.
        test_size (float): Value between 0 and 1 indicating the proportional
            size of the test split. If 0 or None (default), the data is not split.

    Returns:
        If no test_size is given, a single pandas.DataFrame is returned containing all
        the data. If test_size is given, a tuple containing one pandas.DataFrame for
        the train split and another one for the test split is returned.

    Raises:
        FileNotFoundError: If the normal file doesn't exist locally and can't be downloaded from S3.
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
        print("Downloading CSV %s from %s", name, url)
        
        try:
            data = pd.read_csv(url)
            os.makedirs(data_path, exist_ok=True)
            data.to_csv(filename, index=False)
            return data
        except Exception as e:
            error_msg = f"Could not download or find normal file for {name}. "
            error_msg += f"Please ensure the file exists at {filename} or can be downloaded from {url}"
            LOGGER.error(error_msg)
            raise FileNotFoundError(error_msg)

    except Exception as e:
        error_msg = f"Error processing normal file for {name}: {str(e)}"
        LOGGER.error(error_msg)
        raise FileNotFoundError(error_msg)


def format_csv(df, timestamp_column=None, value_columns=None):
    timestamp_column_name = df.columns[timestamp_column] if timestamp_column else df.columns[0]
    value_column_names = df.columns[value_columns] if value_columns else df.columns[1:]

    data = dict()
    data['timestamp'] = df[timestamp_column_name].astype('int64').values
    for column in value_column_names:
        data[column] = df[column].astype(float).values

    return pd.DataFrame(data)


def load_csv(path, timestamp_column=None, value_column=None):
    header = None if timestamp_column is not None else 'infer'
    data = pd.read_csv(path, header=header)

    if timestamp_column is None:
        if value_column is not None:
            raise ValueError("If value_column is provided, timestamp_column must be as well")

        return data

    elif value_column is None:
        raise ValueError("If timestamp_column is provided, value_column must be as well")
    elif timestamp_column == value_column:
        raise ValueError("timestamp_column cannot be the same as value_column")

    return format_csv(data, timestamp_column, value_column)


def load_normal(normal, test_size=None, timestamp_column=None, value_column=None):
    if os.path.isfile(normal):
        data = load_csv(normal, timestamp_column, value_column)
    else:
        data = download_normal(normal)

    data = format_csv(data)

    if test_size is None:
        return data

    test_length = round(len(data) * test_size)
    train = data.iloc[:-test_length]
    test = data.iloc[-test_length:]

    return train, test
