# -*- coding: utf-8 -*-
import numpy as np


def aggregate_rolling_window(y, agg="median"):
    """Aggregate a rolling window sequence.

    Convert a rolling window sequence into a flattened time series.
    Use the aggregation specified to make each timestamp a single value.

    Args:
        y (ndarray):
            Windowed sequences. Each timestamp has multiple predictions.
        agg (string):
            String denoting the aggregation method to use. Default is "median".

    Return:
        ndarray:
            Flattened sequence.
    """
    num_windows = y.shape[0]
    window_size = y.shape[1]

    method = getattr(np, agg)

    signal = [method(y[::-1, :].diagonal(i)) for i in range(-num_windows + 1, window_size)]
    return np.array(signal)
