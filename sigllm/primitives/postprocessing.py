# -*- coding: utf-8 -*-
import numpy as np


def aggregate_rolling_window(y, step_size=1, agg="median"):
    """Aggregate a rolling window sequence.

    Convert a rolling window sequence into a flattened time series.
    Use the aggregation specified to make each timestamp a single value.

    Args:
        y (ndarray):
            Windowed sequences. Each timestamp has multiple predictions.
        step_size (int):
            Stride size used when creating the rolling windows.
        agg (string):
            String denoting the aggregation method to use. Default is "median".

    Return:
        ndarray:
            Flattened sequence.
    """
    num_windows, num_samples, pred_length = y.shape
    num_errors = pred_length + step_size * (num_windows - 1)

    method = getattr(np, agg)
    signal = []

    for i in range(num_errors):
        intermediate = []
        for j in range(max(0, i - num_errors + pred_length), min(i + 1, pred_length)):
            for k in range(num_samples):
                intermediate.append(y[i - j, k, j])

        signal.append(method(np.asarray(intermediate)))

    return np.array(signal)
