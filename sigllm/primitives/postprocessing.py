# -*- coding: utf-8 -*-
import numpy as np


def outliers(predictions):
    Q1, Q3 = np.percentile(predictions, [25, 75])

    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    predictions[(predictions < lower_bound) | (predictions > upper_bound)] = np.nan

    return predictions


def aggregate_rolling_window(y, step_size=1, agg="median", remove_outliers=False):
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
        remove_outliers (bool):
            Indicator to whether remove outliers from the predictions.

    Return:
        ndarray:
            Flattened sequence.
    """
    num_windows, num_samples, pred_length = y.shape
    num_errors = pred_length + step_size * (num_windows - 1)

    if remove_outliers:
        y = outliers(y)

    method = getattr(np, agg)
    signal = []

    for i in range(num_errors):
        intermediate = []
        for j in range(max(0, i - num_errors + pred_length), min(i + 1, pred_length)):
            for k in range(num_samples):
                intermediate.append(y[i - j, k, j])

        signal.append(method(np.asarray(intermediate)))

    return np.array(signal)
