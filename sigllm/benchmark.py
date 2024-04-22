#!/usr/bin/env python
# coding: utf-8

import ast
import logging
import os
import pickle
import warnings
from datetime import datetime
from functools import partial
from glob import glob

import numpy as np
import pandas as pd
from mlstars.custom.timeseries_preprocessing import rolling_window_sequences
from orion.benchmark import _load_signal, _parse_confusion_matrix
from orion.data import load_anomalies, load_signal
from orion.evaluation import CONTEXTUAL_METRICS as METRICS
from orion.evaluation import contextual_confusion_matrix
from orion.primitives.timeseries_anomalies import find_anomalies
from orion.primitives.timeseries_errors import regression_errors
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from sigllm.forecasting import Signal2String, HF, GPT

warnings.simplefilter('ignore')

LOGGER = logging.getLogger(__name__)

BUCKET = 'sintel-orion'
S3_URL = 'https://{}.s3.amazonaws.com/{}'

BENCHMARK_DATA = pd.read_csv(S3_URL.format(
    BUCKET, 'datasets.csv'), index_col=0, header=None).applymap(ast.literal_eval).to_dict()[1]

BENCHMARK_PATH = os.path.join(
    os.path.dirname(__file__),
    'benchmark'
)

os.makedirs(BENCHMARK_PATH, exist_ok=True)

DONE = glob(BENCHMARK_PATH + '*.csv')

BENCHMARK_DATA = {
    k: v[:2] for k, v in BENCHMARK_DATA.items()  # limit to 2 samples
}

MAX = 2

del METRICS['accuracy']
METRICS['confusion_matrix'] = contextual_confusion_matrix
metrics = {k: partial(fun, weighted=False) for k, fun in METRICS.items()}

# setup

# rolling windows settings
window_size = 140
step_size = 1

# signal to string converter settings
decimals = 2
seperator = ','
space = True
rescale = True
trunc = 1

# GPT model settings
model_name = 'gpt-3.5-turbo'
steps = 5
samples = 10
temp = 0.9

# initialization

scaler = MinMaxScaler(feature_range=(0, 1))
converter = Signal2String(decimal=decimals, sep=seperator, space=space, rescale=rescale)
model = GPT(model_name, converter.sep)


# make rolling windows

def run_experiment(signal_name):
    file_name = os.path.join(BENCHMARK_PATH, f'{model_name}_{signal}_{dataset}_{temp}_{samples}')

    data = load_signal(signal_name)
    truth = load_anomalies(signal_name)
    train, test = _load_signal(signal_name, test_split)

    timestamps, values = data.values.T

    start = datetime.utcnow()

    timeseries = scaler.fit_transform(values.reshape(-1, 1))

    X, y, X_index, y_index = rolling_window_sequences(
        timeseries, timestamps,
        target_column=0,
        target_size=1,
        window_size=window_size,
        step_size=step_size
    )

    outputs = []
    probs = []
    for window in tqdm(X):
        input_str = converter.transform(window.flatten())
        output, prob = model.forecast(
            input_str,
            steps=steps,
            samples=samples,
            temp=temp,
            logprobs=True,
            top_logprobs=3
        )

        outputs.append(output)
        probs.append(prob)

    with open(file_name + '_raw_forecast.pkl', 'wb') as f:
        pickle.dump(outputs, f)

    with open(file_name + '_raw_logprobs.pkl', 'wb') as f:
        pickle.dump(probs, f)

    predictions = []
    incomplete = 0
    error = 0

    for output in outputs:
        result = []
        for o in output:
            try:
                pred = converter.reverse_transform(o, trunc=trunc)
                if len(pred) != 1:
                    incomplete += 1
                    continue

            except Exception as ex:
                LOGGER.exception(
                    "Exception reverse transform on signal %s, error %s.",
                    signal_name,
                    ex)
                error += 1
                pred = [None]

            result.append(pred[:trunc])
        predictions.append(result)

    LOGGER.info(f"Recorded {incomplete} incomplete forecasts, and encountered {error} errors.")
    predictions = np.array(predictions, dtype=float)
    predictions = predictions[:, :, 0]
    predictions[predictions > MAX] = predictions.mean()
    pred_median = np.median(predictions.T, axis=0)

    errors = regression_errors(y, pred_median)
    events = find_anomalies(
        errors,
        y_index,
        window_size_portion=0.33,
        window_step_size_portion=0.1,
        fixed_threshold=True)

    anomalies = pd.DataFrame(list(events), columns=['start', 'end', 'score'])
    anomalies['start'] = anomalies['start'].astype('int64')
    anomalies['end'] = anomalies['end'].astype('int64')

    elapsed = datetime.utcnow() - start

    try:
        scores = {
            name: scorer(truth, anomalies, test)
            for name, scorer in metrics.items()
        }
        _parse_confusion_matrix(scores, truth)

        status = 'OK'

    except Exception as ex:
        LOGGER.exception("Exception scoring signal %s, error %s.", signal_name, ex)

        scores = {
            name: 0 for name in metrics.keys()
        }

        status = 'ERROR'

    scores['status'] = status
    scores['incomplete'] = incomplete
    scores['elapsed'] = elapsed.total_seconds()

    raw = {
        'predictions': predictions,
        'anomalies': anomalies,
        'errors': errors
    }

    record = pd.DataFrame.from_records([scores], columns=scores.keys())
    record.to_csv(file_name + '_scores.csv', index=False)

    with open(file_name + '_raw_predictions.pkl', 'wb') as f:
        pickle.dump(raw, f)


if __name__ == "__main__":
    for dataset, signals in BENCHMARK_DATA.items():
        for i, signal in enumerate(signals):
            file_name = os.path.join(BENCHMARK_PATH,
                                     f'{model_name}_{signal}_{dataset}_{temp}_{samples}')
            if file_name + '_scores.csv' in DONE:
                LOGGER.info(f'Skipping {signal}')
                continue

            test_split = False
            if dataset in ['MSL', 'SMAP']:
                test_split = True

            LOGGER.info(f'{i}/{len(signals)} Running experiment for {signal}')
            print(f'{i}/{len(signals)} Running experiment for {signal}')
            run_experiment(signal)
