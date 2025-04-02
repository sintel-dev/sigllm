# -*- coding: utf-8 -*-

import argparse
import ast
import concurrent
import json
import logging
import os
import uuid
import warnings
from copy import deepcopy
from datetime import datetime
from functools import partial
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm
from mlblocks import get_pipelines_paths
from orion.benchmark import _load_signal, _parse_confusion_matrix, _sort_leaderboard
from orion.data import load_anomalies
from orion.evaluation import CONTEXTUAL_METRICS as METRICS
from orion.evaluation import contextual_confusion_matrix
from orion.progress import TqdmLogger, progress

from sigllm import SigLLM
from sigllm.data import load_normal

warnings.simplefilter('ignore')

LOGGER = logging.getLogger(__name__)

BUCKET = 'sintel-sigllm'
S3_URL = 'https://{}.s3.amazonaws.com/{}'

BENCHMARK_PATH = os.path.join(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'), 'benchmark'
)

BENCHMARK_DATA = (
    pd.read_csv(S3_URL.format(BUCKET, 'datasets.csv'), index_col=0, header=None)
    .applymap(ast.literal_eval)
    .to_dict()[1]
)
BENCHMARK_PARAMS = (
    pd.read_csv(S3_URL.format(BUCKET, 'parameters.csv'), index_col=0, header=None)
    .applymap(ast.literal_eval)
    .to_dict()[1]
)

PIPELINE_DIR = os.path.join(os.path.dirname(__file__), 'pipelines')

PIPELINES = {
    'mistral_prompter_restricted': 'mistral_prompter',
    'mistral_prompter_0shot': 'mistral_prompter_0shot',
    'mistral_prompter_1shot': 'mistral_prompter_1shot',
}  # ['mistral_prompter_0shot', 'mistral_prompter_1shot']


def _get_pipeline_directory(pipeline_name):
    if os.path.isfile(pipeline_name):
        return os.path.dirname(pipeline_name)

    pipelines_paths = get_pipelines_paths()
    for base_path in pipelines_paths:
        parts = pipeline_name.split('.')
        number_of_parts = len(parts)

        for folder_parts in range(number_of_parts):
            folder = os.path.join(base_path, *parts[:folder_parts])
            filename = '.'.join(parts[folder_parts:]) + '.json'
            json_path = os.path.join(folder, filename)

            if os.path.isfile(json_path):
                return os.path.dirname(json_path)


def _get_pipeline_hyperparameter(hyperparameters, dataset_name, pipeline_name):
    hyperparameters_ = deepcopy(hyperparameters)

    if hyperparameters:
        hyperparameters_ = hyperparameters_.get(dataset_name) or hyperparameters_
        hyperparameters_ = hyperparameters_.get(pipeline_name) or hyperparameters_

    if hyperparameters_ is None and dataset_name and pipeline_name:
        pipeline_path = _get_pipeline_directory(pipeline_name)
        pipeline_dirname = os.path.basename(pipeline_path)
        file_path = os.path.join(
            pipeline_path, pipeline_dirname + '_' + dataset_name.lower() + '.json'
        )
        if os.path.exists(file_path):
            hyperparameters_ = file_path

    if isinstance(hyperparameters_, str) and os.path.exists(hyperparameters_):
        with open(hyperparameters_) as f:
            hyperparameters_ = json.load(f)

    return hyperparameters_


def _augment_hyperparameters(hyperparameters, few_shot):
    hyperparameters_ = deepcopy(hyperparameters)
    if few_shot:
        for hyperparameter, value in hyperparameters.items():
            if 'time_segments_aggregate' in hyperparameter:
                name = hyperparameter[:-1]
                hyperparameters_[name + '2'] = value

    return hyperparameters_


def _evaluate_signal(
    pipeline, signal, hyperparameter, metrics, test_split=False, few_shot=False, anomaly_path=None
):
    _, test = _load_signal(signal, test_split)
    truth = load_anomalies(signal)

    normal = None
    if few_shot:
        normal = load_normal(signal)

    try:
        LOGGER.info(
            'Scoring pipeline %s on signal %s (test split: %s)', pipeline, signal, test_split
        )

        start = datetime.utcnow()
        pipeline = SigLLM(pipeline, hyperparameter)
        anomalies = pipeline.detect(test, normal=normal)
        elapsed = datetime.utcnow() - start

        scores = {name: scorer(truth, anomalies, test) for name, scorer in metrics.items()}

        status = 'OK'

    except Exception as ex:
        LOGGER.exception(
            'Exception scoring pipeline %s on signal %s (test split: %s), error %s.',
            pipeline,
            signal,
            test_split,
            ex,
        )

        elapsed = datetime.utcnow() - start
        anomalies = pd.DataFrame([], columns=['start', 'end', 'score'])
        scores = {name: 0 for name in metrics.keys()}

        status = 'ERROR'

    if 'confusion_matrix' in metrics.keys():
        _parse_confusion_matrix(scores, truth)

    scores['status'] = status
    scores['elapsed'] = elapsed.total_seconds()
    scores['split'] = test_split

    if anomaly_path:
        anomalies.to_csv(anomaly_path, index=False)

    return scores


def _run_job(args):
    # Reset random seed
    np.random.seed()

    (
        pipeline,
        pipeline_name,
        dataset,
        signal,
        hyperparameter,
        metrics,
        test_split,
        few_shot,
        iteration,
        cache_dir,
        anomaly_dir,
        run_id,
    ) = args

    anomaly_path = anomaly_dir
    if anomaly_dir:
        base_path = str(anomaly_dir / f'{pipeline_name}_{signal}_{dataset}_{iteration}')
        anomaly_path = base_path + '_anomalies.csv'

    LOGGER.info(
        'Evaluating pipeline %s on signal %s dataset %s (test split: %s); iteration %s',
        pipeline_name,
        signal,
        dataset,
        test_split,
        iteration,
    )

    output = _evaluate_signal(
        pipeline, signal, hyperparameter, metrics, test_split, few_shot, anomaly_path
    )
    scores = pd.DataFrame.from_records([output], columns=output.keys())

    scores.insert(0, 'dataset', dataset)
    scores.insert(1, 'pipeline', pipeline_name)
    scores.insert(2, 'signal', signal)
    scores.insert(3, 'iteration', iteration)
    scores['run_id'] = run_id

    if cache_dir:
        base_path = str(cache_dir / f'{pipeline_name}_{signal}_{dataset}_{iteration}_{run_id}')
        scores.to_csv(base_path + '_scores.csv', index=False)

    return scores


def _run_on_dask(jobs, verbose):
    """Run the tasks in parallel using dask."""
    try:
        import dask
    except ImportError as ie:
        ie.msg += (
            '\n\nIt seems like `dask` is not installed.\n'
            'Please install `dask` and `distributed` using:\n'
            '\n    pip install dask distributed'
        )
        raise

    scorer = dask.delayed(_run_job)
    persisted = dask.persist(*[scorer(args) for args in jobs])
    if verbose:
        try:
            progress(persisted)
        except ValueError:
            pass

    return dask.compute(*persisted)


def benchmark(
    pipelines=None,
    datasets=None,
    hyperparameters=None,
    metrics=METRICS,
    rank='f1',
    test_split=False,
    iterations=1,
    workers=1,
    show_progress=False,
    cache_dir=None,
    anomaly_dir=None,
    resume=False,
    output_path=None,
):
    """Run pipelines on the given datasets and evaluate the performance.

    The pipelines are used to analyze the given signals and later on the
    detected anomalies are scored against the known anomalies using the
    indicated metrics.

    Finally, the scores obtained with each metric are averaged accross all the signals,
    ranked by the indicated metric and returned on a ``pandas.DataFrame``.

    Args:
        pipelines (dict or list): dictionary with pipeline names as keys and their
            JSON paths as values. If a list is given, it should be of JSON paths,
            and the paths themselves will be used as names. If not give, all verified
            pipelines will be used for evaluation.
        datasets (dict or list): dictionary of dataset name as keys and list of signals as
            values. If a list is given then it will be under a generic name ``dataset``.
            If not given, all benchmark datasets will be used used.
        hyperparameters (dict or list): dictionary with pipeline names as keys
            and their hyperparameter JSON paths or dictionaries as values. If a list is
            given, it should be of corresponding order to pipelines.
        metrics (dict or list): dictionary with metric names as keys and
            scoring functions as values. If a list is given, it should be of scoring
            functions, and they ``__name__`` value will be used as the metric name.
            If not given, all the available metrics will be used.
        rank (str): Sort and rank the pipelines based on the given metric.
            If not given, rank using the first metric.
        test_split (bool or float): Whether to use the prespecified train-test split. If
            float, then it should be between 0.0 and 1.0 and represent the proportion of
            the signal to include in the test split. If not given, use ``False``.
        iterations (int):
            Number of iterations to perform over each signal and pipeline. Defaults to 1.
        workers (int or str):
            If ``workers`` is given as an integer value other than 0 or 1, a multiprocessing
            Pool is used to distribute the computation across the indicated number of workers.
            If the string ``dask`` is given, the computation is distributed using ``dask``.
            In this case, setting up the ``dask`` cluster and client is expected to be handled
            outside of this function.
        show_progress (bool):
            Whether to use tqdm to keep track of the progress. Defaults to ``True``.
        cache_dir (str):
            If a ``cache_dir`` is given, intermediate results are stored in the indicated directory
            as CSV files as they get computted. This allows inspecting results while the benchmark
            is still running and also recovering results in case the process does not finish
            properly. Defaults to ``None``.
        anomaly_dir (str):
            If a ``anomaly_dir`` is given, detected anomalies will get dumped in the specificed
            directory as csv files. Defaults to ``None``.
        resume (bool):
            Whether to continue running the experiments in the benchmark from the current
            progress in ``cache_dir``.
        output_path (str): Location to save the intermediatry results. If not given,
            intermediatry results will not be saved.

    Returns:
        pandas.DataFrame:
            A table containing the scores obtained with each scoring function accross
            all the signals for each pipeline.
    """
    pipelines = pipelines or PIPELINES
    datasets = datasets or BENCHMARK_DATA
    run_id = os.getenv('RUN_ID') or str(uuid.uuid4())[:10]

    if isinstance(pipelines, list):
        pipelines = {pipeline: pipeline for pipeline in pipelines}

    if isinstance(datasets, list):
        datasets = {'dataset': datasets}

    if isinstance(hyperparameters, list):
        hyperparameters = {
            pipeline: hyperparameter
            for pipeline, hyperparameter in zip(pipelines.keys(), hyperparameters)
        }

    if isinstance(metrics, list):
        metrics_ = dict()
        for metric in metrics:
            if callable(metric):
                metrics_[metric.__name__] = metric
            elif metric in METRICS:
                metrics_[metric] = METRICS[metric]
            else:
                raise ValueError('Unknown metric: {}'.format(metric))

        metrics = metrics_

    if cache_dir:
        cache_dir = Path(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)

    if anomaly_dir:
        anomaly_dir = Path(anomaly_dir)
        os.makedirs(anomaly_dir, exist_ok=True)

    jobs = list()
    for dataset, signals in datasets.items():
        for pipeline_name, pipeline in pipelines.items():
            hyperparameter = _get_pipeline_hyperparameter(hyperparameters, dataset, pipeline_name)
            parameters = BENCHMARK_PARAMS.get(dataset)

            few_shot = True if '1shot' in pipeline_name.lower() else False
            hyperparameter = _augment_hyperparameters(hyperparameter, few_shot)

            if parameters is not None:
                test_split = parameters.values()
            for signal in signals:
                for iteration in range(iterations):
                    if resume:
                        experiment = str(
                            cache_dir / f'{pipeline_name}_{signal}_{dataset}_{iteration}'
                        )
                        if len(glob(experiment + '*.csv')) > 0:
                            LOGGER.warning(f'skipping {experiment}')
                            continue

                    args = (
                        pipeline,
                        pipeline_name,
                        dataset,
                        signal,
                        hyperparameter,
                        metrics,
                        test_split,
                        few_shot,
                        iteration,
                        cache_dir,
                        anomaly_dir,
                        run_id,
                    )
                    jobs.append(args)

    if workers == 'dask':
        scores = _run_on_dask(jobs, show_progress)
    else:
        if workers in (0, 1):
            scores = map(_run_job, jobs)
        else:
            pool = concurrent.futures.ProcessPoolExecutor(workers)
            scores = pool.map(_run_job, jobs)

        scores = tqdm.tqdm(scores, total=len(jobs), file=TqdmLogger())
        if show_progress:
            scores = tqdm.tqdm(scores, total=len(jobs))

    if scores:
        scores = pd.concat(scores)
        if output_path:
            LOGGER.info('Saving benchmark report to %s', output_path)
            scores.to_csv(output_path, index=False)

        return _sort_leaderboard(scores, rank, metrics)

    LOGGER.info('No scores to be recorded.')
    return pd.DataFrame()


def main(pipelines, datasets, resume, workers, output_path, cache_dir, anomaly_dir, **kwargs):
    """Main to call benchmark function."""
    # output path
    output_path = os.path.join(BENCHMARK_PATH, 'results', output_path)

    # metrics
    del METRICS['accuracy']
    METRICS['confusion_matrix'] = contextual_confusion_matrix
    metrics = {k: partial(fun, weighted=False) for k, fun in METRICS.items()}

    results = benchmark(
        pipelines=pipelines,
        datasets=datasets,
        metrics=metrics,
        output_path=output_path,
        workers=workers,
        resume=resume,
        cache_dir=cache_dir,
        anomaly_dir=anomaly_dir,
    )

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--pipelines', nargs='+', type=str, default=PIPELINES)
    parser.add_argument('-d', '--datasets', nargs='+', type=str, default=BENCHMARK_DATA)
    parser.add_argument('-r', '--resume', type=bool, default=False)
    parser.add_argument('-w', '--workers', default=1)

    parser.add_argument('-o', '--output_path', type=str, default='results.csv')
    parser.add_argument('-c', '--cache_dir', type=str, default='cache')
    parser.add_argument('-ad', '--anomaly_dir', type=str, default='anomaly_dir')

    config = parser.parse_args()

    if any([dataset in BENCHMARK_DATA.keys() for dataset in config.datasets]):
        config.datasets = dict((dataset, BENCHMARK_DATA[dataset]) for dataset in config.datasets)

    results = main(**vars(config))
