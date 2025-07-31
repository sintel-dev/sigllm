import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import Mock, call, mock_open, patch

import pandas as pd
from orion.evaluation import CONTEXTUAL_METRICS as METRICS
from orion.evaluation import contextual_confusion_matrix
from pytest import fixture

from sigllm.benchmark import (
    _augment_hyperparameters,
    _evaluate_signal,
    _get_pipeline_directory,
    _get_pipeline_hyperparameter,
    _run_job,
    benchmark,
)


@fixture
def base_hyperparameters():
    return {
        'time_segments_aggregate#1': {'interval': 1},
        'rolling_window#1': {'window_size': 100},
        'other_param': 'other_value',
    }


def test__augment_hyperparameters_without_few_shot(base_hyperparameters):
    result = _augment_hyperparameters(base_hyperparameters, few_shot=False)

    assert result == base_hyperparameters


def test__augment_hyperparameters_with_few_shot(base_hyperparameters):
    result = _augment_hyperparameters(base_hyperparameters, few_shot=True)

    assert len(result) == 4
    assert result['time_segments_aggregate#1'] == base_hyperparameters['time_segments_aggregate#1']
    assert result['time_segments_aggregate#2'] == base_hyperparameters['time_segments_aggregate#1']
    assert result['rolling_window#1'] == base_hyperparameters['rolling_window#1']
    assert result['other_param'] == base_hyperparameters['other_param']


def test__augment_hyperparameters_with_empty_hyperparameters(base_hyperparameters):
    result = _augment_hyperparameters({}, few_shot=True)

    assert result == {}


def test__augment_hyperparameters_with_multiple_aggregate_params():
    hyperparameters = {
        'time_segments_aggregate#1': 'value1',
        'time_segments_aggregate#3': 'value3',
        'other_param': 'other_value',
    }

    result = _augment_hyperparameters(hyperparameters, few_shot=True)

    assert result['time_segments_aggregate#1'] == 'value1'
    assert result['time_segments_aggregate#2'] == 'value1'
    assert result['time_segments_aggregate#3'] == 'value3'
    assert result['time_segments_aggregate#4'] == 'value3'
    assert result['other_param'] == 'other_value'


@patch('os.path.isfile')
def test__get_pipeline_directory(mock_isfile):
    pipeline_path = '/fake/path/pipeline_dir/pipeline.json'
    mock_isfile.return_value = True

    expected = '/fake/path/pipeline_dir'
    result = _get_pipeline_directory(pipeline_path)

    assert result == expected


@patch('sigllm.benchmark.get_pipelines_paths')
@patch('os.path.isfile')
def test__get_pipeline_directory_search_pipeline_paths(mock_isfile, mock_pipeline_paths):
    pipeline_name = 'pipeline'
    base_path = '/fake/path/pipeline/'
    mock_pipeline_paths.return_value = [base_path]

    # first isfile check should be False to enter the pipeline paths loop
    # second isfile check should be True when finding the json file
    mock_isfile.side_effect = [False, True]

    expected = '/fake/path/pipeline'
    result = _get_pipeline_directory(pipeline_name)

    assert result == expected

    mock_isfile.assert_has_calls([
        call(pipeline_name),
        call(os.path.join(base_path, 'pipeline.json')),
    ])
    mock_pipeline_paths.assert_called_once()


class TestGetPipelineHyperparameter(TestCase):
    @classmethod
    def setup_class(cls):
        cls.pipeline_name = 'test_pipeline'
        cls.dataset_name = 'test_dataset'

        cls.temp_dir = tempfile.mkdtemp()

        cls.base_hyperparameters = {'default_param': 'default_value'}

        cls.dataset_hyperparameters = {'test_dataset': {'dataset_param': 'dataset_value'}}

        cls.pipeline_hyperparameters = {
            'test_dataset': {'test_pipeline': {'pipeline_param': 'pipeline_value'}}
        }

    def test__get_pipeline_hyperparameter_exist_direct(self):
        """Test when hyperparameters are directly provided"""
        hyperparameters = {'param1': 'value1', 'param2': 'value2'}

        result = _get_pipeline_hyperparameter(
            hyperparameters, self.dataset_name, self.pipeline_name
        )

        self.assertEqual(result, hyperparameters)

    def test__get_pipeline_hyperparameter_exist_nested(self):
        """Test when hyperparameters are nested by dataset and pipeline"""
        hyperparameters = {
            'test_dataset': {'test_pipeline': {'param1': 'value1', 'param2': 'value2'}}
        }

        result = _get_pipeline_hyperparameter(
            hyperparameters, self.dataset_name, self.pipeline_name
        )

        expected = {'param1': 'value1', 'param2': 'value2'}
        self.assertEqual(result, expected)

    def test__get_pipeline_hyperparameter_do_not_exist(self):
        """Test when no hyperparameters are provided"""
        result = _get_pipeline_hyperparameter(None, self.dataset_name, self.pipeline_name)

        self.assertIsNone(result)

    def test__get_pipeline_hyperparameter_dataset_dependent_hyperparameters(self):
        """Test when hyperparameters are dataset-specific"""
        hyperparameters = {
            'test_dataset': {'param1': 'dataset_value1', 'param2': 'dataset_value2'},
            'other_dataset': {'param1': 'other_value1', 'param2': 'other_value2'},
        }

        result = _get_pipeline_hyperparameter(
            hyperparameters, self.dataset_name, self.pipeline_name
        )

        expected = {'param1': 'dataset_value1', 'param2': 'dataset_value2'}
        self.assertEqual(result, expected)

        result = _get_pipeline_hyperparameter(
            hyperparameters, 'nonexistent_dataset', self.pipeline_name
        )

        self.assertEqual(result, hyperparameters)

    @patch('os.path.exists')
    @patch('json.load')
    def test__get_pipeline_hyperparameter_from_json_file(self, mock_json_load, mock_exists):
        """Test loading hyperparameters from a JSON file"""
        mock_exists.return_value = True
        mock_json_data = {'param1': 'file_value1', 'param2': 'file_value2'}
        mock_json_load.return_value = mock_json_data

        file_path = os.path.join(self.temp_dir, 'test_pipeline_test_dataset.json')

        with patch('builtins.open', mock_open(read_data=json.dumps(mock_json_data))):
            result = _get_pipeline_hyperparameter(file_path, self.dataset_name, self.pipeline_name)

        self.assertEqual(result, mock_json_data)

    @patch('sigllm.benchmark._get_pipeline_directory')
    @patch('os.path.exists')
    @patch('json.load')
    def test__get_pipeline_hyperparameter_none_with_pipeline_path(
        self, mock_json_load, mock_exists, mock_get_pipeline_dir
    ):
        """Test when hyperparameters is None but dataset and pipeline are specified.
        Should look for JSON file in pipeline directory."""
        pipeline_dir = '/fake/path/pipeline_dir'
        mock_get_pipeline_dir.return_value = pipeline_dir
        mock_exists.return_value = True

        expected_hyperparams = {'param1': 'auto_value1', 'param2': 'auto_value2'}
        mock_json_load.return_value = expected_hyperparams

        expected_file_path = os.path.join(
            pipeline_dir, f'{os.path.basename(pipeline_dir)}_{self.dataset_name.lower()}.json'
        )

        with patch('builtins.open', mock_open(read_data=json.dumps(expected_hyperparams))):
            result = _get_pipeline_hyperparameter(None, self.dataset_name, self.pipeline_name)

        mock_get_pipeline_dir.assert_called_once_with(self.pipeline_name)
        mock_exists.assert_called_once_with(expected_file_path)

        self.assertEqual(result, expected_hyperparams)

        mock_exists.return_value = False
        result = _get_pipeline_hyperparameter(None, self.dataset_name, self.pipeline_name)
        self.assertIsNone(result)

    def test__get_pipeline_hyperparameter_fallback_behavior(self):
        """Test the fallback behavior when dataset/pipeline specific params don't exist"""
        hyperparameters = {
            'default': {'param': 'default_value'},
            'test_dataset': {'param': 'dataset_value'},
            'test_pipeline': {'param': 'pipeline_value'},
        }

        # fallback to dataset level
        result = _get_pipeline_hyperparameter(
            hyperparameters, self.dataset_name, 'nonexistent_pipeline'
        )

        self.assertEqual(result['param'], 'dataset_value')

        # fallback to pipeline level
        result = _get_pipeline_hyperparameter(
            hyperparameters,
            'nonexistent_dataset',
            self.pipeline_name,
        )

        self.assertEqual(result['param'], 'pipeline_value')

        # fallback to default when dataset doesn't exist
        result = _get_pipeline_hyperparameter(
            hyperparameters, 'nonexistent_dataset', 'nonexistent_pipeline'
        )

        self.assertEqual(result, hyperparameters)


class TestEvaluateSignal(TestCase):
    @classmethod
    def setup_class(cls):
        cls.pipeline_name = 'test_pipeline'
        cls.signal_name = 'test_signal'
        cls.hyperparameters = {'param': 'value'}
        cls.metrics = {'f1': METRICS['f1']}
        cls.test_split = False
        cls.few_shot = False
        cls.anomaly_path = None

        cls.test_data = pd.DataFrame({'timestamp': [1, 2, 3], 'value': [1, 2, 3]})
        cls.truth_data = pd.DataFrame({'start': [1], 'end': [2]})
        cls.detected_anomalies = pd.DataFrame({'start': [1], 'end': [2], 'score': [0.9]})

    @patch('sigllm.benchmark.load_anomalies')
    @patch('sigllm.benchmark._load_signal')
    @patch('sigllm.benchmark.SigLLM')
    def test__evaluate_signal_success(self, mock_sigllm, mock_load_signal, mock_load_anomalies):
        mock_load_signal.return_value = (None, self.test_data)
        mock_load_anomalies.return_value = self.truth_data

        mock_pipeline = Mock()
        mock_pipeline.detect.return_value = self.detected_anomalies
        mock_sigllm.return_value = mock_pipeline

        result = _evaluate_signal(
            self.pipeline_name,
            self.signal_name,
            self.hyperparameters,
            self.metrics,
            self.test_split,
            self.few_shot,
            self.anomaly_path,
        )

        assert isinstance(result, dict)
        self.assertEqual(result['status'], 'OK')
        self.assertIn('elapsed', result)
        self.assertIn('split', result)
        self.assertIn('f1', result)

        mock_load_signal.assert_called_once_with(self.signal_name, self.test_split)
        mock_load_anomalies.assert_called_once_with(self.signal_name)
        mock_sigllm.assert_called_once_with(
            self.pipeline_name, hyperparameters=self.hyperparameters
        )
        mock_pipeline.detect.assert_called_once_with(self.test_data, normal=None)

    @patch('sigllm.benchmark.load_anomalies')
    @patch('sigllm.benchmark._load_signal')
    @patch('sigllm.benchmark.SigLLM')
    def test__evaluate_signal_fail(self, mock_sigllm, mock_load_signal, mock_load_anomalies):
        mock_load_signal.return_value = (None, self.test_data)
        mock_load_anomalies.return_value = self.truth_data
        mock_sigllm.side_effect = Exception('Test error')

        result = _evaluate_signal(
            self.pipeline_name,
            self.signal_name,
            self.hyperparameters,
            self.metrics,
            self.test_split,
            self.few_shot,
            self.anomaly_path,
        )

        assert isinstance(result, dict)
        self.assertEqual(result['status'], 'ERROR')
        self.assertIn('elapsed', result)
        self.assertIn('split', result)
        self.assertIn('f1', result)

        mock_load_signal.assert_called_once_with(self.signal_name, self.test_split)
        mock_load_anomalies.assert_called_once_with(self.signal_name)
        mock_sigllm.assert_called_once_with(
            self.pipeline_name, hyperparameters=self.hyperparameters
        )

    @patch('sigllm.benchmark.load_normal')
    @patch('sigllm.benchmark.load_anomalies')
    @patch('sigllm.benchmark._load_signal')
    @patch('sigllm.benchmark.SigLLM')
    def test__evaluate_signal_with_few_shot(
        self,
        mock_sigllm,
        mock_load_signal,
        mock_load_anomalies,
        mock_load_normal,
    ):
        mock_load_normal.return_value = self.test_data
        mock_load_signal.return_value = (None, self.test_data)
        mock_load_anomalies.return_value = self.truth_data

        mock_pipeline = Mock()
        mock_pipeline.detect.return_value = self.detected_anomalies
        mock_sigllm.return_value = mock_pipeline

        result = _evaluate_signal(
            self.pipeline_name,
            self.signal_name,
            self.hyperparameters,
            self.metrics,
            self.test_split,
            few_shot=True,
            anomaly_path=self.anomaly_path,
        )

        assert isinstance(result, dict)
        self.assertEqual(result['status'], 'OK')

        mock_pipeline.detect.assert_called_once_with(self.test_data, normal=self.test_data)
        mock_load_normal.assert_called_once_with(self.signal_name)
        mock_load_signal.assert_called_once_with(self.signal_name, self.test_split)
        mock_load_anomalies.assert_called_once_with(self.signal_name)
        mock_sigllm.assert_called_once_with(
            self.pipeline_name, hyperparameters=self.hyperparameters
        )

    @patch('sigllm.benchmark.load_anomalies')
    @patch('sigllm.benchmark._load_signal')
    @patch('sigllm.benchmark.SigLLM')
    def test__evaluate_signal_with_anomaly_path(
        self, mock_sigllm, mock_load_signal, mock_load_anomalies
    ):
        anomaly_path = 'test_anomalies.csv'
        mock_load_signal.return_value = (None, self.test_data)
        mock_load_anomalies.return_value = self.truth_data

        mock_pipeline = Mock()
        mock_pipeline.detect.return_value = self.detected_anomalies
        mock_sigllm.return_value = mock_pipeline

        with patch.object(pd.DataFrame, 'to_csv') as mock_to_csv:
            _evaluate_signal(
                self.pipeline_name,
                self.signal_name,
                self.hyperparameters,
                self.metrics,
                self.test_split,
                self.few_shot,
                anomaly_path,
            )

            mock_to_csv.assert_called_once_with(anomaly_path, index=False)

    @patch('sigllm.benchmark.load_anomalies')
    @patch('sigllm.benchmark._load_signal')
    @patch('sigllm.benchmark.SigLLM')
    def test_evaluate_signal_error(self, mock_sigllm, mock_load_signal, mock_load_anomalies):
        mock_load_signal.side_effect = Exception('Test error')

        with self.assertRaises(Exception):
            result = _evaluate_signal(
                self.pipeline_name,
                self.signal_name,
                self.hyperparameters,
                self.metrics,
                self.test_split,
                self.few_shot,
                self.anomaly_path,
            )

            assert isinstance(result, dict)
            self.assertEqual(result['status'], 'ERROR')
            self.assertIn('elapsed', result)
            self.assertEqual(result['f1'], 0)

            mock_load_signal.assert_called_once_with(self.signal_name, self.test_split)
            mock_load_anomalies.assert_not_called()
            mock_sigllm.assert_not_called()

    @patch('sigllm.benchmark.load_anomalies')
    @patch('sigllm.benchmark._load_signal')
    @patch('sigllm.benchmark.SigLLM')
    def test__evaluate_signal_confusion_matrix(
        self, mock_sigllm, mock_load_signal, mock_load_anomalies
    ):
        mock_load_signal.return_value = (None, self.test_data)
        mock_load_anomalies.return_value = self.truth_data

        mock_pipeline = Mock()
        mock_pipeline.detect.return_value = self.detected_anomalies
        mock_sigllm.return_value = mock_pipeline

        scores = (0, 0, 0, 1)
        metrics_with_cm = {
            'f1': METRICS['f1'],
            'confusion_matrix': Mock(autospec=contextual_confusion_matrix, return_value=scores),
        }

        result = _evaluate_signal(
            self.pipeline_name,
            self.signal_name,
            self.hyperparameters,
            metrics_with_cm,
            self.test_split,
            self.few_shot,
            self.anomaly_path,
        )

        assert 'tp' in result
        assert 'fp' in result
        assert 'fn' in result
        assert 'tn' in result


class TestRunJob(TestCase):
    @classmethod
    def setup_class(cls):
        cls.pipeline = 'test_pipeline'
        cls.pipeline_name = 'test_pipeline'
        cls.dataset = 'test_dataset'
        cls.signal = 'test_signal'
        cls.hyperparameters = {'param': 'value'}
        cls.metrics = {'f1': METRICS['f1']}
        cls.test_split = False
        cls.few_shot = False
        cls.iteration = 0
        cls.run_id = 'test_run'
        cls.anomaly_path = None

        cls.temp_dir = tempfile.mkdtemp()
        cls.cache_dir = Path(cls.temp_dir) / 'cache'
        cls.anomaly_dir = Path(cls.temp_dir) / 'anomalies'

        os.makedirs(cls.cache_dir, exist_ok=True)

    @classmethod
    def teardown_class(cls):
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    @patch('sigllm.benchmark._evaluate_signal')
    def test_run_job_basic(self, mock_evaluate):
        mock_output = {'f1': 0.8, 'status': 'OK', 'elapsed': 1.0, 'split': False}
        mock_evaluate.return_value = mock_output

        args = (
            self.pipeline,
            self.pipeline_name,
            self.dataset,
            self.signal,
            self.hyperparameters,
            self.metrics,
            self.test_split,
            self.few_shot,
            self.iteration,
            None,  # cache_dir
            None,  # anomaly_dir
            self.run_id,
        )

        result = _run_job(args)

        self.assertIsInstance(result, pd.DataFrame)

        row = result.iloc[0]
        self.assertEqual(row['pipeline'], self.pipeline_name)
        self.assertEqual(row['dataset'], self.dataset)
        self.assertEqual(row['signal'], self.signal)
        self.assertEqual(row['iteration'], self.iteration)
        self.assertEqual(row['run_id'], self.run_id)

        mock_evaluate.assert_called_once_with(
            self.pipeline,
            self.signal,
            self.hyperparameters,
            self.metrics,
            self.test_split,
            self.few_shot,
            None,  # anomaly_path
        )

    @patch('sigllm.benchmark._evaluate_signal')
    def test_run_job_with_cache(self, mock_evaluate):
        mock_output = {'f1': 0.8, 'status': 'OK', 'elapsed': 1.0, 'split': False}
        mock_evaluate.return_value = mock_output

        args = (
            self.pipeline,
            self.pipeline_name,
            self.dataset,
            self.signal,
            self.hyperparameters,
            self.metrics,
            self.test_split,
            self.few_shot,
            self.iteration,
            self.cache_dir,
            None,  # anomaly_dir
            self.run_id,
        )

        _run_job(args)

        expected_cache_file = (
            self.cache_dir / f'{self.pipeline_name}'
            f'_{self.signal}_{self.dataset}_{self.iteration}_{self.run_id}_scores.csv'
        )

        self.assertTrue(expected_cache_file.exists())

    @patch('sigllm.benchmark._evaluate_signal')
    def test_run_job_with_anomaly_dir(self, mock_evaluate):
        mock_output = {'f1': 0.8, 'status': 'OK', 'elapsed': 1.0, 'split': False}
        mock_evaluate.return_value = mock_output

        args = (
            self.pipeline,
            self.pipeline_name,
            self.dataset,
            self.signal,
            self.hyperparameters,
            self.metrics,
            self.test_split,
            self.few_shot,
            self.iteration,
            None,  # cache_dir
            self.anomaly_dir,
            self.run_id,
        )

        _run_job(args)

        expected_anomaly_path = str(
            self.anomaly_dir / f'{self.pipeline_name}'
            f'_{self.signal}_{self.dataset}_{self.iteration}_anomalies.csv'
        )

        mock_evaluate.assert_called_once_with(
            self.pipeline,
            self.signal,
            self.hyperparameters,
            self.metrics,
            self.test_split,
            self.few_shot,
            expected_anomaly_path,
        )


class TestBenchmark(TestCase):
    @classmethod
    def setup_class(cls):
        cls.pipeline_name = 'test_pipeline'
        cls.dataset_name = 'test_dataset'
        cls.signal_name = 'test_signal'
        cls.run_id = 'test_run'

        cls.pipelines = {cls.pipeline_name: 'pipeline_path'}
        cls.datasets = {cls.dataset_name: [cls.signal_name]}
        cls.hyperparameters = {'param': 'value'}
        cls.metrics = {'f1': METRICS['f1']}

        cls.temp_dir = tempfile.mkdtemp()
        cls.cache_dir = Path(cls.temp_dir) / 'cache'
        cls.anomaly_dir = Path(cls.temp_dir) / 'anomalies'

        cls.expected_columns = [
            'pipeline',
            'rank',
            'dataset',
            'signal',
            'iteration',
            'f1',
            'status',
            'elapsed',
            'split',
            'run_id',
        ]

    @classmethod
    def teardown_class(cls):
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    @patch('sigllm.benchmark._run_job')
    def test_benchmark_basic(self, mock_run_job):
        mock_output = pd.DataFrame({
            'pipeline': [self.pipeline_name],
            'dataset': [self.dataset_name],
            'signal': [self.signal_name],
            'iteration': [0],
            'f1': [0.8],
            'status': ['OK'],
            'elapsed': [1.0],
            'split': [False],
            'run_id': [self.run_id],
        })
        mock_run_job.return_value = mock_output

        result = benchmark(
            pipelines=self.pipelines,
            datasets=self.datasets,
            hyperparameters=self.hyperparameters,
            metrics=self.metrics,
        )

        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(all(col in result.columns for col in self.expected_columns))
        mock_run_job.assert_called_once()

    @patch('sigllm.benchmark._run_job')
    def test_benchmark_with_list_inputs(self, mock_run_job):
        mock_output = pd.DataFrame({
            'pipeline': [self.pipeline_name],
            'dataset': ['dataset'],
            'signal': [self.signal_name],
            'iteration': [0],
            'f1': [0.8],
            'status': ['OK'],
            'elapsed': [1.0],
            'split': [False],
            'run_id': [self.run_id],
        })
        mock_run_job.return_value = mock_output

        result = benchmark(
            pipelines=[self.pipeline_name],
            datasets=[self.signal_name],
            hyperparameters=[self.hyperparameters],
            metrics=list(self.metrics.values()),
        )

        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(all(col in result.columns for col in self.expected_columns))

    def test_benchmark_with_metric_error(self):
        with self.assertRaises(ValueError):
            benchmark(
                pipelines=self.pipelines,
                datasets=self.datasets,
                hyperparameters=self.hyperparameters,
                metrics=['f1', 'nonexistent_metric'],
            )

    @patch('sigllm.benchmark._run_job')
    def test_benchmark_with_cache(self, mock_run_job):
        os.makedirs(self.cache_dir, exist_ok=True)

        mock_output = pd.DataFrame({
            'pipeline': [self.pipeline_name],
            'dataset': [self.dataset_name],
            'signal': [self.signal_name],
            'iteration': [0],
            'f1': [0.8],
            'status': ['OK'],
            'elapsed': [1.0],
            'split': [False],
            'run_id': [self.run_id],
        })
        mock_run_job.return_value = mock_output

        result = benchmark(
            pipelines=self.pipelines, datasets=self.datasets, cache_dir=self.cache_dir
        )

        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue((self.cache_dir).exists())

    @patch('sigllm.benchmark._run_job')
    def test_benchmark_with_anomaly_dir(self, mock_run_job):
        os.makedirs(self.anomaly_dir, exist_ok=True)

        mock_output = pd.DataFrame({
            'pipeline': [self.pipeline_name],
            'dataset': [self.dataset_name],
            'signal': [self.signal_name],
            'iteration': [0],
            'f1': [0.8],
            'status': ['OK'],
            'elapsed': [1.0],
            'split': [False],
            'run_id': [self.run_id],
        })
        mock_run_job.return_value = mock_output

        result = benchmark(
            pipelines=self.pipelines, datasets=self.datasets, anomaly_dir=self.anomaly_dir
        )

        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue((self.anomaly_dir).exists())

    @patch('sigllm.benchmark._run_job')
    def test_benchmark_with_output_path(self, mock_run_job):
        output_path = os.path.join(self.temp_dir, 'output.csv')

        mock_output = pd.DataFrame({
            'pipeline': [self.pipeline_name],
            'dataset': [self.dataset_name],
            'signal': [self.signal_name],
            'iteration': [0],
            'f1': [0.8],
            'status': ['OK'],
            'elapsed': [1.0],
            'split': [False],
            'run_id': [self.run_id],
        })
        mock_run_job.return_value = mock_output

        result = benchmark(
            pipelines=self.pipelines,
            datasets=self.datasets,
            anomaly_dir=self.anomaly_dir,
            output_path=output_path,
        )

        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(os.path.exists(output_path))

    @patch('sigllm.benchmark._run_job')
    def test_benchmark_with_iterations(self, mock_run_job):
        mock_output = pd.DataFrame({
            'pipeline': [self.pipeline_name],
            'dataset': [self.dataset_name],
            'signal': [self.signal_name],
            'iteration': [0],
            'f1': [0.8],
            'status': ['OK'],
            'elapsed': [1.0],
            'split': [False],
            'run_id': [self.run_id],
        })
        mock_run_job.return_value = mock_output

        result = benchmark(pipelines=self.pipelines, datasets=self.datasets, iterations=3)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(mock_run_job.call_count, 3)

    @patch('sigllm.benchmark._run_job')
    def test_benchmark_empty_results(self, mock_run_job):
        empty_pipelines = ['nonexistent_pipeline']
        empty_datasets = {'nonexistent_dataset': []}

        # Run benchmark with minimal configuration
        result = benchmark(pipelines=empty_pipelines, datasets=empty_datasets)

        mock_run_job.assert_not_called()

        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)

    @patch('sigllm.benchmark._run_job')
    def test_benchmark_with_resume(self, mock_run_job):
        os.makedirs(self.cache_dir, exist_ok=True)

        cached_file = (
            self.cache_dir / f'{self.pipeline_name}'
            f'_{self.signal_name}_{self.dataset_name}_0_test_scores.csv'
        )

        pd.DataFrame({
            'pipeline': [self.pipeline_name],
            'dataset': [self.dataset_name],
            'signal': [self.signal_name],
            'iteration': [0],
            'f1': [0.8],
            'status': ['OK'],
            'elapsed': [1.0],
            'split': [False],
            'run_id': [self.run_id],
        }).to_csv(cached_file, index=False)

        result = benchmark(
            pipelines=self.pipelines, datasets=self.datasets, cache_dir=self.cache_dir, resume=True
        )

        mock_run_job.assert_not_called()

        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)
