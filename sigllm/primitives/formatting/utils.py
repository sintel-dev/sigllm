import logging

import numpy as np
import pandas as pd
from mlblocks import MLPipeline

logger = logging.getLogger(__name__)


def create_test_data(N=25):
    """Create test data for formatting validation."""
    x1 = np.linspace(10, 9 + N, N) / 100
    x2 = np.array([i % 2 for i in range(N)])
    x3 = np.linspace(N + 40, 41, N) / 100

    return pd.DataFrame({
        'timestamp': np.linspace(0, 3600 * (N - 1), N),
        'x1': x1,
        'x2': x2,
        'x3': x3,
    })


def test_multivariate_formatting_validity(method, verbose=False):
    """Test that formatting method can round-trip data correctly."""
    if verbose:
        logger.info('Testing multivariate formatting method validity')

    raw_data = create_test_data().to_numpy()[:, 1:]
    windowed_data = np.array([raw_data[i : i + 15, :] for i in range(0, len(raw_data) - 15, 1)])
    data = (1000 * windowed_data).astype(int)
    if verbose:
        logger.info('Data shape: %s', data.shape)

    string_data = method.format_as_string(data, **method.config)
    LLM_mock_output = np.array(string_data).reshape(-1, 1)
    if verbose:
        logger.info('LLM mock output: %s', LLM_mock_output)
    integer_data = method.format_as_integer(LLM_mock_output, **method.config)
    if verbose:
        logger.info('Format as string output: %s', string_data)

    assert isinstance(string_data, list)
    assert isinstance(string_data[0], str)
    assert isinstance(integer_data, np.ndarray)

    if len(integer_data.flatten()) == len(data.flatten()):
        assert np.all(integer_data.flatten() == data.flatten())
    elif len(integer_data.flatten()) == len(data[:, :, 0].flatten()):
        assert np.all(integer_data.flatten() == data[:, :, 0].flatten())
    else:
        raise ValueError('Validation suite failed: Dimensions do not match')


def run_pipeline(
    method,
    data=None,
    interval=3600,
    window_size=15,
    verbose=True,
    samples=7,
    normalize=False,
    temp=0.1,
    multivariate_allowed_symbols=None,
    pipeline_name='mistral_detector',
    stride=1,
    n_clusters=2,
    strategy='scaling',
    steps_ahead=None,
):
    """Run the forecasting pipeline.

    Args:
        method (subclass of MultivariateFormattingMethod): The method to run the pipeline for.
        data (pd.DataFrame): The data to run the pipeline on.
        interval (int): The interval between timestamps in the data.
        window_size (int): The context length for each prediction window.
        samples (int): The number of times to run the LLM on each window.
        normalize (bool): Whether to normalize the data before running.
        multivariate_allowed_symbols (list): The allowed symbols for LLMs
            to output aside from digits.
        pipeline_name: The name of the pipeline we are wrapping
            (choice of `mistral_detector` and `gpt_detector`).
        stride: The gap between consecutive prediction windows.
        n_clusters: Not yet supported. Will be used with the `binning`
            pre-processing strategy in the future.
        strategy: For now, must be `scaling`. We will add option for
            `binning` in the future.
        steps_ahead: The amount of steps ahead to predict in each window.

    Returns:
        The errors, y_hat, and y for the pipeline.
    """
    if data is None:
        data = create_test_data()

    pipeline = MLPipeline(pipeline_name)
    digits_per_timestamp = method.config.get('digits_per_timestamp', 2)

    num_dims = len(data.columns) - 1

    if steps_ahead is not None:
        max_steps = max(steps_ahead)
        hf_steps = max_steps * (num_dims + 1)  # adding some padding here
    else:
        hf_steps = 2

    test_hyperparameters = {
        'mlstars.custom.timeseries_preprocessing.time_segments_aggregate#1': {
            'interval': interval
        },
        'sigllm.primitives.forecasting.custom.rolling_window_sequences#1': {
            'target_column': 0,
            'window_size': window_size,
            'target_size': max(steps_ahead) if steps_ahead else 1,
            'step_size': stride,
        },
        'sigllm.primitives.forecasting.huggingface.HF#1': {
            'samples': samples,
            'temp': temp,
            'multivariate_allowed_symbols': (
                [] if multivariate_allowed_symbols is None else multivariate_allowed_symbols
            ),
            'steps': hf_steps,
        },
    }

    if strategy == 'scaling':
        test_hyperparameters['sigllm.primitives.transformation.Float2Scalar#1'] = {
            'decimal': digits_per_timestamp,
            'rescale': True,
        }
    elif strategy == 'binning':
        test_hyperparameters['sigllm.primitives.transformation.Float2Scalar#1'] = {
            'strategy': 'binning',
            'n_clusters': n_clusters,
        }
        raise ValueError('Note that binning is not supported for now.')
    else:
        raise ValueError(
            f'Invalid strategy: {strategy}. Note that only scaling is supported for now.'
        )

    pipeline.set_hyperparameters(test_hyperparameters)
    if normalize:
        data = method.normalize_data(data)
    context = pipeline.fit(data, start_=0, output_=3)
    context['X'] = method.format_as_string(context['X'], **method.config)

    if method.method_name == 'persistence_control':
        context['y_hat'] = context['X']
    else:
        context = pipeline.fit(**context, start_=5, output_=5)

    if verbose:
        logger.info('y_hat example: %s', context['y_hat'][0][0])

    if steps_ahead is not None:
        return _process_multi_step_results(method, context, pipeline, steps_ahead, verbose)

    context['y_hat'] = method.format_as_integer(context['y_hat'], trunc=1)
    if verbose:
        logger.info('y_hat example: %s', context['y_hat'][0][0])

    context = pipeline.fit(**context, start_=7, output_=10)
    errors = np.round(context['errors'], 7)
    if verbose:
        logger.info('y_hat: %s', context['y_hat'])
        logger.info('y: %s', context['y'])
        logger.info('errors: %s', errors)

    return errors, context['y_hat'], context['y']


def _process_multi_step_results(method, context, pipeline, steps_ahead, verbose):
    """Process results for multi-step-ahead prediction.

    For multi-step predictions with stride > 1, we skip aggregate_rolling_window
    since there's no overlap between predictions. Each window gives one prediction
    per step, indexed sequentially (0, 1, 2, ...) regardless of actual stride.

    Returns:
        dict {step : {'errors': [...], 'y_hat': [...], 'y': [...]}}
    """
    y_hat_by_step = method.format_as_integer(context['y_hat'], steps_ahead=steps_ahead)

    results = {}

    for step in steps_ahead:
        step_context = context.copy()

        y_hat_step = y_hat_by_step[step]
        y_hat_float = np.array(
            [[v if v is not None else np.nan for v in row] for row in y_hat_step], dtype=float
        )
        step_context['y_hat'] = np.expand_dims(y_hat_float, axis=-1)
        step_context = pipeline.fit(**step_context, start_=7, output_=7)
        y_hat_agg = np.nanmedian(step_context['y_hat'], axis=1).squeeze()
        y_for_step = context['y'][:, step - 1] if context['y'].ndim > 1 else context['y']
        errors = np.round(y_hat_agg - y_for_step, 7)

        if verbose:
            logger.info(
                'Step %s - y_hat shape: %s, errors shape: %s', step, y_hat_agg.shape, errors.shape
            )

        results[step] = {
            'errors': errors,
            'y_hat': y_hat_agg,
            'y': y_for_step,
        }

    return results
