# -*- coding: utf-8 -*-

"""
Main module.

SigLLM is an extension to Orion's core module
"""
import logging
from typing import Union

import pandas as pd
from mlblocks import MLPipeline
from orion import Orion

LOGGER = logging.getLogger(__name__)

INTERVAL_PRIMITIVE = "mlstars.custom.timeseries_preprocessing.time_segments_aggregate#1"
DECIMAL_PRIMITIVE = "sigllm.primitives.transformation.Float2Scalar#1"
WINDOW_SIZE_PRIMITIVE = "sigllm.primitives.forecasting.custom.rolling_window_sequences#1"


class SigLLM(Orion):
    """SigLLM Class.

    The SigLLM Class provides the main anomaly detection functionalities
    of SigLLM and is responsible for the interaction with the underlying
    MLBlocks pipelines.

    Args:
        pipeline (str, dict or MLPipeline):
            Pipeline to use. It can be passed as:
                * An ``str`` with a path to a JSON file.
                * An ``str`` with the name of a registered pipeline.
                * An ``MLPipeline`` instance.
                * A ``dict`` with an ``MLPipeline`` specification.
        interval (int):
            Number of time points between one sample and another.
        decimal (int):
            Number of decimal points to keep from the float representation.
        window_size (int):
            Size of the input window.
        hyperparameters (dict):
            Additional hyperparameters to set to the Pipeline.
    """
    DEFAULT_PIPELINE = 'mistral_detector'

    def _augment_hyperparameters(self, primitive, key, value):
        if not value:
            return

        if self._hyperparameters is None:
            self._hyperparameters = {
                primitive: {}
            }
        else:
            if primitive not in self._hyperparameters:
                self._hyperparameters[primitive] = {}

        self._hyperparameters[primitive][key] = value

    def __init__(self, pipeline: Union[str, dict, MLPipeline] = None, interval: int = None,
                 decimal: int = None, window_size: int = None, hyperparameters: dict = None):
        self._pipeline = pipeline or self.DEFAULT_PIPELINE
        self._hyperparameters = hyperparameters
        self._mlpipeline = self._get_mlpipeline()
        self._fitted = False

        self.interval = interval
        self.decimal = decimal
        self.window_size = window_size

        self._augment_hyperparameters(INTERVAL_PRIMITIVE, 'interval', interval)
        self._augment_hyperparameters(DECIMAL_PRIMITIVE, 'decimal', decimal)
        self._augment_hyperparameters(WINDOW_SIZE_PRIMITIVE, 'window_size', window_size)

    def __repr__(self):
        if isinstance(self._pipeline, MLPipeline):
            pipeline = '\n'.join(
                '    {}'.format(primitive) for primitive in self._pipeline.to_dict()['primitives'])

        elif isinstance(self._pipeline, dict):
            pipeline = '\n'.join(
                '    {}'.format(primitive) for primitive in self._pipeline['primitives'])

        else:
            pipeline = '    {}'.format(self._pipeline)

        hyperparameters = None
        if self._hyperparameters is not None:
            hyperparameters = '\n'.join(
                '    {}: {}'.format(step, value) for step, value in self._hyperparameters.items())

        return (
            'SigLLM:\n{}\n'
            'hyperparameters:\n{}\n'
        ).format(
            pipeline,
            hyperparameters
        )

    def detect(self, data: pd.DataFrame, visualization: bool = False, **kwargs) -> pd.DataFrame:
        """Detect anomalies in the given data..

        If ``visualization=True``, also return the visualization
        outputs from the MLPipeline object.

        Args:
            data (DataFrame):
                Input data, passed as a ``pandas.DataFrame`` containing
                exactly two columns: timestamp and value.
            visualization (bool):
                If ``True``, also capture the ``visualization`` named
                output from the ``MLPipeline`` and return it as a second
                output.

        Returns:
            DataFrame or tuple:
                If visualization is ``False``, it returns the events
                DataFrame. If visualization is ``True``, it returns a
                tuple containing the events DataFrame followed by the
                visualization outputs dict.
        """
        if not self._fitted:
            self._mlpipeline = self._get_mlpipeline()

        result = self._detect(self._mlpipeline.fit, data, visualization, **kwargs)
        self._fitted = True

        return result
