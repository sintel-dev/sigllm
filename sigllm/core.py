# -*- coding: utf-8 -*-

"""
Main module.

This is an extension to Orion's core module
"""
from typing import Union

from mlblocks import MLPipeline
from orion import Orion


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
        window_size (int):
            Size of the input window.
        steps (int):
            Number of steps ahead to forecast.

        hyperparameters (dict):
            Additional hyperparameters to set to the Pipeline.
    """

    def __init__(self, pipeline: Union[str, dict, MLPipeline] = None,
                 hyperparameters: dict = None):
        self._pipeline = pipeline or self.DEFAULT_PIPELINE
        self._hyperparameters = hyperparameters
        self._mlpipeline = self._get_mlpipeline()
        self._fitted = False
