# -*- coding: utf-8 -*-

"""
Main module.

This is an extension to Orion's core module
"""
from typing import Union

from mlblocks import MLPipeline
from orion import Orion

from sigllm.primitives.prompting.anomalies import get_anomaly_list_within_seq, str2idx
from sigllm.primitives.prompting.data import sig2str

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


def get_anomalies(seq, msg_func, model_func, num_iters=1, alpha=0.5):
    """Get LLM anomaly detection results.

    The function get the LLM's anomaly detection and converts them into an 1D array

    Args:
        seq (ndarray):
            The sequence to detect anomalies.
        msg_func (func):
            Function to create message prompt.
        model_func (func):
            Function to get LLM answer.
        num_iters (int):
            Number of times to run the same query.
        alpha (float):
            Percentage of total number of votes that an index needs to have to be
            considered anomalous. Default: 0.5

    Returns:
        ndarray:
            1D array containing anomalous indices of the sequence.
    """
    message = msg_func(sig2str(seq, space=True))
    res_list = []
    for i in range(num_iters):
        res = model_func(message)
        ano_ind = str2idx(res, len(seq))
        res_list.append(ano_ind)
    return get_anomaly_list_within_seq(res_list, alpha=alpha)
