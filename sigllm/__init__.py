# -*- coding: utf-8 -*-

"""Top-level package for sigllm."""

__author__ = 'MIT Data To AI Lab'
__email__ = 'dailabmit@gmail.com'
__version__ = '0.0.1.dev0'

import os

_BASE_PATH = os.path.abspath(os.path.dirname(__file__))
MLBLOCKS_PRIMITIVES = os.path.join(_BASE_PATH, 'primitives', 'jsons')
MLBLOCKS_PIPELINES = tuple([
    os.path.join(_BASE_PATH, 'pipelines', 'prompter'),
    os.path.join(_BASE_PATH, 'pipelines', 'detector')
])
