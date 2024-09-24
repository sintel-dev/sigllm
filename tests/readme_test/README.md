<p align="left">
<img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt=“DAI-Lab” />
<i>An open source project from Data to AI Lab at MIT.</i>
</p>

[![Development Status](https://img.shields.io/badge/Development%20Status-2%20--%20Pre--Alpha-yellow)](https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha)
[![Python](https://img.shields.io/badge/Python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue)](https://badge.fury.io/py/sigllm) 
[![PyPi Shield](https://img.shields.io/pypi/v/sigllm.svg)](https://pypi.python.org/pypi/sigllm)
[![Run Tests](https://github.com/sintel-dev/sigllm/actions/workflows/tests.yml/badge.svg)](https://github.com/sintel-dev/sigllm/actions/workflows/tests.yml)
[![Downloads](https://pepy.tech/badge/sigllm)](https://pepy.tech/project/sigllm)


# SigLLM

Using Large Language Models (LLMs) for time series anomaly detection.

<!-- - Documentation: https://sintel-dev.github.io/sigllm -->
- Homepage: https://github.com/sintel-dev/sigllm

# Overview

SigLLM is an extension of the Orion library, built to detect anomalies in time series data using LLMs.
We provide two types of pipelines for anomaly detection:
* **Prompter**: directly prompting LLMs to find anomalies in time series.
* **Detector**: using LLMs to forecast time series and finding anomalies through by comparing the real and forecasted signals.

For more details on our pipelines, please read our [paper](https://arxiv.org/pdf/2405.14755).

# Quickstart

## Install with pip

The easiest and recommended way to install **SigLLM** is using [pip](https://pip.pypa.io/en/stable/):

```bash
pip install sigllm
```
This will pull and install the latest stable release from [PyPi](https://pypi.org/).


In the following example we show how to use one of the **SigLLM Pipelines**.

# Detect anomalies using a SigLLM pipeline

We will load a demo data located in `tutorials/data.csv` for this example:

```python3
import pandas as pd

data = pd.read_csv('data.csv')
data.head()
```

which should show a signal with `timestamp` and `value`.
```
     timestamp      value
0   1222840800   6.357008
1   1222862400  12.763547
2   1222884000  18.204697
3   1222905600  21.972602
4   1222927200  23.986643
5   1222948800  24.906765
```

In this example we use `gpt_detector` pipeline and set some hyperparameters. In this case, we set the thresholding strategy to dynamic. The hyperparameters are optional and can be removed.

In addtion, the `SigLLM` object takes in a `decimal` argument to determine how many digits from the float value include. Here, we don't want to keep any decimal values, so we set it to zero.

```python3
from sigllm import SigLLM

hyperparameters = {
    "orion.primitives.timeseries_anomalies.find_anomalies#1": {
        "fixed_threshold": False
    }
}

sigllm = SigLLM(
    pipeline='gpt_detector',
    decimal=0,
    hyperparameters=hyperparameters
)
```

Now that we have initialized the pipeline, we are ready to use it to detect anomalies:

```python3
anomalies = sigllm.detect(data)
```
> :warning: Depending on the length of your timeseries, this might take time to run.

The output of the previous command will be a ``pandas.DataFrame`` containing a table of detected anomalies:

```
        start         end  severity
0  1225864800  1227139200  0.625879
```

# Resources

Additional resources that might be of interest:
* Learn about [Orion](https://github.com/sintel-dev/Orion).
* Read our [paper](https://arxiv.org/pdf/2405.14755).


# Citation

If you use **SigLLM** for your research, please consider citing the following paper:

Sarah Alnegheimish, Linh Nguyen, Laure Berti-Equille, Kalyan Veeramachaneni. [Can Large Language Models be Anomaly Detectors for Time Series?](https://arxiv.org/pdf/2405.14755).

```
@inproceedings{alnegheimish2024sigllm,
  title={Can Large Language Models be Anomaly Detectors for Time Series?},
  author={Alnegheimish, Sarah and Nguyen, Linh and Berti-Equille, Laure and Veeramachaneni, Kalyan},
  booktitle={2024 IEEE International Conferencze on Data Science and Advanced Analytics (IEEE DSAA)},
  organization={IEEE},
  year={2024}
}
```