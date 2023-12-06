<p align="left">
<img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt=“DAI-Lab” />
<i>An open source project from Data to AI Lab at MIT.</i>
</p>

<!-- Uncomment these lines after releasing the package to PyPI for version and downloads badges -->
<!--[![PyPI Shield](https://img.shields.io/pypi/v/sigllm.svg)](https://pypi.python.org/pypi/sigllm)-->
<!--[![Downloads](https://pepy.tech/badge/sigllm)](https://pepy.tech/project/sigllm)-->
[![Github Actions Shield](https://img.shields.io/github/workflow/status/sintel-dev/sigllm/Run%20Tests)](https://github.com/sintel-dev/sigllm/actions)



# sigllm

Signals plus LLMs

- Documentation: https://sintel-dev.github.io/sigllm
- Homepage: https://github.com/sintel-dev/sigllm

# Overview

TODO: Provide a short overview of the project here.

# Install

## Requirements

**sigllm** has been developed and tested on [Python 3.8, 3.9, 3.10 and 3.11](https://www.python.org/downloads/)

Also, although it is not strictly required, the usage of a [virtualenv](https://virtualenv.pypa.io/en/latest/)
is highly recommended in order to avoid interfering with other software installed in the system
in which **sigllm** is run.

These are the minimum commands needed to create a virtualenv using python3.8 for **sigllm**:

```bash
pip install virtualenv
virtualenv -p $(which python3.6) sigllm-venv
```

Afterwards, you have to execute this command to activate the virtualenv:

```bash
source sigllm-venv/bin/activate
```

Remember to execute it every time you start a new console to work on **sigllm**!

<!-- Uncomment this section after releasing the package to PyPI for installation instructions
## Install from PyPI

After creating the virtualenv and activating it, we recommend using
[pip](https://pip.pypa.io/en/stable/) in order to install **sigllm**:

```bash
pip install sigllm
```

This will pull and install the latest stable release from [PyPI](https://pypi.org/).
-->

## Install from source

With your virtualenv activated, you can clone the repository and install it from
source by running `make install` on the `stable` branch:

```bash
git clone git@github.com:sintel-dev/sigllm.git
cd sigllm
git checkout stable
make install
```

## Install for Development

If you want to contribute to the project, a few more steps are required to make the project ready
for development.

Please head to the [Contributing Guide](https://sintel-dev.github.io/sigllm/contributing.html#get-started)
for more details about this process.

# Quickstart

In this short tutorial we will guide you through a series of steps that will help you
getting started with **sigllm**.

TODO: Create a step by step guide here.

# What's next?

For more details about **sigllm** and all its possibilities
and features, please check the [documentation site](
https://sintel-dev.github.io/sigllm/).
