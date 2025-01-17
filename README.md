# Implementation of Deep Speech 2 Automatic Speech Recognition model from Scratch

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This repository contains a implementation of Deep Speech 2 Automatic Speech Recognition model made form scratch with PyTorch. Deep Speech 2 is an end-to-end deep learning model designed for Automatic Speech Recognition (ASR). It uses a combination of convolutional neural networks (CNNs) and recurrent neural networks (RNNs) to process raw audio features (e.g., spectrograms) and predict corresponding text transcriptions. The model is based on [Baidu DeepSpeech2](https://arxiv.org/abs/1512.02595) paper and follows this architecture:

<p align="center">
<img alt="model architecture" src="https://velog.velcdn.com/images/pass120/post/5b167fc2-1d24-4b91-8d91-5baef1b6a541/image.png" width="500"></p>

## Installation

Follow these steps to install the project:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env
   ```

   
1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

## How To Use

To train a model, run the following command:

```bash
python3 train.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

Where `CONFIG_NAME` is a config from `src/configs` and `HYDRA_CONFIG_ARGUMENTS` are optional arguments.

To run inference (evaluate the model or save predictions):

```bash
python3 inference.py HYDRA_CONFIG_ARGUMENTS
```

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
