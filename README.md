# Multi-Machine Disruption Prediction Challenge

This repository contains a streamlined version of the code that achieved 1st place in the [Multi-Machine Disruption Prediction Challenge for Fusion Energy by ITU](https://zindi.africa/competitions/multi-machine-disruption-prediction-challenge/) competition on [Zindi](https://zindi.africa/).

The code involves two main scripts:
* `predictions.py` - a script that uses logistic regression to predict whether each pulse is disruptive or not, based on features extracted from signals, where each signal is used as a separate predictor.
* `submission.py` - a script that creates a submission by averaging the predictions from multiple signals/predictors.

This code uses [python](https://www.python.org/), [numpy](https://numpy.org/), [pandas](https://pandas.pydata.org/), [h5py](https://www.h5py.org/), [scikit-learn](https://scikit-learn.org/) and [tsfresh](https://tsfresh.readthedocs.io/).

## How to set up folders

Download `C-Mod data.zip` from the [competition website](https://zindi.africa/competitions/multi-machine-disruption-prediction-challenge/data) and unzip it to some location.

In `predictions.py`, line `18`, set `cmod_data_folder` to point to the location of the C-Mod data folder (the `cmod` folder that comes from `C-Mod data.zip`).

For platform independence, it is better to use `os.path.join()` rather than using explicit forward slashes (`/`) or backward slashes (`\`) to build such path.

## Order in which to run code

1. Run `python predictions.py`.

   This will generate the output files `coefficients.csv` and `predictions.csv`.

2. Run `python submission.py`.

   This will generate the output file `submission.csv`.

## Explanations of features used

This code uses:
* [h5py](https://www.h5py.org/) to read HDF files,
* [pandas](https://pandas.pydata.org/) to process CSV files,
* [tsfresh](https://tsfresh.readthedocs.io/) to compute features from time series,
* [scikit-learn](https://scikit-learn.org/) to implement a logistics regression pipeline.

## Environment for the code to run

This code runs in an environment containing:
* python 3.11.6
* numpy 1.26.1
* pandas 2.1.1
* h5py 3.10.0
* scikit-learn 1.3.1
* tsfresh 0.20.1

## Hardware needed

No special hardware. No accelerators needed (e.g. GPUs).

However, some parts of the code are multi-core, since `tsfresh` uses Python's `multiprocessing` during feature computation.

## Expected run time for each script

* `predictions.py` might take a few hours to run (e.g. 2 to 3 hours, depending on hardware)
* `submission.py` is immediate.
