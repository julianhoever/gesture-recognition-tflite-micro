# gesture-recognition-tflite-micro

## Prerequisites
- [Anaconda / Miniconda](https://docs.anaconda.com/miniconda/#quick-command-line-install)
- [Poetry](https://python-poetry.org/docs/#installation)

## Install

1. Clone repo
```bash
git clone git@github.com:julianhoever/gesture-recognition-tflite-micro.git
cd gesture-recognition-tflite-micro
```

2. When interested in the hardware implementation: Initialize submodules
```bash
git submodule sync
git submodule update --init --recursive
```

3. When interested in the machine learning part: Create conda environment and install dependencies using poetry
```bash
conda create -n gesture-recognition-tflite-micro python=3.12
conda activate gesture-recognition-tflite-micro
poetry install
```
