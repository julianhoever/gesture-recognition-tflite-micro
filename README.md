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

3. When interested in the machine learning part: Create conda environment
```bash
conda create -n gesture-recognition-tflite-micro python=3.12
conda activate gesture-recognition-tflite-micro
poetry install
```

## What works?
- TensorFlow -> TensorFlow Lite -> MCU

## What does (currently) not work?
- PyTorch -> ? -> Hardware
    - PyTorch -> ONNX -> TensorFlow -> TensorFlow Lite -> MCU
        - [ONNX-TF](https://github.com/onnx/onnx-tensorflow)
            - Deprecated and uses deprecated libraries
    - PyTorch -> TensorFlow Lite -> MCU
        - [ai-edge-torch](https://github.com/google-ai-edge/ai-edge-torch)
            - Very new tool by Google
            - At least for me, not usable at this point
                - Maybe improves in the future
    - PyTorch -> MCU
        - [ExecuTorch](https://pytorch.org/executorch-overview)
            - Looks promising but cannot be tested because I cannot install it
    - PyTorch -> Manual implementation -> MCU
        - [CMSIS-NN](https://arm-software.github.io/CMSIS-NN/latest/index.html)
            - Used by TensorFlow Lite
            - Very manual way
            - Model needs to be written in C/C++ code and all parameters must be extracted from the python model
            - Needs more work to get this working
            - To automate this process feels like, creating a second "Creator" tool...

## Still open...
- ???