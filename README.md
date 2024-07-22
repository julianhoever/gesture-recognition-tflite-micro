# gesture-recognition-tflite-rp2040

## Prerequisites
- [Anaconda / Miniconda](https://docs.anaconda.com/miniconda/#quick-command-line-install)

## Install (ML Dev)
```bash
git clone git@github.com:julianhoever/gesture-recognition-tflite-rp2040.git

cd gesture-recognition-tflite-rp2040

conda env create -f environment.yml

conda activate gesture-recognition-tflite-rp2040
```

## What works?
- TensorFlow -> TensorFlow Lite -> MCU

## What does not work?
- PyTorch -> ? -> Hardware
    - PyTorch -> ONNX -> TensorFlow -> TensorFlow Lite -> MCU
        - [ONNX-TF](https://github.com/onnx/onnx-tensorflow)
            - Deprecated and uses deprecated libraries
    - PyTorch -> TensorFlow Lite -> MCU
        - [ai-edge-torch](https://github.com/google-ai-edge/ai-edge-torch)
            - Very new tool by Google
            - At least for me, not usable at this point
                - Maybe improves in the future


## Still open...
- PyTorch -> MCU
    - [ExecuTorch](https://pytorch.org/executorch-overview)
