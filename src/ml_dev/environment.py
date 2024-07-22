from pathlib import Path

import torch

# Paths

DATA_ROOT = Path("data/gestures")
OUTPUTS_DIR = Path("outputs")

TF_OUTPUTS_DIR = OUTPUTS_DIR / "tensorflow"
TF_MODEL_WEIGHTS_FILE = TF_OUTPUTS_DIR / "model.weights.h5"
TF_TFLITE_MODEL_FILE = TF_OUTPUTS_DIR / "model.tflite"
TF_CPP_ARRAY_MODEL_FILE = TF_OUTPUTS_DIR / "model.cpp"

PT_OUTPUTS_DIR = OUTPUTS_DIR / "pytorch"
PT_MODEL_WEIGHTS_FILE = PT_OUTPUTS_DIR / "model.weights.pt"
PT_TFLITE_MODEL_FILE = PT_OUTPUTS_DIR / "model.tflite"
PT_CPP_ARRAY_MODEL_FILE = PT_OUTPUTS_DIR / "model.cpp"

# ML Constants

SAMPLE_SHAPE = (125, 3)


def _determine_available_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


PT_DEVICE = _determine_available_device()
