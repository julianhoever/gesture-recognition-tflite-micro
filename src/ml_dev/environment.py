from pathlib import Path

import torch

# Paths

DATA_ROOT = Path("data/gestures")
OUTPUTS_DIR = Path("outputs")

TF_MODEL_WEIGHTS_FILE = OUTPUTS_DIR / "model.weights.h5"
PT_MODEL_WEIGHTS_FILE = OUTPUTS_DIR / "model.weights.pt"
TFLITE_MODEL_FILE = OUTPUTS_DIR / "model.tflite"
CPP_ARRAY_MODEL_FILE = OUTPUTS_DIR / "model.cpp"

# ML Constants

SAMPLE_SHAPE = (125, 3)


def _determine_available_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


PT_DEVICE = _determine_available_device()
