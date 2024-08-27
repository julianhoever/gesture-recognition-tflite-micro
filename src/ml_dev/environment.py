from pathlib import Path

# Paths

DATA_ROOT = Path("data/gestures")
OUTPUTS_DIR = Path("outputs")

MODEL_WEIGHTS_FILE = OUTPUTS_DIR / "model.weights.h5"
TFLITE_MODEL_FILE = OUTPUTS_DIR / "model.tflite"
CPP_ARRAY_MODEL_FILE = OUTPUTS_DIR / "model.cpp"

# ML Constants

SAMPLE_SHAPE = (125, 3)
