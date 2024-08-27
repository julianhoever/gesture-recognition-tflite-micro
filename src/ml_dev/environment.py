from pathlib import Path

# Paths

DATA_ROOT = Path("data/gestures")
OUTPUTS_DIR = Path("outputs")

TF_OUTPUTS_DIR = OUTPUTS_DIR / "tensorflow"
TF_MODEL_WEIGHTS_FILE = TF_OUTPUTS_DIR / "model.weights.h5"
TF_TFLITE_MODEL_FILE = TF_OUTPUTS_DIR / "model.tflite"
TF_CPP_ARRAY_MODEL_FILE = TF_OUTPUTS_DIR / "model.cpp"

# ML Constants

SAMPLE_SHAPE = (125, 3)
