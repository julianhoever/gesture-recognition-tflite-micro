from pathlib import Path


DATA_ROOT = Path("data/gestures")
OUTPUTS_DIR = Path("outputs")

KERAS_MODEL_FILE = OUTPUTS_DIR / "model.keras"
TFLITE_MODEL_FILE = OUTPUTS_DIR / "model.tflite"
C_ARRAY_MODEL_FILE = OUTPUTS_DIR / "model.c"
