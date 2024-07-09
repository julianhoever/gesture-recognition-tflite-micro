from pathlib import Path

import numpy as np

from ml_dev.gesture_dataset import load_gesture_data
from ml_dev.preprocessing import normalize
from ml_dev.tflite.execute_tflite_model import execute_tflite_model
from ml_dev.environment import DATA_ROOT, TFLITE_MODEL_FILE


def _load_data(training: bool) -> tuple[np.ndarray, np.ndarray]:
    x, y = load_gesture_data(DATA_ROOT, training, window_size=125, stride=1)
    x = normalize(x)
    return x.numpy(), y.numpy()


def main() -> None:
    x_train, y_train = _load_data(training=True)
    x_val, y_val = _load_data(training=False)

    pred_train = execute_tflite_model(TFLITE_MODEL_FILE, x_train)

    print()


if __name__ == "__main__":
    main()
