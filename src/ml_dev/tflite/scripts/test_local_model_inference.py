import numpy as np

from ml_dev.gesture_dataset import load_gesture_data
from ml_dev.preprocessing import normalize
from ml_dev.tflite.execute_tflite_model import execute_uint8_tflite_model
from ml_dev.environment import DATA_ROOT, TFLITE_MODEL_FILE, SAMPLE_SHAPE


def main() -> None:
    x_train, y_train = _load_data(training=True)
    x_val, y_val = _load_data(training=False)

    pred_train = execute_uint8_tflite_model(TFLITE_MODEL_FILE, x_train)
    pred_val = execute_uint8_tflite_model(TFLITE_MODEL_FILE, x_val)

    print(f"Train Accuracy: {_acc(pred_train, y_train):.04f}")
    print(f"Validation Accuracy: {_acc(pred_val, y_val):.04f}")


def _load_data(training: bool) -> tuple[np.ndarray, np.ndarray]:
    x, y = load_gesture_data(DATA_ROOT, training, window_size=SAMPLE_SHAPE[0], stride=1)
    x = normalize(x).numpy()
    y = np.argmax(y.numpy(), axis=-1)
    return x, y


def _acc(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sum(pred == target) / len(target))


if __name__ == "__main__":
    main()
