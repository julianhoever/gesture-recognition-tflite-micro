import keras
import tensorflow as tf

from ml_dev.gesture_dataset import load_gesture_data
from ml_dev.preprocessing import normalize
from ml_dev.environment import DATA_ROOT, KERAS_MODEL_FILE


def main() -> None:
    x_train, y_train = load_gesture_data(DATA_ROOT, training=True)
    x_val, y_val = load_gesture_data(DATA_ROOT, training=False)
    x_train, x_val = normalize(x_train), normalize(x_val)

    model = keras.saving.load_model(KERAS_MODEL_FILE)

    pred_train = model.predict(x_train)
    pred_val = model.predict(x_val)

    print(f"Train Accuracy: {_acc(pred_train, y_train):.04f}")
    print(f"Validation Accuracy: {_acc(pred_val, y_val):.04f}")


def _label_indices(one_hot_labels: tf.Tensor) -> tf.Tensor:
    return tf.argmax(one_hot_labels, axis=-1)


def _acc(pred: tf.Tensor, target: tf.Tensor) -> float:
    pred_labels = _label_indices(pred)
    target_labels = _label_indices(target)
    correct_predicted = tf.reduce_sum(tf.where(pred_labels == target_labels, 1, 0))
    return float(correct_predicted / len(target))


if __name__ == "__main__":
    main()
