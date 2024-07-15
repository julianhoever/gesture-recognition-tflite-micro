from functools import partial

import tensorflow as tf

from ml_dev.gesture_cnn_model import gesture_cnn_model
from ml_dev.gesture_dataset import load_gesture_data
from ml_dev.preprocessing import normalize
from ml_dev.environment import DATA_ROOT, MODEL_WEIGHTS_FILE, SAMPLE_SHAPE


def main() -> None:
    load_data = partial(
        load_gesture_data, data_root=DATA_ROOT, window_size=SAMPLE_SHAPE[0]
    )
    x_train, y_train = load_data(training=True)
    x_val, y_val = load_data(training=False)
    x_train, x_val = normalize(x_train), normalize(x_val)

    model = gesture_cnn_model((None, *SAMPLE_SHAPE))
    model.load_weights(MODEL_WEIGHTS_FILE)

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
