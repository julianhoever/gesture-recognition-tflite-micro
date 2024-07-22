from pathlib import Path

import tensorflow as tf

from ml_dev.gesture_dataset import (
    LABEL_NAMES,
    load_gesture_data as _load_raw_gesture_data,
)


def load_gesture_data(
    data_root: Path, training: bool, window_size: int = 125, stride: int = 1
) -> tuple[tf.Tensor, tf.Tensor]:
    samples, labels = _load_raw_gesture_data(data_root, training, window_size, stride)
    samples = tf.convert_to_tensor(samples)
    labels = tf.one_hot(labels, depth=len(LABEL_NAMES))
    return samples, labels
