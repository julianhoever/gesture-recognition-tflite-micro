import tensorflow as tf

ABS_MAX_DATA = 20
ABS_MAX_DEVICE = 511
DEVICE_SPECIFIC_SCALING = ABS_MAX_DEVICE / ABS_MAX_DATA


def _center_channels(data: tf.Tensor) -> tf.Tensor:
    mean = tf.expand_dims(tf.reduce_mean(data, axis=-2), axis=-2)
    return data - mean


def preprocess(data: tf.Tensor) -> tf.Tensor:
    data *= DEVICE_SPECIFIC_SCALING
    data = tf.clip_by_value(data, -ABS_MAX_DEVICE, ABS_MAX_DEVICE)
    data = _center_channels(data)
    return data
