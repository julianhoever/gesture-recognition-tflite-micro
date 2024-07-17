import tensorflow as tf


def _center_channels(data: tf.Tensor) -> tf.Tensor:
    mean = tf.expand_dims(tf.reduce_mean(data, axis=-2), axis=-2)
    return data - mean


def _rescale(data: tf.Tensor) -> tf.Tensor:
    abs_max = tf.reduce_max(tf.abs(data), axis=(-2, -1))
    abs_max = tf.reshape(abs_max, (-1, 1, 1))
    return data / abs_max


def preprocess(data: tf.Tensor) -> tf.Tensor:
    x = _center_channels(data)
    x = _rescale(x)
    return x
