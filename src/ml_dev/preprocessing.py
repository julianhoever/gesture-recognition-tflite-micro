import keras
import tensorflow as tf


def normalize(data: tf.Tensor) -> tf.Tensor:
    norm = keras.layers.Normalization(axis=(0, -1))
    norm.adapt(data)
    return norm(data)
