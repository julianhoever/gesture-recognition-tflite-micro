from collections.abc import Generator
from typing import Any

import keras
import tensorflow as tf

from ml_dev.gesture_dataset import load_gesture_data
from ml_dev.preprocessing import normalize
from ml_dev.environment import DATA_ROOT, KERAS_MODEL_FILE, TFLITE_MODEL_FILE


def representative_dataset() -> Generator[list[tf.Tensor], Any, Any]:
    samples, _ = load_gesture_data(DATA_ROOT, training=True)
    samples = normalize(samples)
    for sample in samples:
        sample_with_batch_dim = tf.expand_dims(sample, axis=0)
        yield [sample_with_batch_dim]


def main() -> None:
    model = keras.saving.load_model(KERAS_MODEL_FILE)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_model = converter.convert()

    with TFLITE_MODEL_FILE.open("wb") as out_file:
        out_file.write(tflite_model)


if __name__ == "__main__":
    main()
