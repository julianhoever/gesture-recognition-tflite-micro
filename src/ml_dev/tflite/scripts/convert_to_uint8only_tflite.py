from collections.abc import Generator
from typing import Any

import tensorflow as tf

from ml_dev.gesture_cnn_model import gesture_cnn_model
from ml_dev.gesture_dataset import load_gesture_data
from ml_dev.preprocessing import preprocess
from ml_dev.environment import (
    DATA_ROOT,
    MODEL_WEIGHTS_FILE,
    TFLITE_MODEL_FILE,
    SAMPLE_SHAPE,
)


def representative_dataset() -> Generator[list[tf.Tensor], Any, Any]:
    samples, _ = load_gesture_data(
        DATA_ROOT, training=True, window_size=SAMPLE_SHAPE[0]
    )
    samples = preprocess(samples)
    for sample in samples:
        sample_with_batch_dim = tf.expand_dims(sample, axis=0)
        yield [sample_with_batch_dim]


def main() -> None:
    model = gesture_cnn_model((None, *SAMPLE_SHAPE))
    model.load_weights(MODEL_WEIGHTS_FILE)

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
