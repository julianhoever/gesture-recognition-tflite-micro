from pathlib import Path

import numpy as np
import tensorflow as tf


def execute_uint8_tflite_model(tflite_file: Path, samples: np.ndarray) -> np.ndarray:
    interpreter = tf.lite.Interpreter(model_path=str(tflite_file))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    predictions = np.zeros(len(samples), dtype=np.int32)
    for i, sample in enumerate(samples):
        input_scale, input_zero_point = input_details["quantization"]
        sample = sample / input_scale + input_zero_point

        sample = np.expand_dims(sample, axis=0).astype(input_details["dtype"])
        interpreter.set_tensor(input_details["index"], sample)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])[0]

        predictions[i] = output.argmax()

    return predictions
