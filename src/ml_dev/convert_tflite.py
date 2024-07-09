import keras
import tensorflow as tf

from ml_dev.environment import KERAS_MODEL_FILE, TFLITE_MODEL_FILE


def main() -> None:
    model = keras.saving.load_model(KERAS_MODEL_FILE)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with TFLITE_MODEL_FILE.open("wb") as out_file:
        out_file.write(tflite_model)


if __name__ == "__main__":
    main()
