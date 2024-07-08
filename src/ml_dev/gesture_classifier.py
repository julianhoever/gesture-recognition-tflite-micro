from pathlib import Path

import keras
import tensorflow as tf

from ml_dev.gesture_dataset import load_gesture_dataset

DATA_ROOT = Path("data/gestures")


class GestureModel(keras.Model):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = keras.layers.Flatten()
        self.hidden = keras.layers.Dense(units=100, activation="relu")
        self.outputs = keras.layers.Dense(units=4, activation="softmax")

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = self.flatten(inputs)
        x = self.hidden(x)
        return self.outputs(x)

    def build(self, input_shape: tuple) -> None:
        for layer in self.layers:
            layer.build(input_shape)
            input_shape = layer.compute_output_shape(input_shape)
        self.built = True


def main() -> None:
    ds_train = load_gesture_dataset(DATA_ROOT, train_split=True)
    ds_test = load_gesture_dataset(DATA_ROOT, train_split=False)

    model = GestureModel()
    model.build(input_shape=(125, 3))
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.Accuracy()],
    )
    model.fit(
        x=ds_train,
        validation_data=ds_test,
        batch_size=64,
        epochs=10,
    )


if __name__ == "__main__":
    main()
