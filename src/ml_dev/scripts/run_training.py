from functools import partial
from pathlib import Path

import keras
import tensorflow as tf
import matplotlib.pyplot as plt

from ml_dev.gesture_dataset import load_gesture_data
from ml_dev.preprocessing import normalize
from ml_dev.gesture_cnn_model import gesture_cnn_model
from ml_dev.environment import DATA_ROOT, OUTPUTS_DIR, KERAS_MODEL_FILE


def main() -> None:
    OUTPUTS_DIR.mkdir(exist_ok=True)

    (x_train, y_train), (x_val, y_val) = _load_data()
    model = gesture_cnn_model((None, 125, 3))
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[keras.metrics.CategoricalAccuracy()],
    )
    history = model.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_val, y_val),
        batch_size=64,
        epochs=20,
        shuffle=True,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                filepath=KERAS_MODEL_FILE,
                monitor="val_categorical_accuracy",
                save_best_only=True,
            )
        ],
    )
    model.load_weights(KERAS_MODEL_FILE)
    model.evaluate(x=x_val, y=y_val)

    _plot_train_history(history)


def _load_data() -> tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]]:
    load_data = partial(
        load_gesture_data, data_root=DATA_ROOT, window_size=125, stride=1
    )
    x_train, y_train = load_data(training=True)
    x_val, y_val = load_data(training=False)

    x_train = normalize(x_train)
    x_val = normalize(x_val)

    return (x_train, y_train), (x_val, y_val)


def _plot_train_history(history: keras.callbacks.History) -> None:
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(history.history["loss"])
    ax.plot(history.history["val_loss"])
    fig.savefig(OUTPUTS_DIR / "train_history.png")


if __name__ == "__main__":
    main()
