from functools import partial

import keras
import tensorflow as tf
import matplotlib.pyplot as plt

from ml_dev.gesture_dataset import load_gesture_data
from ml_dev.preprocessing import normalize
from ml_dev.gesture_cnn_model import gesture_cnn_model
from ml_dev.environment import DATA_ROOT, OUTPUTS_DIR, KERAS_MODEL_FILE


def main() -> None:
    OUTPUTS_DIR.mkdir(exist_ok=True)

    window_size = 125

    (x_train, y_train), (x_val, y_val) = _load_data(window_size)
    model = gesture_cnn_model((None, window_size, 3))
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-3),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[keras.metrics.Accuracy()],
    )
    history = model.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_val, y_val),
        batch_size=512,
        epochs=100,
        shuffle=True,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                filepath=KERAS_MODEL_FILE,
                monitor="val_accuracy",
                save_best_only=True,
            )
        ],
    )
    model.load_weights(KERAS_MODEL_FILE)
    model.evaluate(x=x_val, y=y_val)

    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.show()


def _load_data(
    window_size: int,
) -> tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]]:
    load_data = partial(
        load_gesture_data, data_root=DATA_ROOT, window_size=window_size, stride=1
    )
    x_train, y_train = load_data(training=True)
    x_val, y_val = load_data(training=False)

    x_train = normalize(x_train)
    x_val = normalize(x_val)

    return (x_train, y_train), (x_val, y_val)


if __name__ == "__main__":
    main()
