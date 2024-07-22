import keras


def gesture_cnn_model(input_shape: tuple[int | None, ...]) -> keras.Sequential:
    model = keras.Sequential(
        [
            keras.layers.DepthwiseConv1D(kernel_size=4),
            keras.layers.Conv1D(filters=32, kernel_size=1, activation="relu"),
            keras.layers.MaxPool1D(pool_size=3),
            keras.layers.BatchNormalization(),
            keras.layers.DepthwiseConv1D(kernel_size=4),
            keras.layers.Conv1D(filters=12, kernel_size=1, activation="relu"),
            keras.layers.MaxPool1D(pool_size=3),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(rate=0.3),
            keras.layers.Flatten(),
            keras.layers.Dense(units=4, activation="softmax"),
        ]
    )
    model.build(input_shape=input_shape)
    return model
