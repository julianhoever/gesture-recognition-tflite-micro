import keras


def gesture_cnn_model(input_shape: tuple[int | None, ...]) -> keras.Sequential:
    def conv_block(filters: int, kernel_size: int) -> keras.Sequential:
        return keras.Sequential(
            [
                keras.layers.Conv1D(filters=filters, kernel_size=kernel_size),
                keras.layers.ReLU(),
                keras.layers.BatchNormalization(),
            ]
        )

    model = keras.Sequential(
        [
            keras.layers.AvgPool1D(pool_size=16, strides=1),
            conv_block(filters=16, kernel_size=4),
            keras.layers.Flatten(),
            keras.layers.Dense(units=32, activation="relu"),
            keras.layers.Dropout(rate=0.2),
            keras.layers.Dense(units=4, activation="softmax"),
        ]
    )
    model.build(input_shape=input_shape)
    return model
