import tensorflow as tf

def build_cnn_bilstm_model(
    seq_length,
    n_features=1
):

    model = tf.keras.models.Sequential([

        tf.keras.layers.Input(
            shape=[seq_length, n_features]
        ),

        tf.keras.layers.Conv1D(
            128,
            kernel_size=3,
            activation='relu'
        ),

        tf.keras.layers.MaxPooling1D(
            pool_size=2
        ),

        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                150,
                return_sequences=True
            )
        ),

        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                50,
                return_sequences=False
            )
        ),

        tf.keras.layers.Dense(
            64,
            activation='relu'
        ),

        tf.keras.layers.Dense(
            32,
            activation='relu'
        ),

        tf.keras.layers.Dense(1)

    ])

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001
    )

    model.compile(
        optimizer=optimizer,
        loss='mean_squared_error'
    )

    return model