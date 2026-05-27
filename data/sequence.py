import numpy as np

def create_sequences(data, seq_len):

    X, y = [], []

    for i in range(seq_len, len(data)):

        X.append(
            data[i - seq_len:i]
        )

        y.append(data[i])

    return np.array(X), np.array(y)


def prepare_datasets(
    train_data,
    val_data,
    test_data,
    seq_length
):

    X_train, y_train = create_sequences(
        train_data,
        seq_length
    )

    X_val, y_val = create_sequences(
        val_data,
        seq_length
    )

    X_test, y_test = create_sequences(
        test_data,
        seq_length
    )

    X_train = X_train.reshape(
        (
            X_train.shape[0],
            X_train.shape[1],
            1
        )
    )

    X_val = X_val.reshape(
        (
            X_val.shape[0],
            X_val.shape[1],
            1
        )
    )

    X_test = X_test.reshape(
        (
            X_test.shape[0],
            X_test.shape[1],
            1
        )
    )

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    )