from sklearn.metrics import (
    r2_score,
    mean_absolute_percentage_error
)

def evaluate_model(
    model,
    X_test,
    y_test,
    scaler,
    seq_length,
    batch_size
):

    y_pred = model.predict(X_test)

    if scaler is not None:

        y_test_reshaped = y_test.reshape(
            -1,
            1
        )

        y_pred_reshaped = y_pred.reshape(
            -1,
            1
        )

        y_test_inv = scaler.inverse_transform(
            y_test_reshaped
        )

        y_pred_inv = scaler.inverse_transform(
            y_pred_reshaped
        )

    else:

        y_test_inv = y_test

        y_pred_inv = y_pred

    mape = (
        mean_absolute_percentage_error(
            y_test_inv,
            y_pred_inv
        ) * 100
    )

    r2 = r2_score(
        y_test_inv,
        y_pred_inv
    )

    print('')

    print('Model Training Completed')

    print(
        f'Sequence length used: '
        f'{seq_length}'
    )

    print(
        f'Batch size used: '
        f'{batch_size}'
    )

    print(f'MAPE: {mape:.4f}%')

    print(f'R² Score: {r2:.4f}\n')

    model.summary()

    return y_pred_inv, y_test_inv, mape, r2