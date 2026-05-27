import os
import tensorflow as tf

from utils.seed import (
    set_random_seeds
)

from data.loader import (
    load_data
)

from data.preprocessing import (
    normalize_data,
    train_val_test_split
)

from data.sequence import (
    prepare_datasets
)

from models.cnn_bilstm import (
    build_cnn_bilstm_model
)

from utils.metrics import (
    evaluate_model
)

from utils.plotting import (
    plot_predictions,
    plot_loss,
    plot_metrics
)

MODEL_DIR = "outputs/models"

os.makedirs(MODEL_DIR, exist_ok=True)


def run_time_series_pipeline(
    file_path,
    seq_length=64,
    epochs=100,
    batch_size=64,
    seed=42
):

    set_random_seeds(seed)

    # Load data
    df = load_data(file_path)

    data = df['Close'].values.reshape(
        -1,
        1
    )

    data_scaled, scaler = normalize_data(
        data
    )

    # Split data
    train_data, val_data, test_data = (
        train_val_test_split(
            data_scaled
        )
    )

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    ) = prepare_datasets(
        train_data,
        val_data,
        test_data,
        seq_length
    )

    model = build_cnn_bilstm_model(
        seq_length
    )

    early_stopping = (
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(
            X_val,
            y_val
        ),
        callbacks=[early_stopping],
        verbose=1
    )

    y_pred, y_test_inv, mape, r2 = evaluate_model(
        model,
        X_test,
        y_test,
        scaler,
        seq_length,
        batch_size
    )

    # Visualization
    plot_loss(history)

    plot_predictions(
        y_test_inv.flatten(),
        y_pred.flatten(),
        "CNN-BiLSTM: Predicted vs Actual"
    )

    plot_metrics(mape, r2)

    # Save model
    model.save(
        f"{MODEL_DIR}/cnn_bilstm_model.keras"
    )

    print("\nModel successfully saved to outputs/models/")
    print("Figures successfully saved to outputs/figures/")

    return (
        model,
        history,
        (mape, r2),
        (y_test_inv, y_pred)
    )


if __name__ == "__main__":

    file_path = 'https://drive.google.com/uc?id=1GjHIxQMRU99PvMkCiQHe4Y_mq4HztIR0'

    model, history, metrics, predictions = (
        run_time_series_pipeline(
            file_path=file_path,
            seq_length=16,
            epochs=100,
            batch_size=16,
            seed=16
        )
    )