import os
import matplotlib.pyplot as plt

FIGURE_DIR = "outputs/figures"

os.makedirs(FIGURE_DIR, exist_ok=True)


def plot_predictions(
    actual,
    predicted,
    title="Predicted vs Actual"
):

    plt.figure(figsize=(14, 7))

    plt.plot(
        actual,
        label='Actual Price',
        color='blue'
    )

    plt.plot(
        predicted,
        label='Predicted Price',
        color='red',
        linestyle='--'
    )

    plt.title(title)

    plt.xlabel('Index')

    plt.ylabel('Price')

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        f"{FIGURE_DIR}/prediction_vs_actual.png",
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()


def plot_loss(history):

    plt.figure(figsize=(12, 6))

    plt.plot(
        history.history['loss'],
        label='Training Loss'
    )

    plt.plot(
        history.history['val_loss'],
        label='Validation Loss'
    )

    plt.title('Model Loss')

    plt.xlabel('Epoch')

    plt.ylabel('Loss')

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        f"{FIGURE_DIR}/training_loss.png",
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()


def plot_metrics(mape, r2):

    fig, ax = plt.subplots(
        1,
        2,
        figsize=(12, 5)
    )

    ax[0].bar(
        ['MAPE'],
        [mape],
        color='orangered'
    )

    ax[0].set_title(
        'Mean Absolute Percentage Error'
    )

    ax[0].set_ylabel('Value (%)')

    ax[0].text(
        0,
        mape / 2,
        f'{mape:.2f}%',
        ha='center',
        va='center',
        color='white',
        fontweight='bold'
    )

    ax[1].bar(
        ['R²'],
        [r2],
        color='royalblue'
    )

    ax[1].set_title('R² Score')

    ax[1].set_ylim(0, 1.1)

    ax[1].set_ylabel('Value')

    ax[1].text(
        0,
        r2 / 2,
        f'{r2:.4f}',
        ha='center',
        va='center',
        color='white',
        fontweight='bold'
    )

    plt.tight_layout()

    plt.savefig(
        f"{FIGURE_DIR}/evaluation_metrics.png",
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()