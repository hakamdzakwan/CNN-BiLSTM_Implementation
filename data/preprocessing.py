from sklearn.preprocessing import MinMaxScaler

def normalize_data(data):

    scaler = MinMaxScaler(
        feature_range=(0, 1)
    )

    data_scaled = scaler.fit_transform(data)

    return data_scaled, scaler


def train_val_test_split(
    data,
    train_ratio=0.7,
    val_ratio=0.15
):

    train_size = int(len(data) * train_ratio)

    val_size = int(len(data) * val_ratio)

    train_data = data[:train_size]

    val_data = data[
        train_size:train_size + val_size
    ]

    test_data = data[
        train_size + val_size:
    ]

    return (
        train_data,
        val_data,
        test_data
    )