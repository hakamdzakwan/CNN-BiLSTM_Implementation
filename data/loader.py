import pandas as pd

def load_data(file_path):

    df = pd.read_csv(file_path)

    df['Date'] = pd.to_datetime(df['Date'])

    df = df.sort_values('Date')

    df.set_index('Date', inplace=True)

    return df