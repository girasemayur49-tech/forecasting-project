import pandas as pd
import numpy as np

def load_data():
    df = pd.read_csv("data/data.csv")
    return df

def prepare_lstm_data(df):
    values = df['value'].values
    X, y = [], []

    for i in range(len(values)-1):
        X.append(values[i])
        y.append(values[i+1])

    return np.array(X).reshape(-1,1,1), np.array(y)