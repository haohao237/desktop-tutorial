import pandas as pd

def preprocess_data():
    # CSVを読み込み
    df = pd.read_csv('data/data.csv')
    return df
