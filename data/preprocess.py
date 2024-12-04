import pandas as pd

def preprocess_data():
    # CSVを読み込み、必要な前処理（例: テキストの正規化）を実行
    df = pd.read_csv('data/data.csv')
    return df
