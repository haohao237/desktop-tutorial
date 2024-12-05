import pandas as pd

def preprocess_data():
    # CSVを読み込み、必要な前処理（例: テキストの正規化）を実行
    df = pd.read_csv('data/data.csv')
    
    # 必要に応じてテキストのクリーニングや正規化を行います
    # 例: 大文字・小文字の統一、特殊文字の除去など
    df['question'] = df['question'].str.lower()
    df['explanation'] = df['explanation'].str.lower()

    return df
