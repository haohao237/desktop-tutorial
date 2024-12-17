import pandas as pd
from sklearn.model_selection import train_test_split

# データ前処理
def preprocess_data(file_path):
    data = pd.read_csv(file_path)

    # 特徴量とラベルの分離
    X = data[["question", "choice1", "choice2", "choice3", "choice4"]]
    y = data["label"]

    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    train_X, test_X, train_y, test_y = preprocess_data("data.csv")
    print("データ前処理完了！")
