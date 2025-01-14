import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os  # ディレクトリ作成のために必要

# データの読み込み
data = pd.read_csv("/app/data/answers.csv")
questions = pd.read_csv("/app/data/questions.csv")
merged_data = data.merge(questions, on="question_id")

# フィードバック生成用の特徴量
merged_data["correct_rate"] = merged_data.groupby("tag")["is_correct"].transform("mean")
merged_data["avg_time"] = merged_data.groupby("tag")["time_taken"].transform("mean")

# 正規化
merged_data["correct_rate"] = (merged_data["correct_rate"] - merged_data["correct_rate"].min()) / (merged_data["correct_rate"].max() - merged_data["correct_rate"].min())
merged_data["avg_time"] = (merged_data["avg_time"] - merged_data["avg_time"].min()) / (merged_data["avg_time"].max() - merged_data["avg_time"].min())

# フィードバック文を生成するためのラベルを設定
merged_data["feedback"] = merged_data.apply(
    lambda row: f"{row['tag']}の正答率は{row['correct_rate']:.2f}で、平均解答時間は{row['avg_time']:.2f}秒です。", axis=1
)

# ラベルをカテゴリ ID に変換
feedback_labels = merged_data["feedback"].astype("category")
merged_data["label_id"] = feedback_labels.cat.codes  # 数値ラベルに変換

class FeedbackDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        features = torch.tensor([row['correct_rate'], row['avg_time']], dtype=torch.float32)
        label = row["label_id"]
        return features, torch.tensor(label, dtype=torch.long)

# モデル定義
class FeedbackModel(nn.Module):
    def __init__(self, num_labels):
        super(FeedbackModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, num_labels)
        )

    def forward(self, x):
        return self.fc(x)

# データセットとデータローダ
train_dataset = FeedbackDataset(merged_data)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# モデルの初期化
num_labels = len(feedback_labels.cat.categories)
model = FeedbackModel(num_labels)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 学習
for epoch in range(50):  # エポック数を増やしてみる
    model.train()
    for features, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs.view(-1, num_labels), labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/50], Loss: {loss.item():.4f}")

# モデル保存
try:
    save_dir = "/app/models/feedback_model"
    os.makedirs(save_dir, exist_ok=True)  # ディレクトリが存在しなければ作成
    model_path = os.path.join(save_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"モデルが {model_path} に保存されました。")
except Exception as e:
    print(f"モデル保存中にエラーが発生しました: {e}")
