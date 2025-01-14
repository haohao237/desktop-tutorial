import torch
import pandas as pd
from train_model import FeedbackModel

# モデルのロード
model = FeedbackModel()
model.load_state_dict(torch.load("/app/models/feedback_model/model.pth"))
model.eval()

# ユーザーの回答データを解析
answers = pd.read_csv("/app/data/answers.csv")
questions = pd.read_csv("/app/data/questions.csv")
merged_data = answers.merge(questions, on="question_id")

# 各タグ別にフィードバックを生成
for tag, group in merged_data.groupby("tag"):
    correct_rate = group["is_correct"].mean()
    avg_time = group["time_taken"].mean()

    # フィードバックを生成
    input_data = torch.tensor([correct_rate, avg_time], dtype=torch.float32).unsqueeze(0)
    feedback = model(input_data).item()

    print(f"Tag: {tag}, Correct Rate: {correct_rate:.2f}, Avg Time: {avg_time:.2f}")
    print(f"Feedback: {feedback}") 
