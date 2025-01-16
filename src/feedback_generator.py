import torch
import pandas as pd
from train_model import FeedbackModel

# 学習時に利用したラベルの数を取得
questions = pd.read_csv("/app/data/questions.csv")
answers = pd.read_csv("/app/data/answers.csv")
merged_data = answers.merge(questions, on="question_id")

# フィードバックラベルのユニーク数を取得
merged_data["correct_rate"] = merged_data.groupby("tag")["is_correct"].transform("mean")
merged_data["avg_time"] = merged_data.groupby("tag")["time_taken"].transform("mean")
feedback_labels = pd.read_csv("/app/data/feedback_samples.csv")  # feedback_samples.csvをロードしてラベルを取得
num_labels = len(feedback_labels['feedback'].unique())  # 修正：ユニークなフィードバックラベル数を取得

# モデルのロード
model = FeedbackModel(num_labels=num_labels)  # num_labels を渡す
model.load_state_dict(torch.load("/app/models/feedback_model/model.pth"))
model.eval()

# ユーザーの回答データを解析
feedback_results = []

for tag, group in merged_data.groupby("tag"):
    correct_rate = group["is_correct"].mean()
    avg_time = group["time_taken"].mean()

    # 特徴量をモデルに入力
    input_data = torch.tensor([correct_rate, avg_time], dtype=torch.float32).unsqueeze(0)
    feedback_id = torch.argmax(model(input_data), dim=1).item()  # 最も確率が高いクラスを取得
    feedback_text = feedback_labels.loc[feedback_labels['feedback_id'] == feedback_id, 'feedback'].values[0]  # 修正：ラベルIDからテキストを取得

    # フィードバック内容を格納
    feedback_results.append({
        "tag": tag,
        "correct_rate": correct_rate,
        "avg_time": avg_time,
        "feedback": feedback_text
    })

    # フィードバック内容を出力
    print(f"Tag: {tag}, Correct Rate: {correct_rate:.2f}, Avg Time: {avg_time:.2f}, Feedback: {feedback_text}")

# フィードバック結果をCSVに保存
feedback_df = pd.DataFrame(feedback_results)
feedback_df.to_csv("/app/data/generated_feedback.csv", index=False)

print("フィードバック生成が完了しました。")
