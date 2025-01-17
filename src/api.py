from fastapi import FastAPI
import torch
import pandas as pd
from train_model import FeedbackModel

app = FastAPI()

# モデルとラベルのロード
hidden_size = 64
num_labels = 100  # 適切な数を指定
model = FeedbackModel(input_size=2, hidden_size=hidden_size, num_labels=num_labels)
model.load_state_dict(torch.load("/app/models/feedback_model/model.pth"))
model.eval()

feedback_labels = pd.read_csv("/app/data/feedback_samples.csv")

@app.post("/generate_feedback/")
async def generate_feedback(correct_rate: float, avg_time: float):
    input_data = torch.tensor([correct_rate, avg_time], dtype=torch.float32).unsqueeze(0)
    feedback_id = torch.argmax(model(input_data), dim=1).item()
    feedback_text = feedback_labels.loc[feedback_labels["feedback_id"] == feedback_id, "feedback"].values[0]
    return {"feedback": feedback_text}
