import requests
import pandas as pd

API_SERVER_URL = "http://172.16.2.32:11434/api/chat"
questions_df = pd.read_csv('./data/questions.csv')
answers_df = pd.read_csv('./data/answers.csv')

def generate_feedback(question_id):
    # 問題情報を取得
    question_info = questions_df[questions_df['question_id'] == question_id].iloc[0]
    user_answer = answers_df[answers_df['question_id'] == question_id]['is_correct'].iloc[0]

    # フィードバック作成
    feedback = {
        "question_id": question_info['question_id'],
        "question": question_info['question'],
        "correct_answer": question_info['correct_answer'],
        "user_answer": user_answer,
        "time_taken": answers_df[answers_df['question_id'] == question_id]['time_taken'].iloc[0]
    }

    # APIリクエスト送信
    response = requests.post(API_SERVER_URL, headers={"Content-Type": "application/json"}, json=feedback)
    response.raise_for_status()

    print(response.text)

# 複数の問題に対してフィードバックを生成
def generate_feedback_batch(question_ids):
    for question_id in question_ids:
        generate_feedback(question_id)

# 例として複数の問題IDを指定
generate_feedback_batch([1, 2, 3, 4, 5])
