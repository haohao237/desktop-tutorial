from openai import OpenAI
import pandas as pd

questions_df = pd.read_csv('./data/questions.csv')
answers_df = pd.read_csv('./data/answers.csv')

client = OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama', # required, but unused
)

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

    # APIリクエスト作成
    response = client.chat.completions.create(
        model="ELYZA",
        messages=[
            {"role": "system", "content": "あなたはAIのスペシャリストです。"},
            {"role": "user", "content": feedback}
        ]
    )
    print(response.choices[0].message.content)

# 複数の問題に対してフィードバックを生成
def generate_feedback_batch(question_ids):
    for question_id in question_ids:
        generate_feedback(question_id)

# 例として複数の問題IDを指定
generate_feedback_batch([1, 2, 3, 4, 5])
