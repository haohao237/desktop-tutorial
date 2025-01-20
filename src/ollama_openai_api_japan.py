from openai import OpenAI
import pandas as pd

# Ollama APIクライアントの設定
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama',  # required, but unused
)

# 質問と回答データの読み込み
questions_df = pd.read_csv('../data/questions.csv')
answers_df = pd.read_csv('../data/answers.csv')

def generate_feedback_by_tag(tag):
    # タグに関連付けられた問題の取得
    tagged_questions_df = questions_df[questions_df['tag'] == tag]
    tagged_answers_df = answers_df[answers_df['tag'] == tag]

    correct_count = tagged_answers_df['is_correct'].sum()
    total_count = tagged_answers_df.shape[0]
    avg_response_time = tagged_answers_df['time_taken'].mean()

    feedback = f"{tag}の正答率は{correct_count}/{total_count}で、平均解答時間は{avg_response_time:.2f}秒です。復習を行い、理解を深めましょう。\n"

    # フィードバックをOLLAMAに生成させる
    response = client.chat.completions.create(
        model="ELYZA",
        messages=[
            {"role": "system", "content": "あなたはAI検定の指導者で、受験者に役立つ日本語のフィードバックを提供します。フィードバックは簡潔で正確にし、不要な英語や冗長な情報は含めないでください。"},
            {"role": "user", "content": feedback}
        ]
    )

    # フィードバックの出力
    ollama_feedback = response.choices[0].message.content.strip()  # 不要な空白を削除
    print(f"### {tag} ###")
    print(ollama_feedback)
    print("\n")

# 各タグごとの全体フィードバックを生成
generate_feedback_by_tag('深層学習')
generate_feedback_by_tag('法規・倫理')
generate_feedback_by_tag('基礎数学')
generate_feedback_by_tag('AI概論')
generate_feedback_by_tag('機械学習')
