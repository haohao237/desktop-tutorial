from flask import Flask, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd
import torch

app = Flask(__name__)

# モデルとトークナイザーのロード
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# GPUが利用可能ならば設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


@app.route("/feedback", methods=["GET"])
def generate_feedback():
    """APIエンドポイント: フィードバック生成"""
    try:
        # ユーザーの回答結果をロード
        results_file = "../results/results.csv"
        results = pd.read_csv(results_file)

        feedback_list = []

        # 結果をループしてフィードバックを生成
        for _, row in results.iterrows():
            # プロンプトの修正: 新しい列に基づいてフィードバックを生成
            prompt = (
                f"ユーザーID: {row['user_id']}が解答した質問ID: {row['question_id']}（タグ: {row['tag']}）. "
                f"回答結果: {'正解' if row['is_correct'] else '不正解'}. "
                f"解答時間: {row['time_taken']}秒. "
                "この情報を元に改善点を教えてください。"
            )

            # モデルへの入力を準備
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            # GPT-2でフィードバック生成
            outputs = model.generate(
                inputs["input_ids"], max_length=100, num_return_sequences=1, temperature=0.7
            )
            suggestion = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # フィードバックリストに追加
            feedback_list.append({
                "user_id": row["user_id"],
                "question_id": row["question_id"],
                "tag": row["tag"],
                "feedback": suggestion
            })

        return jsonify(feedback_list)

    except FileNotFoundError:
        return jsonify({"error": "results.csvが見つかりません。"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
