import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class FeedbackModel:
    def __init__(self, model_name='gpt2'):
        # GPT-2を使用してフィードバックを生成するモデル
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    def generate_feedback(self, input_text):
        # 入力テキストに基づいてフィードバックを生成する
        inputs = self.tokenizer.encode(input_text, return_tensors='pt')
        outputs = self.model.generate(inputs, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2)

        feedback = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return feedback

    def calculate_feedback(self, question, time_taken, correct_rate, tag):
        # ユーザーの回答時間、正答率、タグに基づいてフィードバックを生成
        input_text = f"ユーザーの回答情報: 問題: {question}, 正答率: {correct_rate}, 回答時間: {time_taken}秒, タグ: {tag}. フィードバック: "
        feedback = self.generate_feedback(input_text)
        return feedback
