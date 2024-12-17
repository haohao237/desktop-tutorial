from transformers import GPT2Tokenizer

class FeedbackTokenizer:
    def __init__(self, model_name='gpt2'):
        # トークナイザーのロード
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    def encode(self, text):
        # テキストをトークンにエンコードする
        return self.tokenizer.encode(text, return_tensors='pt')

    def decode(self, tokens):
        # トークンをテキストにデコードする
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
