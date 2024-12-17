from transformers import GPT2LMHeadModel
import torch

# モデル定義
def load_model():
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()  # 推論モードに設定
    if torch.cuda.is_available():
        model.cuda()  # GPU利用
    return model

# モデルを利用してテキスト生成
def generate_text(prompt, model, tokenizer):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()

    outputs = model.generate(input_ids, max_length=50, num_return_sequences=1, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    print("モデルロードテスト")
    model = load_model()
