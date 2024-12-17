from transformers import GPT2Tokenizer

# トークナイザーのロード
def load_tokenizer():
    return GPT2Tokenizer.from_pretrained("gpt2")

if __name__ == "__main__":
    tokenizer = load_tokenizer()
    print("トークナイザーロード完了！")
