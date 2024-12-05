from transformers import T5Tokenizer

def get_tokenizer():
    return T5Tokenizer.from_pretrained('t5-small')
