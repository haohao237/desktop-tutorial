from transformers import T5ForConditionalGeneration

def create_model():
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    return model
