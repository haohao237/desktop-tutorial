from transformers import BertForSequenceClassification

def create_model():
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)
    return model
