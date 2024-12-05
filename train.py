import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
from sklearn.model_selection import train_test_split

# データの読み込みと前処理
df = pd.read_csv('data/data.csv')

# データセットを分割
train_texts, val_texts, train_labels, val_labels = train_test_split(df['question'], df['label'], test_size=0.2)

# トークナイザーの準備
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# データセットクラスの作成
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# データローダーの作成
train_dataset = CustomDataset(train_texts.tolist(), train_labels.tolist(), tokenizer)
val_dataset = CustomDataset(val_texts.tolist(), val_labels.tolist(), tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8)

# モデルの定義
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)

# トレーニング設定
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# モデル訓練
trainer.train()

# 訓練したモデルの保存
model.save_pretrained('./models')
tokenizer.save_pretrained('./models')
