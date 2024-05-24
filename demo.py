import pandas as pd
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
import torch


# 读取数据
df = pd.read_csv("./dev.csv")
texts = df["text"].tolist()
labels = df["labels"].tolist()

# 初始化tokenizer和模型
# model_name = "bert-base-uncased"
model_name = "bert-base-chinese"
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(set(labels)))

# 对文本进行编码
train_encodings = tokenizer(texts, truncation=True, padding=True)


# 封装数据为PyTorch Dataset
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item


    def __len__(self):
        return len(self.labels)


train_dataset = TextDataset(train_encodings, labels)


# 设置训练参数并创建Trainer
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

# 开始训练
trainer.train()