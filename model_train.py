import pandas as pd
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

train_data_path = "THUCNews/data/train.csv"
dev_data_path = "THUCNews/data/dev.csv"

train_data = pd.read_csv(train_data_path)
train_texts = train_data["0"].tolist()
train_labels = train_data["1"].tolist()

dev_data = pd.read_csv(dev_data_path)
eval_texts = dev_data["0"].tolist()
eval_labels = dev_data["1"].tolist()

# 分类标签数
num_labels=len(set(train_labels))

# 预训练模型
model_name = "./bert-base-chinese"
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# 对文本进行编码
# 入参详解 请参考 https://blog.csdn.net/weixin_42924890/article/details/139269528
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=64)
eval_encodings = tokenizer(eval_texts, truncation=True, padding=True, max_length=64)


class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # print(idx)
        # for key, val in self.encodings.items():
        #     print(key)
        #     print(val)
        #     print(val[idx])
            
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item


    def __len__(self):
        return len(self.labels)


train_dataset = TextDataset(train_encodings, train_labels)
eval_dataset = TextDataset(eval_encodings, eval_labels)


# 核心 设置训练参数并创建Trainer
training_args = TrainingArguments(
    output_dir='./results', # 模型保存路径
    logging_dir="./logs",  # 日志保存路径
    save_strategy="steps", # 保存策略，按steps保存
    save_total_limit=1, # 保存模型的最大数量
    evaluation_strategy="steps",
    save_steps=250, # 每250个step保存一次
    eval_steps=125, # 每125个step评估一次
    load_best_model_at_end=True, # 训练结束后加载在评估过程中表现最好的模型
    num_train_epochs=5, # 训练轮数，epoch数
    per_device_train_batch_size=32, # 训练时每个设备上的batch大小
    per_device_eval_batch_size=32, # 评估时每个设备上的batch大小
    warmup_steps=1250, # 预热步数，用于学习率warmup
    weight_decay=0.001, # 权重衰减，防止过拟合
    dataloader_drop_last=True,  # 是否丢弃最后一个不完整的batch
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# 开始训练
trainer.train()