import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from transformers import BertTokenizerFast, BertForSequenceClassification
import torch
import seaborn as sns
from torch.utils.data import DataLoader
from datasets import Dataset
import matplotlib.pyplot as plt
import model_config
from peft import get_peft_model, PeftModel, LoraConfig, TaskType

# 加载测试集
test_data_path = model_config.test_data_path
test_data = pd.read_csv(test_data_path) 
# 简单测试请指定测试集数量
# test_data = pd.read_csv(test_data_path, nrows=32)
texts = test_data["0"].tolist()
labels = test_data["1"].tolist()
# 原始模型
model_path = model_config.model_name_tokenizer_path
model_path_loar = model_config.model_path_lora

tokenizer = BertTokenizerFast.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=10)

config = LoraConfig(
    task_type=TaskType.SEQ_CLS, 
    target_modules=["query", "key", "value"],
    inference_mode=True,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)
# 加载lora权重
model = PeftModel.from_pretrained(model, model_id=model_path_loar, config=config)

# 定义数据具体处理逻辑
def collate_fn(batch):
    texts = [item["text"] for item in batch]
    labels = [item["label"] for item in batch]
    encoding = tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
    encoding["labels"] = torch.tensor(labels)
    return encoding


batch_size = model_config.test_batch_size
# 创建Dataset对象
dataset = Dataset.from_dict({"text": texts, "label": labels})
data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

predictions = []
for batch in data_loader:
    inputs = {k: v for k, v in batch.items() if k != "labels"}
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    if model.config.num_labels > 2:
        # 多分类任务，取概率最高的类别
        batch_predictions = torch.argmax(logits, dim=1).tolist()
    else:
        # 二分类任务，取大于0.5的概率作为正类
        batch_predictions = (logits > 0.5).squeeze().tolist()

    predictions.extend(batch_predictions)

# 计算准确度、精确度和召回率
accuracy = accuracy_score(labels, predictions)
precision = precision_score(labels, predictions, average="weighted", zero_division=0)
recall = recall_score(labels, predictions, average="weighted", zero_division=0)

# 输出结果
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# 绘制混淆矩阵
cm = confusion_matrix(labels, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.savefig("img/confusion_matrix_lora.png")
plt.show()
