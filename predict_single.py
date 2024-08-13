from transformers import BertTokenizerFast, BertForSequenceClassification
import torch

model_path = "./results/checkpoint-11250"

tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese') 
model = BertForSequenceClassification.from_pretrained(model_path)

# 文本对应真实分类为3
new_texts  = ["应届生考研爱“抱团” 互相鼓励共同提高"]
encoded_texts = tokenizer(new_texts, padding=True, truncation=True, max_length=64, return_tensors='pt')


# 使用模型进行推理
with torch.no_grad():
    outputs = model(**encoded_texts)

# print(outputs)
# 获取模型的预测
logits = outputs.logits
print(logits)

predictions = torch.argmax(logits, dim=-1) 
print(predictions)
