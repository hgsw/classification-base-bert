from transformers import BertTokenizerFast, BertForSequenceClassification
import torch

model_path = './results/checkpoint-1500'

tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese') 
model = BertForSequenceClassification.from_pretrained(model_path)

# 新的输入文本
new_texts  = ["体验2D巅峰倚天屠龙记十大创新概览", "同步A股首秀：港股缩量回调"]

# 编码输入文本
# truncation文本超过了tokenizer允许的最大序列长度（如模型的max_position_embeddings）截断
encoded_texts = tokenizer(new_texts, padding=True, truncation=True, return_tensors='pt')


# 使用模型进行推理
with torch.no_grad():
    outputs = model(**encoded_texts)

print(outputs)
# 获取模型的预测
logits = outputs.logits
print(logits)

# 如果模型是多分类任务，取概率最高的类别
if model.config.num_labels > 2:
    predictions = torch.argmax(logits, dim=1).tolist()
else:
    # 对于二分类任务，取大于0.5的概率作为正类
    predictions = (logits > 0.5).squeeze().tolist()

print(predictions)
# predictions = torch.argmax(logits, dim=-1) # 假设是分类任务，取最大概率的类别
# print(predictions)
