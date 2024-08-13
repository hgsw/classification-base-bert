from transformers import BertModel, BertTokenizer

# 指定模型名称
model_name = "bert-base-chinese"

# 下载模型
model = BertModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# 保存模型
save_directory = "./bert-base-chinese"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)