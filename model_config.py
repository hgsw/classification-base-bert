# model_config.py
""" 通用配置文件，请仔细阅读，各文件路径确保存在且正确
"""

# 训练集地址，按需修改
train_data_path = "THUCNews/data/train.csv"

# 验证集地址，按需修改
dev_data_path = "THUCNews/data/dev.csv"

# 训练后的模型文件地址，按需修改
model_path = "./results/checkpoint-11250"

# 原始模型以及分词器地址，按需修改
model_name_tokenizer_path = "/data/data/transformer/classification_demo/bert-base-chinese"

# 加载测试集，按需修改
test_data_path = "THUCNews/data/test.csv"

# 批量测试时的batch_size
test_batch_size = 16

# LoRA微调后的模型文件地址，按需修改
model_path_lora = "./lora_results/checkpoint-11250"
