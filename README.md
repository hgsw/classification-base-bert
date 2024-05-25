# classification-base-bert

由于模型文件过大，上传比较麻烦，保留了bert-base-chinese目录，但是bert-base-chinese的基础模型没有上传，先需要到Hugging Face官网下载[下载地址](https://huggingface.co/google-bert/bert-base-chinese/tree/main)，仅下载pytorch_model.bin模型文件即可，将下载好的文件放到项目的bert-base-chinese目录即可。

基础模型下载好后，直接运行或调试demo.py文件即可，训练完成后会有results/目录这里是微调后的新模型文件。

新训练后的模型可以用predict.py文件执行。