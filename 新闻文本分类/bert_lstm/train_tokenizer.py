from transformers import BertTokenizer
from datasets import Dataset

# 假设你的数据在一个文本文件中，每一行都是隐匿后的字符序列
file_path = "dataset/all_data.txt"

# 读取数据并创建 Dataset
def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    return Dataset.from_dict({"text": [line.strip() for line in lines]})

dataset = load_data(file_path)

# Step 1: 创建自定义词典
vocab = set()
for example in dataset["text"]:
    tokens = example.split(" ")
    vocab.update(tokens)

# 将词典写入到文件中（BERT tokenizer 需要一个词典文件）
with open("vocab.txt", "w") as vocab_file:
    for token in ["[PAD]","[CLS]","[SEP]","[MASK]"]:
        vocab_file.write(f"{token}\n")
    for token in sorted(vocab):
        vocab_file.write(f"{token}\n")

# Step 2: 使用自定义词典创建 tokenizer
tokenizer = BertTokenizer(vocab_file="vocab.txt", do_lower_case=False,model_max_length=512)
tokenizer.save_pretrained("./bert_model")
