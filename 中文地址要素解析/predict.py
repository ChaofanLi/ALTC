import torch
from torch.utils.data import DataLoader
from loader import load_file, load_test_file,DataGenerator
from transformers import AutoTokenizer
from config import Config
from model import TokenClassificationWithCRF

tokenizer = AutoTokenizer.from_pretrained(Config["tokenizer_path"])

train_sentences, train_labels = load_file(Config["train_data_path_all"])
dev_sentences,dev_labels = load_file(Config["valid_data_path"])

tags_list = ['O']
for labels in (train_labels + dev_labels):
    for tag in labels:
        if tag not in tags_list:
            tags_list.append(tag)

tag2id = {tag: i for i, tag in enumerate(tags_list)}
id2tag = {i: tag for i, tag in enumerate(tags_list)}

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model=TokenClassificationWithCRF(Config["bert_path"], num_labels=len(tag2id))
model.load_state_dict(torch.load(Config["model_path_all"]))
model.to(device)
model.eval()

test_sentences,test_labels = load_test_file(Config["test_data_path"])
test_dataset = DataGenerator(test_sentences, test_labels, tokenizer, tag2id)
test_dataloader = DataLoader(test_dataset, batch_size=Config["batch_size"])

file_name = "output.txt"

with open(file_name, "w",encoding="utf-8") as file:

    i = 1
    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

        predictions = outputs["logits"]

        for input_id, prediction, label, attention_mask in zip(batch['input_ids'], predictions, batch['labels'], batch['attention_mask']):

            input_id = input_id.tolist()
            attention_mask = attention_mask.tolist()
            
            valid_indices = [i for i, mask in enumerate(attention_mask) if mask == 1]
            
            prediction2 = [id2tag[t] for i, t in enumerate(prediction) if i in valid_indices]
            label2 = [id2tag[t.item()] for i, t in enumerate(label) if i in valid_indices]
            
            sentence = tokenizer.decode(input_id,skip_special_tokens=True).replace(" ","")
            
            prediction_str = ' '.join(prediction2[1:-1])
            line = f"{i}\u0001{sentence}\u0001{prediction_str}\n"
            file.write(line)
            i += 1
