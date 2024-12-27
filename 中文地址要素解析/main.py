import torch
from model import TokenClassificationWithCRF
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import evaluate
from config import Config
from loader import load_file
from loader import DataGenerator
import warnings
warnings.filterwarnings("ignore")

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

train_dataset = DataGenerator(train_sentences, train_labels, tokenizer, tag2id)
eval_dataset = DataGenerator(dev_sentences, dev_labels, tokenizer, tag2id)

model=TokenClassificationWithCRF(Config["bert_path"], num_labels=len(tag2id))
train_dataloader = DataLoader(train_dataset, batch_size=Config["batch_size"], shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=Config["batch_size"])
optimizer = AdamW(model.parameters(), lr=Config["learning_rate"])
num_training_steps = Config["epoch"] * len(train_dataloader)
lr_scheduler = get_scheduler(name=Config["lr_scheduler_mod"], optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(Config["epoch"]):
    for step, batch in enumerate(train_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        if step % 100 == 0:
            print(f'Step {step} - Training loss: {loss}')

torch.save(model.state_dict(), Config["model_path_all"])

metric  = evaluate.load('seqeval')

model.eval()
for batch in eval_dataloader:
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
        
        metric.add(prediction=prediction2, reference=label2)
            
results = metric.compute()
print({
    "precision": results['overall_precision'],
    "recall": results['overall_recall'],
    "f1": results['overall_f1'],
    "accuracy": results['overall_accuracy']
})