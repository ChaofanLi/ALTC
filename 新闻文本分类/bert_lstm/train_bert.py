import torch
from tqdm import tqdm
from transformers import RobertaTokenizer
from transformers import RobertaConfig
from transformers import RobertaForMaskedLM
from transformers import AdamW

tokenizer = RobertaTokenizer.from_pretrained('tokenizer',max_len=512)
with open('corpus.txt', 'r', encoding='utf-8') as fp:
    lines = fp.read().split('\n')
batch = tokenizer.batch_encode_plus(lines,padding=True, truncation=True, return_tensors='pt')

labels = batch.input_ids
mask = batch.attention_mask
input_ids = labels.detach().clone()
rand = torch.rand(input_ids.shape)
mask_arr = (rand < .15) * (input_ids != 0) * (input_ids != 1) * (input_ids != 2)
for i in range(input_ids.shape[0]):
    selection = torch.flatten(mask_arr[i].nonzero()).tolist()
    input_ids[i, selection] = 3

encodings = {'input_ids': input_ids, 'attention_mask': mask, 'labels': labels}
class Dataset(torch.utils.data.Dataset):
 def __init__(self, encodings):
     self.encodings = encodings

 def __len__(self):
     return self.encodings['input_ids'].shape[0]

 def __getitem__(self, i):
     return {key: tensor[i] for key, tensor in self.encodings.items()}

dataset = Dataset(encodings)
loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

config = RobertaConfig(
     vocab_size=30_522,  # we align this to the tokenizer vocab_size
     max_position_embeddings=514,
     hidden_size=768,
     num_attention_heads=12,
     num_hidden_layers=6,
     type_vocab_size=1
 )

model = RobertaForMaskedLM(config)

torch.cuda.empty_cache()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.train()
optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

epochs = 3
for epoch in range(epochs):
    loop = tqdm(loader, leave=True)
    for batch in loop:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask,
                     labels=labels)
        loss = outputs.loss
        loss.backward()
        optim.step()
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())
    model.save_pretrained(f'bert_model/epoch_{epoch}')
    torch.cuda.empty_cache()
