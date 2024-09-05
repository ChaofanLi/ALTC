from transformers import BertTokenizer

import torch
from tqdm import tqdm
from transformers import RobertaConfig
from transformers import RobertaForMaskedLM
from transformers import AdamW

tokenizer = BertTokenizer.from_pretrained('bert_model',max_len=512)
# sentrnce="[CLS] 2786 3659 5724 5619 2396 5602 1324 414 6352 314 1152 2465 1744 4109 2218 1565 5330 5602 3370 3700 3700 2106 192 6045 7539 2465 5264 3917 2768 1302 5284 1779 2210 6040 2936 5330 5264 3750 1744 4109 1080 6122 6050 23 2109 5598 6890 3242 6407 4480 3750 3223 3976 936 7058 5330 648 3137 5497 2786 433 900 4822 2400 3750 1744 4109 6469 1066 4301 5226 281 3694 1018 1066"
# print(tokenizer.encode(sentrnce))

with open('dataset/all_train_data.txt', 'r', encoding='utf-8') as fp:
    lines = fp.read().split('\n')
    
batch = tokenizer.batch_encode_plus(lines,padding=True, truncation=True, return_tensors='pt')
print(batch["input_ids"].shape)

labels = batch["input_ids"]

mask = batch["attention_mask"]

# make copy of labels tensor, this will be input_ids
input_ids = labels.detach().clone()
# create random array of floats with equal dims to input_ids
rand = torch.rand(input_ids.shape)
# mask random 15% where token is not 0 [PAD], 1 [CLS], or 2 [SEP]
mask_arr = (rand < .15) * (input_ids != 0) * (input_ids != 1) * (input_ids != 2)
# loop through each row in input_ids tensor (cannot do in parallel)
for i in range(input_ids.shape[0]):
    # get indices of mask positions from mask array
    selection = torch.flatten(mask_arr[i].nonzero()).tolist()
    # mask input_ids
    input_ids[i, selection] = 3  # our custom [MASK] token == 3
    
encodings = {'input_ids': input_ids, 'attention_mask': mask, 'labels': labels}
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        # store encodings internally
        self.encodings = encodings

    def __len__(self):
        # return the number of samples
        return self.encodings['input_ids'].shape[0]

    def __getitem__(self, i):
        # return dictionary of input_ids, attention_mask, and labels for index i
        return {key: tensor[i] for key, tensor in self.encodings.items()}

dataset = Dataset(encodings)

loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
 
config = RobertaConfig(
    vocab_size=tokenizer.vocab_size,
    max_position_embeddings=514,
    hidden_size=768,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1
)

model = RobertaForMaskedLM(config)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# and move our model over to the selected device
model.to(device)

# activate training mode
model.train()
# initialize optimizer
optim = AdamW(model.parameters(), lr=1e-4)

epochs = 6

for epoch in range(epochs):
    # setup loop with TQDM and dataloader
    loop = tqdm(loader, leave=True)
    for batch in loop:
        # initialize calculated gradients (from prev step)
        optim.zero_grad()
        # pull all tensor batches required for training
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        # process
        outputs = model(input_ids, attention_mask=attention_mask,
                        labels=labels)
        # extract loss
        loss = outputs.loss
        # calculate loss for every parameter that needs grad update
        loss.backward()
        # update parameters
        optim.step()
        # print relevant info to progress bar
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())
        
model.save_pretrained('bert_model_new') 
