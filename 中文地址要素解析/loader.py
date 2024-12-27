import torch
from torch.utils.data import Dataset
import re


def load_file(file_path):
    sentences = []
    labels = []
    pua_pattern = re.compile("[\uE000-\uF8FF]|[\u200b\u200d\u200e]")
    with open(file_path, 'r', encoding='utf-8') as f:
        sentence = []
        label = []
        for line in f:
            line = line.strip()
            
            if len(line) == 0:
                if len(sentence) > 0:
                    sentences.append(sentence)
                    labels.append(label)
                sentence = []
                label = []
            else:
                parts = line.split()
                word = parts[0]
                tag = parts[1]
                word = re.sub(pua_pattern, "", word)
                if word:
                    sentence.append(word)
                    label.append(tag)
        if len(sentence) > 0:
            sentences.append(sentence)
            labels.append(label)
    return sentences, labels

def load_test_file(file_path):
    sentences = []
    labels = []
    pua_pattern = re.compile("[\uE000-\uF8FF]|[\u200b\u200d\u200e]")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            ids, words = line.strip().split('\001')
            words = re.sub(pua_pattern, '', words)
            label=['O' for x in range(0,len(words))]
            sentence=[]
            for c in words:
                sentence.append(c)
            sentences.append(sentence)
            labels.append(label)
    return sentences,labels

class DataGenerator(Dataset):
    def __init__(self, sentences, labels, tokenizer, tag2id):
        
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.tag2id = tag2id
        
        self.encodings  = tokenizer(sentences, is_split_into_words=True,padding=True)
        
        self.encoded_labels = []
        for label, input_id in zip(labels, self.encodings['input_ids']):
            t = len(input_id) - len(label)-1
            label = ['O'] + label + ['O']*t
            self.encoded_labels.append([tag2id[l] for l in label])

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        input_ids = torch.LongTensor(self.encodings['input_ids'][idx])
        attention_mask = torch.LongTensor(self.encodings['attention_mask'][idx])
        labels = torch.LongTensor(self.encoded_labels[idx])
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
    
    def get_item(self,idx):
        pass
    
