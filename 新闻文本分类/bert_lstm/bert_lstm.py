import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim import Adam, SGD
from transformers import BertModel
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer,RobertaForMaskedLM
from transformers import BertTokenizer

import warnings
warnings.filterwarnings("ignore")

tokenizer = BertTokenizer.from_pretrained('train_bert_2/bert_model',max_len=512)

config={"max_length":512,
        "batch_size":64,
        "class_num":14,
        "num_layers":6,
        "hidden_size":768,
        "pooling_style":"avg",
        "train_data_path":"datasets/new_data.csv",
        "valid_data_path":"datasets/valid_data.csv",
        "test_data_path":"downloads/test_a.csv",
        "bert_model_path":"train_bert_2/bert_model_new",
        "model_path":"models/bert_lstm/bert_lstm_plravg_bz64.pth",
        "optimizer":"adam",
        "learning_rate":0.00001,
        "epoch":5,
        "kernel_size": 3,
        "tokenizer":tokenizer,
        "vocab_size":tokenizer.vocab_size,
       }

class DataGenerator:
    def __init__(self, data_path, config,typ):
        self.tokenizer = config["tokenizer"]
        self.data_path = data_path
        self.config = config
        self.typ=typ
        self.load()

    def load(self):
        self.data = []
        df = pd.read_csv(self.data_path)
        for i in range(len(df)):
            sentence=df["text"].iloc[i]
            sequence=self.tokenizer(sentence)["input_ids"]
            if self.typ=="train":
                encode_x = torch.LongTensor(padding(sequence,self.config))
                encode_y= torch.LongTensor([df["label"].iloc[i]])
                self.data.append([encode_x,encode_y])
            else:
                encode_x = torch.LongTensor(padding(sequence,self.config))
                encode_y = torch.LongTensor([0])
                self.data.append([encode_x,encode_y])
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]



def padding(input_sequence, config):
    input_sequence = input_sequence[:config["max_length"]]
    input_sequence += [0] * (config["max_length"] - len(input_sequence))
    return input_sequence


def load_data(data_path, config, shuffle=True,typ="train"):
    dg = DataGenerator(data_path, config,typ)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl

# class TorchModel(nn.Module):
#     def __init__(self, config):
#         super(TorchModel, self).__init__()
#         class_num = config["class_num"]
#         self.encoder= RobertaForMaskedLM.from_pretrained(config["bert_model_path"],return_dict=False)
#         hidden_size = self.encoder.config.hidden_size
#         self.classify = nn.Linear(hidden_size, class_num)
#         self.pooling_style = config["pooling_style"]
#         self.loss = nn.functional.cross_entropy

#     def forward(self, x, target=None):
#         x = self.encoder(x,output_hidden_states=True)
#         x=x[1][-1][:, 0, :]
#         predict = self.classify(x)

#         if target is not None:
#             return self.loss(predict, target.squeeze())
#         else:
#             return predict

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        class_num = config["class_num"]
        self.encoder = BertLSTM()
        hidden_size = self.encoder.bert.config.hidden_size
        self.classify = nn.Linear(hidden_size, class_num)
        self.pooling_style = config["pooling_style"]
        self.loss = nn.functional.cross_entropy

    def forward(self, x, target=None):
        x = self.encoder(x)
        if isinstance(x, tuple): 
            x = x[0]
            
        if self.pooling_style == "max":
            self.pooling_layer = nn.MaxPool1d(x.shape[1])
        else:
            self.pooling_layer = nn.AvgPool1d(x.shape[1])
            
        x = self.pooling_layer(x.transpose(1, 2)).squeeze()

        predict = self.classify(x)

        if target is not None:
            return self.loss(predict, target.squeeze())
        else:
            return predict

class BertLSTM(nn.Module):
    def __init__(self):
        super(BertLSTM, self).__init__()
        self.bert = RobertaForMaskedLM.from_pretrained(config["bert_model_path"],return_dict=False)
        self.rnn = nn.LSTM(self.bert.config.hidden_size, self.bert.config.hidden_size, batch_first=True)

    def forward(self, x):
        bert_output = self.bert(x,output_hidden_states=True)
        hidden_states = bert_output[1] # 获取最后一个隐藏层的输出
        last_hidden_state = hidden_states[-1]
        cls_output = last_hidden_state[:, 0, :]  # 获取CLS token的输出
        cls_output = cls_output.unsqueeze(1)  # 添加一个维度以适应LSTM的输入要求
        lstm_output, _ = self.rnn(cls_output)
        return lstm_output


    
def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


class Evaluator:
    def __init__(self,config,model):
        self.config=config
        self.model=model
        self.valid_data=load_data(self.config["valid_data_path"],config)
        self.state_dic={"correct":0,"wrong":0}

    def eval(self,epoch):
        print("开始测试第%d轮模型效果：" % epoch)
        self.model.eval()
        self.state_dic={"correct":0,"wrong":0}
        for index,batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data=[b.cuda() for b in batch_data]
            input_ids,labels=batch_data
            with torch.no_grad():
                pred_result=self.model(input_ids)
            self.write_stats(labels,pred_result)
        acc=self.show_stats()
        return acc

    def write_stats(self,labels,pred_result):
        assert  len(labels)==len(pred_result)
        for true_label,pred_label in zip(labels,pred_result):
            pred_label=torch.argmax(pred_label)
            if int(true_label)==int(pred_label):
                self.state_dic["correct"]+=1
            else:
                self.state_dic["wrong"]+=1
        return

    def show_stats(self):
        correct=self.state_dic["correct"]
        wrong=self.state_dic["wrong"]
        print("预测集合条目总量：%d" % (correct + wrong))
        print("预测正确条目：%d，预测错误条目：%d" % (correct, wrong))
        print("预测准确率：%f" % (correct / (correct + wrong)))
        print("--------------------")
        return correct / (correct + wrong)


def main(config):
    # 加载训练数据
    train_data = load_data(config["train_data_path"], config)
    # 加载模型
    model = TorchModel(config)
    # 判断GPU是否可用
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        print("设备GPU可用，迁移模型至GPU")
        model = model.cuda()
    # 加载优化器
    optimizer = choose_optimizer(config, model)
    # 加载模型训练效果
    evaluator = Evaluator(config, model)
    # 训练
    for epoch in range(config["epoch"]):
        model.train()
        print("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in tqdm(enumerate(train_data)):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            optimizer.zero_grad()
            input_ids, labels = batch_data
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if index % 200 == 0:
                print("batch loss %f" % loss)
        print("epoch average loss: %f" % np.mean(train_loss))
        acc = evaluator.eval(epoch)
        # torch.cuda.empty_cache()
    torch.save(model.state_dict(), config["model_path"])
    return acc

class Predictor():
    def __init__(self,config,model):
        self.config=config
        self.model=model
        self.valid_data=load_data(self.config["test_data_path"],config,shuffle=False,typ="test")

    def predict(self):
        self.model.eval()
        pred=[]
        for index,batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data=[b.cuda() for b in batch_data]
            input_ids,labels = batch_data
            with torch.no_grad():
                pred_result=self.model(input_ids)
            for labels,pred_label in zip(labels,pred_result):
                pred_label=torch.argmax(pred_label)
                pred.append(int(pred_label))
        submit_data = pd.read_csv("downloads/test_a_sample_submit.csv")
        submit_data["label"]=pred
        submit_data.to_csv("downloads/test_a_sample_submit.csv")

        
    
    
    

if __name__=="__main__":
    # # 训练
    # acc=main(config)
    # # 预测
    model=TorchModel(config)
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        print("设备GPU可用，迁移模型至GPU")
        model = model.cuda()
    model.load_state_dict(torch.load(config["model_path"]))
    predictor=Predictor(config,model)
    predictor.predict()
