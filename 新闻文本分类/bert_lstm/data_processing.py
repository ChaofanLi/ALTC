import os
import pandas as pd

# train_data_path="../downloads/train_set.csv"
# test_data_path="../downloads/test_a.csv"

# train_data=pd.read_csv(train_data_path, sep='\t', encoding='UTF-8')
# test_data=pd.read_csv(test_data_path, sep='\t', encoding='UTF-8')

# with open('dataset/train_data.txt', 'w', encoding='utf-8') as file:
#     for line in train_data['text']:
#         file.write(line + '\n') 

# with open('dataset/test_data.txt', 'w', encoding='utf-8') as file:
#     for line in test_data['text']:
#         file.write(line + '\n') 
        
# 合并 train.txt 和 test.txt 到 all_data.txt
# with open('dataset/train_data.txt', 'r', encoding='utf-8') as train_file, \
#      open('dataset/test_data.txt', 'r', encoding='utf-8') as test_file, \
#      open('dataset/all_data.txt', 'w', encoding='utf-8') as all_data_file:
    
#     # 读取并写入 train.txt 内容
#     for line in train_file:
#         all_data_file.write(line)
    
#     # 读取并写入 test.txt 内容
#     for line in test_file:
#         all_data_file.write(line)

with open('dataset/all_data.txt', 'r', encoding='utf-8') as all_data,\
    open('dataset/all_train_data.txt', 'w', encoding='utf-8') as all_train_data:
    for line in all_data:
        word_ls=line.strip().split()
        word_num=len(word_ls)//512
        for num in range(word_num+1):
            sentence=" ".join(word_ls[512*num:512*(num+1)])
            all_train_data.write(sentence + '\n')
