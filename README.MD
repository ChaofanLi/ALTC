# 阿里天池竞赛

## 1. 新闻文本分类
### a. 赛题连接
  https://tianchi.aliyun.com/competition/entrance/531810/information
### b. 赛题数据
  赛题以新闻数据为赛题数据，数据集报名后可见并可下载。赛题数据为新闻文本，并按照字符级别进行匿名处理。整合划分出14个候选分类类别：财经、彩票、房产、股票、家居、教育、科技、社会、时尚、时政、体育、星座、游戏、娱乐的文本数据。赛题数据由以下几个部分构成：训练集20w条样本，测试集A包括5w条样本，测试集B包括5w条样本。为了预防选手人工标注测试集的情况，我们将比赛数据的文本按照字符级别进行了***匿名处理***。
### c. 测评标准
  评价标准为类别f1_score的均值，选手提交结果与实际测试集的类别进行对比，结果越大越好。  
  $F1=2*\frac{precision∗recall}{precision+recall}$  
### d. Algorithm 1 StackGatedCNN
  score:0.93512    
  rank: 693/14830  
###<train bert>
  (1). 空格拆分数字表示的字符，去重构建词表，添加特殊字符；  
  (2). 训练tokenizer;  
  (3). 训练bert.  

### e. Algorithm 2 bert+LSTM max_pooler  
  score:0.9609   
  rank: 77/14830  
### f. Algorithm 3 bert+LSTM avg_pooler
  score:0.9594  
  rank: 693/14830  
### g. Algorithm 4 bert only
  score:0.9570  
  rank: 693/14830  

