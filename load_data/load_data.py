#!/usr/bin/env python
# -*- coding: utf-8 -*-


from numpy.lib.function_base import average
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
import jieba
import os, sys
sys.path.append(os.path.dirname(__file__))
import json
import thulac
import multiprocessing
import time 
import jieba
import torch
from tqdm import tqdm
from sklearn.svm import LinearSVC
import joblib
from  torch.utils.data import Dataset
from transformers import BertTokenizer,RobertaTokenizer
from  matplotlib import pyplot as plt
import matplotlib

# jieba.initialize()
cutmodel = thulac.thulac(seg_only=True)

print("*********************load_data.py******************************")

def cut_text(query):
    """分词"""
    return cutmodel.cut(query, text=True)

def query_cut(query):
    '''
    @description: word segment
    @param {type} query: input data
    @return:
    list of cut word
    '''
    return " ".join(jieba.lcut(query))


def init():
    f = open('files/law.txt', 'r', encoding='utf8')
    law = {}
    lawname = {}
    line = f.readline()
    while line:
        lawname[len(law)] = line.strip()
        law[line.strip()] = len(law)
        line = f.readline()
    f.close()

    f = open('files/accu.txt', 'r', encoding='utf8')
    accu = {}
    accuname = {}
    line = f.readline()
    while line:
        accuname[len(accu)] = line.strip()
        accu[line.strip()] = len(accu)
        line = f.readline()
    f.close()
    return law, accu, lawname, accuname


law, accu, lawname, accuname = init()


def gettime(time) ->int :
    # 将刑期用分类模型来做
    v = int(time['imprisonment'])

    if time['death_penalty']:
        return 0
    if time['life_imprisonment']:
        return 1
    elif v > 10 * 12:
        return 2
    elif v > 7 * 12:
        return 3
    elif v > 5 * 12:
        return 4
    elif v > 3 * 12:
        return 5
    elif v > 2 * 12:
        return 6
    elif v > 1 * 12:
        return 7
    elif v > 9:
        return 8
    elif v > 6:
        return 9
    elif v > 3:
        return 10
    else:
        return 11


def getlabel(d, kind="time"):
    global law
    global accu

    # 做单标签
    if kind == 'law':
        # 返回多个类的第一个
        return law[str(d['meta']['relevant_articles'][0])]
    if kind == 'accu':
        return accu[d['meta']['accusation'][0].replace("[", '').replace(']', '')]

    if kind == 'time':
        return gettime(d['meta']['term_of_imprisonment'])

    return kind



def processes_data(alldata,kind ="time", word=True):
    """预处理数据
       word ：bool 是否分词
       kind : label 类型也就是任务类型
      """
    X,label =[],[]
    for data in tqdm(alldata):
        X.append(data["fact"])
        label.append(getlabel(data, kind))
    return (X ,label)


def multiprocess_data(data, worker=4, word=True,kind="time"):
    cpu_count = multiprocessing.cpu_count() // 2
    # multiprocessing.set_start_method('spawn')
    if worker == -1 or worker > cpu_count:
        worker = cpu_count
    data_length = len(data)
    chunk_size = data_length // worker
    start, end = 0, 0

    pool = multiprocessing.Pool(processes=worker,)
    result = []
    T1 = time.time()
    while end < data_length:
        end = start + chunk_size
        if end > data_length:
            end = data_length
        res = pool.apply_async(processes_data, (data[start:end],kind, word))
        start = end
        result.append(res)
    pool.close()
    pool.join()
    X,Y=[],[]
    for i in result:
        x,y = i.get()
        X.extend(x)
        Y.extend(y)
    end_time = time.time()
    print('_______multiprocess time: %.4f seconds.______' % (end_time - T1))
    return X,Y


def read_Data(path, worker=4, word=True,kind="time"):
    with open(path, 'r', encoding='utf8') as fp:
        alltext = []
        lines = fp.readlines()
        for line in lines:
            d = json.loads(line)
            if word:
                d["fact"]=cut_text(d["fact"])
            alltext.append(d)
        
        X,Y = multiprocess_data(alltext, worker=worker, word=word,kind=kind)
        # sample_weight=[Y.count(i)  for i in set(Y)]
    return X,Y


def show_label(path,kind="time"):
    with open(path, 'r', encoding='utf8') as fp:
        labels=[]
        length_fact=[]
        lines = fp.readlines()
        for line in tqdm(lines):
            d = json.loads(line)
            labels.append(getlabel(d, kind))
            length_fact.append(len(d['fact']))
            # if d['meta']['term_of_imprisonment']['life_imprisonment']:
            #     labels.append(-1)
            # elif d['meta']['term_of_imprisonment']['death_penalty'] :
            #     labels.append(-2)
            # else:
            #     labels.append(d['meta']['term_of_imprisonment']["imprisonment"])

    label_num=[(i,labels.count(i))  for i in set(labels)]
    sample_weight=[labels.count(i)  for i in set(labels)]
    print(max(length_fact))
    print(min(length_fact))
    print( len([i for i in length_fact if i>512]))
    print( len([i for i in length_fact if i>1024]))
    print( len([i for i in length_fact if i>2048]))
    print( len([i for i in length_fact if i>4096]))
    # matplotlib.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
    matplotlib.rcParams['axes.unicode_minus']=False     # 正常显示负号  
    # matplotlib.ticker.set_major_formatter(False)
    plt.hist(length_fact,bins=10, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.xlabel("mouths")
    # 显示纵轴标签
    plt.ylabel("numbers")
    # 显示图标题
    plt.title("")
    plt.savefig("files/label_fact.png")
    # plt.show()


def build_dic(path,num_words=50000):
    """构建词频字典"""
    from  collections import OrderedDict
    with open(path, 'r', encoding='utf8') as fp:
        my_dict = OrderedDict()
        lines = fp.readlines()
        for line in tqdm(lines):
            d = json.loads(line)
            seq = cut_text(d['fact'])
            for word in seq.split():
                if word  not in my_dict:
                    my_dict[word] = 1
                else:
                    my_dict[word] += 1
    my_dict = OrderedDict(sorted(my_dict.items(),key = lambda t:t[1],reverse=True))
    with open("files/vocal.txt" ,"w",encoding='utf8') as fp:
        for word in my_dict:
            fp.write(word)
            fp.write('\n')

def deal_samples(path,save_path,kind="time"):
    """将数据集分词后保存加快训练速度"""
    from  collections import OrderedDict
    fp2 = open(save_path ,"w",encoding='utf8')
    with open(path, 'r', encoding='utf8') as fp:
        # my_dict = OrderedDict()
        lines = fp.readlines()
        for line in tqdm(lines):
            d = json.loads(line)
            seq = cut_text(d['fact'])
            d['fact'] = seq
            d=json.dumps(d,ensure_ascii=False)
            fp2.write(d)
            fp2.write('\n')
    fp2.close()

def build_tokenizer(files,num_words=50000):
    """构建tokenizer"""
    tokenizer = {}
    with open(files ,"r",encoding='utf8') as fp:
        words = json.load(fp)
        for i,word in enumerate(words[:num_words]):
            tokenizer[word.strip()] = i+1
            
    return tokenizer



class Law_dataset(Dataset):
    def __init__(self,config,prefix) -> None:
        super().__init__()
        file_path = config.data_dir+ prefix +".json"
        self.X,self.Y  = read_Data(file_path,config.workers,config.word,config.kind)
        # self.tfidf = joblib.load('premodel/tfidf.model')
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_path)
        self.tokenizer.model_max_length =  512
        self.tf_tokenizer = build_tokenizer("files/vocab.json")
        self.max_length = config.max_length
        self.config = config

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sentence, labels = self.X[idx],self.Y[idx]
        text_dict={}
        if self.config.model_name != "bert_model":
            text_dict["input_ids"] = self.mytokenizer(sentence)
        else:
            text_dict = self.tokenizer.encode_plus(
                sentence,
                max_length=self.max_length,
                return_token_type_ids=True,
                return_attention_mask=True,
                truncation=True
                )
        text_dict['labels'] = labels
        return text_dict
    
    def mytokenizer(self,sentence):
        res = []
        for word in sentence.split():
            res.append(self.tf_tokenizer.get(word,0))
        return   res
        


def collate_to_max_length(batch):    
    """
    动态padding,返回Tensor
    :param batch:
    :return: 每个batch id和label
    """
    def padding(indice, max_length, pad_idx=0):
        """
        填充每个batch的句子长度
        """
        pad_indice =[]
        for item in indice:
            if len(item) < max_length:
                pad_indice.append(item + [pad_idx] * max(0, max_length - len(item)))
            else:
                pad_indice.append(item[:max_length])
       
        return torch.tensor(pad_indice)

    token_ids = [data["input_ids"] for data in batch]
    max_length = min(max([len(t) for t in token_ids]),512)# batch中样本的最大的长度

    labels = torch.tensor([data["labels"] for data in batch])
    token_type_ids = [data["token_type_ids"] for data in batch]
    attention_mask = [data["attention_mask"] for data in batch]
    # 填充每个batch的sample
    token_ids_padded = padding(token_ids, max_length)
    token_type_ids_padded = padding(token_type_ids, max_length)
    attention_mask_padded = padding(attention_mask, max_length)

    return token_ids_padded, attention_mask_padded, token_type_ids_padded, labels


def collate_to_max_length_(batch,maxlength=512):    
    """
    动态padding,返回Tensor
    :param batch:
    :return: 每个batch id和label
    """
    def padding(indice, max_length, pad_idx=0):
        """
        填充每个batch的句子长度
        """
        pad_indice =[]
        for item in indice:
            if len(item) < max_length:
                pad_indice.append(item + [pad_idx] * max(0, max_length - len(item)))
            else:
                pad_indice.append(item[:max_length])
       
        return torch.tensor(pad_indice)

    token_ids = [data["input_ids"] for data in batch]
    max_length = min(max([len(t) for t in token_ids]),maxlength)# batch中样本的最大的长度

    labels = torch.tensor([data["labels"] for data in batch])
    token_ids_padded = padding(token_ids, max_length)
   
    return token_ids_padded, labels



def train_tfidf(train_data,dim = 10000):
    tfidf = TFIDF(
        min_df=5,
        max_features=dim,
        ngram_range=(1, 3),
        use_idf=1,
        smooth_idf=1
    )
    tfidf.fit(train_data)

    return tfidf

def train_SVC(vec, label):
    SVC = LinearSVC()
    SVC.fit(vec, label)
    return SVC

if __name__ =="__main__":
    # show_label("files/first_stage/train.json","time")
    # build_dic("files/first_stage/train.json")
    deal_samples("files/first_stage/train.json","files/train_cut/train.json")
    deal_samples("files/first_stage/test.json","files/train_cut/test.json")
    # X,Y= read_Data("files/data_train.json")

    # tfidf = train_tfidf(X)
    # vec = tfidf.transform(X)

    # print('time SVC')
    # # model = train_SVC(vec,Y)


    # # joblib.dump(model, 'model/time.model')
    # joblib.dump(tfidf, 'model/tfidf.model')

