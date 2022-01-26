

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import logging
import torch
import torch.optim as optim
from load_data.load_data import Law_dataset
from models import BERTModel,FGM,TextcnnModel,TextDGCNN,RNNClassifier,resnet20
from tqdm import tqdm
import torch.nn as nn
import argparse
import json
from functools import partial
from torchmetrics import functional 
from torch.nn import functional as F
from torch.nn.modules import CrossEntropyLoss
from torch.utils.data import DataLoader
import torchmetrics 
from logs.log_utils import Logger
import numpy as np
from load_data.load_data import collate_to_max_length,Law_dataset,collate_to_max_length_
# optim中定义了各种各样的优化方法，包括SGD
log_basic=Logger(filename="logs/train.log")
logger = log_basic.logger


class myconfig():
    model_name="bert_model"
    bert_path ="premodel/roberta"
    batch_size = 128
    lr=0.00001
    workers= 8
    max_length=512
    data_dir = "files/train_cut/"
    save_path = "model/"
    num_classes = 12
    device = "cuda:0"
    device_ids=[0,1]
    dropout = 0.3
    log_path = "logs/train.log"
    is_fgm = False # 是否使用对抗学习
    kind = "time"   # 什么任务
    word = False   #词粒还是字粒
    epoch = 10
    is_reweight = True     # 是否加权 平衡样本
    weight= [2396, 5173, 23753, 19615,23817,74090,120089,202054,268483,223249,340862,407275]


def getmodel(args):
    model_name = args.model_name.lower()
    if model_name =="bert_model":
        return BERTModel(args)
    elif model_name =="textcnn":
        return TextcnnModel(args)
    elif model_name =="textdgcnn":
        return TextDGCNN(args)
    elif model_name =="rnnclassifier":
        return RNNClassifier(args)
    elif model_name =="resnet":
        return resnet20(args)


def get_dataloader(args, prefix="train") -> DataLoader:
    """get training dataloader"""
    print("___________load {} data ..._____________".format(prefix))
    dataset = Law_dataset(args,prefix)
    if args.model_name == "bert_model":
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            collate_fn=partial(collate_to_max_length),
            shuffle=True,
            drop_last=False
        )
    else:
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            collate_fn=partial(collate_to_max_length_),
            shuffle=True, 
            drop_last=False
        )
    return dataloader


def ce_loss(outputs, labels):
    loss = F.cross_entropy(outputs, labels)
    return loss

def focal_loss(labels, logits, alpha, gamma):

    BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + 
            torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss



def CB_loss(logits,labels ,samples_per_cls, no_of_classes, loss_type="softmax", beta=0.999, gamma=2):
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes
    pass
    # print(labels)
    labels_one_hot = F.one_hot(labels, no_of_classes).float()

    weights = torch.tensor(weights).float().cuda()
    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1,no_of_classes)

    if loss_type == "focal":
        cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
    elif loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, weights = weights)
    elif loss_type == "softmax":
        pred = logits.softmax(dim = 1)
        cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)
    return cb_loss




def accuracy(outputs, labels):
    predict_scores = F.softmax(outputs, dim=1)
    predict_labels = torch.argmax(predict_scores, dim=-1) 
    acc = functional.accuracy(predict_labels, labels)
    return acc

def test(device):
    test_acc = torchmetrics.Accuracy().to(device)
    test_recall = torchmetrics.Recall(average='none', num_classes=3).to(device)
    test_precision = torchmetrics.Precision(average='none', num_classes=3).to(device)
    args = myconfig()
    model_state=torch.load('bert_model.bin',map_location=device)
    net = BERTModel(model_state["config"]).to(device=device).eval()
    net.load_state_dict(model_state["state_dict"])
    testloader = get_dataloader(args, prefix='test')
    correct = 0
    total = 0

    with torch.no_grad():
        for texts, mask, tokens, labels in tqdm(testloader):
            texts = texts.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            tokens = tokens.to(device)
            # 预测
            outputs = net((texts, mask, tokens))
            predict_scores = F.softmax(outputs, dim=1)
            predict_labels =  torch.argmax(predict_scores, dim=-1)
            test_acc(predict_labels, labels)
            test_recall(predict_labels, labels)
            test_precision(predict_labels, labels)
            correct += (predict_labels == labels).sum().item()
            total+= args.batch_size
    total_acc = test_acc.compute()
    total_recall = test_recall.compute()
    total_precision = test_precision.compute()
    print(f"Test Error: \n Accuracy: {(100 * correct/total):>0.2f}%, "
          f"torch metrics acc: {(100 * total_acc):>0.2f}%\n")
    print("recall of every test dataset class: ", total_recall)
    print("precision of every test dataset class: ", total_precision)
    

def evaluate(config, model,test=False):
    model.eval()
    test_acc = torchmetrics.Accuracy()
    test_recall = torchmetrics.Recall(average='none', num_classes=config.num_classes)
    test_precision = torchmetrics.Precision(average='none', num_classes=config.num_classes)
    loss_total = 0
    data_iter=get_dataloader(config,'test')
    all_outputs=torch.tensor([]).cuda()
    all_labels=torch.tensor([],dtype=int).cuda()
    with torch.no_grad():
        print("evaling...")
        for data in tqdm(data_iter):
            if config.model_name == "bert_model":
                texts, mask, tokens, labels =data

                texts = texts.to(config.device)
                mask = mask.to(config.device)
                tokens = tokens.to(config.device)

                outputs = model(texts, mask, tokens)
            else:
                texts, labels = data
                texts = texts.to(config.device)
                outputs = model(texts)
            predict_labels =  torch.argmax(outputs, dim=-1).to("cpu")
            test_acc(predict_labels, labels)
            test_recall(predict_labels, labels)
            test_precision(predict_labels, labels)

            labels = labels.to(config.device)
            loss = ce_loss(outputs, labels)
            loss_total += loss
            all_outputs=torch.cat([all_outputs,outputs])
            all_labels=torch.cat([all_labels,labels])

       
    total_acc = test_acc.compute()
    total_recall = test_recall.compute()
    total_precision = test_precision.compute()
    logger.info(" %s val_acc ,val_recall ,val_precsion"%config.model_name,)
    logger.info((total_acc,total_recall,total_precision))

    all_outputs=all_outputs.reshape(-1,config.num_classes)
    all_labels=all_labels.reshape(-1)
    if test:
      pass 
    acc=accuracy(all_outputs, all_labels)

    return acc, loss_total / len(data_iter)


def train(args):
    log_basic.rebuild_log(args.log_path)
    trainloader = get_dataloader(args)
    net = getmodel(args)
    net = nn.DataParallel(net,device_ids=args.device_ids).cuda()
    fgm=FGM(net)
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    print("%s Start Training..."%args.model_name)
    dev_best_loss=0.5
    all_acc ,all_loss = [],[]
    for epoch in range(args.epoch):
        # 我们用一个变量来记录每100个batch的平均loss
        loss100= 0.0
        # 我们的dataloader派上了用场
        print("train_epoch: %d"%epoch)
        for i, data in enumerate(tqdm(trainloader)):
            if args.model_name =="bert_model":
                inputs, mask, ids, labels = data
                inputs, labels = inputs.cuda(), labels.cuda() # 注意需要复制到GPU
                mask, ids = mask.cuda(), ids.cuda()
                outputs = net(inputs, mask, ids)
                if args.is_reweight:
                    loss = CB_loss(outputs, labels,args.weight,args.num_classes)
                else:
                    loss= ce_loss(outputs, labels)

                loss.backward()
                if myconfig.is_fgm:
                    fgm.attack()
                    output2 = net(inputs, mask, ids)
                    loss_adv= ce_loss(output2, labels)
                    loss_adv.backward()
                    fgm.restore()
            else:
                inputs, labels = data
                inputs, labels = inputs.cuda(), labels.cuda() # 注意需要复制到GPU
                outputs = net(inputs)
                if args.is_reweight:
                    loss = CB_loss(outputs, labels,args.weight,args.num_classes)
                else:
                    loss= ce_loss(outputs, labels)
                loss.backward()
                if myconfig.is_fgm:
                    fgm.attack()
                    output2 = net(inputs)
                    loss_adv= ce_loss(output2, labels)
                    loss_adv.backward()
                    fgm.restore()
            optimizer.step()
            optimizer.zero_grad()
            loss100 += loss.item()     
            if i % 100 == 99:
                logger.info('[Epoch %d, Batch %5d] train_loss: %.6f' %
                      (epoch + 1, i + 1, loss100 / 100))
                loss100 = 0.0
        if epoch%2 == 0:
            eval_acc, eval_loss=evaluate(args,net)
            net.train()
            all_acc.append(eval_acc)
            all_loss.append(eval_loss)
            logger.info('[Epoch %d, Batch %5d] eval_loss: %.9f eval_acc: %.9f' %
                        (epoch + 1, i + 1, eval_loss,eval_acc))

        if eval_loss < dev_best_loss or eval_acc> 0.45:
            dev_best_loss = eval_loss
            if not os.path.exists(args.save_path+"/"+args.model_name+"/"):
                os.makedirs(args.save_path+"/"+args.model_name+"/")
            torch.save({"state_dict":net.state_dict(),"config":args,"result":(eval_loss,eval_acc),
                        "all_loss":all_loss,"all_acc":all_acc}, 
                            args.save_path+"/"+args.model_name+"/"+args.model_name+'%.4f.bin'%eval_acc)
        
    logger.info("%s Done Training!"%args.model_name)


    

if __name__ == "__main__":
    # torch.rand(2,3).cuda()
    # task_list=["textcnn","rnnclassifier","textdgcnn","resnet","bert_model"]
    # task_list=["textcnn","rnnclassifier","textdgcnn","resnet"]
    task_list =["bert_model"]
    tack_batch_size =[60]
    # task_list=["resnet"]
    # task_list=["bert_model"]
    for index,model_name in  enumerate(task_list):
        myconfig.batch_size = tack_batch_size[index]
        myconfig.model_name = model_name 
        myconfig.log_path ="logs/"+ model_name +".log"
        if model_name == "bert_model":
            myconfig.data_dir = "files/first_stage/"
            myconfig.epoch = 10
        train(myconfig)
