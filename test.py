#!C:\pythonCode
# -*- coding: utf-8 -*-
# @Time : 2021/7/9 16:49
# @Author : pobby
# @File : run.py
# @Software: PyCharm

import torch
import os 
os.environ['CUDA_VISIBLE_DEVICES'] ="2,3"

# from files.longformer.longformer import Longformer, LongformerConfig
# from files.longformer.sliding_chunks import pad_to_window_size
# from transformers import RobertaTokenizer,BertTokenizer
# from files.longformer.longformer import *


# # config = LongformerConfig.from_pretrained('schen/longformer-chinese-base-4096')
# # model = Longformer.from_pretrained('schen/longformer-chinese-base-4096', config=config)
# config = LongformerConfig.from_pretrained('premodel/longformer-base-4096')

# # choose the attention mode 'n2', 'tvm' or 'sliding_chunks'
# # 'n2': for regular n2 attantion
# # 'tvm': a custom CUDA kernel implementation of our sliding window attention
# # 'sliding_chunks': a PyTorch implementation of our sliding window attention
# config.attention_mode = 'sliding_chunks'

# model = Longformer.from_pretrained('premodel/longformer-base-4096', config=config)
# tokenizer = BertTokenizer.from_pretrained('premodel/longformer-base-4096')
# tokenizer.model_max_length = model.config.max_position_embeddings

# SAMPLE_TEXT = ' '.join(['Hello world! '] * 10 )  # long input document

# input_ids = torch.tensor(tokenizer.encode(SAMPLE_TEXT)).unsqueeze(0)  # batch of size 1

# # TVM code doesn't work on CPU. Uncomment this if `config.attention_mode = 'tvm'`
# # model = model.cuda(); input_ids = input_ids.cuda()

# # Attention mask values -- 0: no attention, 1: local attention, 2: global attention
# attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device) # initialize to local attention
# attention_mask[:, [1, 4, 21,]] = 2  # Set global attention based on the task. For example,
#                                      # classification: the <s> token
#                                      # QA: question tokens

# # padding seqlen to the nearest multiple of 512. Needed for the 'sliding_chunks' attention
# input_ids, attention_mask = pad_to_window_size(
#         input_ids, attention_mask, config.attention_window[0], tokenizer.pad_token_id)

# output = model(input_ids, attention_mask=attention_mask)[0]
# print(output)


data = torch.rand([2,3])
data.cuda()
pass