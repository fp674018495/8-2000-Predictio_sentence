  


import torch.nn as nn
import torch.nn.functional as F
import torch
import torchmetrics
from transformers import  BertConfig
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
from sklearn.svm import LinearSVC
from transformers import RobertaModel,BertModel,BertConfig,RobertaConfig
import os,sys
sys.path.append(os.path.dirname(__file__))
from files.longformer.longformer import Longformer, LongformerConfig
# import longformer

def train_tfidf(train_data,dim = 5000):
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



class BERTModel(nn.Module):
    def __init__(self, config):
        super(BERTModel, self).__init__()
        self.config = config
        model_config = BertConfig.from_pretrained(config.bert_path)
        self.bert = BertModel.from_pretrained(config.bert_path,config=model_config)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(model_config.hidden_size, config.num_classes)
        self.dropout_layer = nn.Dropout(config.dropout)
    def forward(self, inputs,mask,token_type_ids):
        context = inputs  # 输入的句子
        _, pooled = self.bert(context,
                              attention_mask=mask,
                              token_type_ids=token_type_ids)
        pooled = self.dropout_layer(pooled)
        out = self.fc(pooled)
        return out



class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}
 
    def attack(self, epsilon=0.2, emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad,dim=2)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)
 
    def restore(self, emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}



class Config_textcnn:
    learning_rate = 1e-3  # 学习率
    filter_sizes = (2, 3, 4)  # 卷积核尺寸
    num_filters = 256
    n_vocab = 60000
    embedding_pretrained = False
    embed = 300

class TextcnnModel(nn.Module):
    def __init__(self, config):
        super(TextcnnModel, self).__init__()
        if Config_textcnn.embedding_pretrained:
            self.embedding = nn.Embedding.from_pretrained(Config_textcnn.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(Config_textcnn.n_vocab, Config_textcnn.embed, padding_idx=0)
            # self.embedding.weight.requires_grad=True
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, Config_textcnn.num_filters, (k, Config_textcnn.embed)) for k in Config_textcnn.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(Config_textcnn.num_filters * len(Config_textcnn.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # [128,256,32]
        x = F.max_pool1d(x, x.size(2)).squeeze(2)  # [128,256]
        return x  # [128,256]

    def forward(self, x):
        out = self.embedding(x)  # # [128,32,300]
        out = out.unsqueeze(1)  # [128,1,32,300]
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)  # [128,1,32,300]
        out = self.dropout(out)  # [128,768]
        out = self.fc(out)  # [128,10]
        return out



class Config_DGCNN:
    learning_rate = 1e-3  # 学习率
    filter_sizes = (2, 3, 4)  # 卷积核尺寸
    num_filters = 256
    n_vocab = 55867
    embedding_pretrained = False
    embed = 300
    dgccn_params= ((3,1), (5,1), (7,1))

class TextDGCNN(nn.Module):
    def __init__(self, config):
        super(TextDGCNN, self).__init__()
        if Config_DGCNN.embedding_pretrained:
            self.embedding = nn.Embedding.from_pretrained(Config_DGCNN.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(Config_DGCNN.n_vocab, Config_DGCNN.embed, padding_idx=0)
            # self.embedding.weight.requires_grad=True
        self.dropout = nn.Dropout(config.dropout)
        self.dgcnn = nn.ModuleList(
            DGCNNLayer(Config_DGCNN.embed, Config_DGCNN.embed, k_size=param[0], dilation_rate=param[1]) for param in
            Config_DGCNN.dgccn_params)
        self.fc = nn.Linear(Config_DGCNN.embed, config.num_classes)
        # self.position_embedding = nn.Embedding(512, Config.embed)

    def forward(self, x):
        word_emb = self.embedding(x)  # # [64,256,300]
        # pos_emb = self.position_embedding(x)
        mask = (x != 0)  # [64,256]
        out = word_emb
        for dgcnn in self.dgcnn:
            out = dgcnn(out, mask)
        out=self.dropout(out)
        out = torch.max(out, dim=1)[0]
        out = self.fc(out)  # [128,10]
        return out


class DGCNNLayer(nn.Module):

    def __init__(self, in_channels, out_channels, k_size=3, dilation_rate=1, dropout=0.1):
        super(DGCNNLayer, self).__init__()
        self.k_size = k_size
        self.dilation_rate = dilation_rate
        self.hid_dim = out_channels
        self.pad_size = int(self.dilation_rate * (self.k_size - 1) / 2)
        self.dropout_layer = nn.Dropout(dropout)
        # self.liner_layer = nn.Linear(int(out_channels / 2), out_channels)
        self.glu_layer = nn.GLU()
        self.conv_layer = nn.Conv1d(in_channels, out_channels * 2, kernel_size=k_size, dilation=dilation_rate,
                                    padding=(self.pad_size))
        self.layer_normal = nn.LayerNorm(in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x,mask=None):
        '''
        :param x: shape: [batch_size, seq_length, channels(embeddings)]
        :return:
        '''
        x_r = x
        x = x.permute(0, 2, 1)  # [batch_size, hidden_size, seq_length]
        x = self.conv_layer(x)  # [batch_size, 2*hidden_size, seq_length]
        x = x.permute(0, 2, 1)  # [batch_size, seq_length, 2*hidden_size]
        x = self.glu_layer(x)  # [batch_size, seq_length, hidden_size]
        x = self.dropout_layer(x)  #
        mask = mask.unsqueeze(2).repeat(1, 1, self.hid_dim).float()
        x = x * mask
        return self.layer_normal(x + x_r)



class RNNConfig:
    hidden_size = 256
    vocab_size = 60000
    requires_grad = True
    pretrain_embeddings = False
    pretrain_embeddings_path = None
    num_layers = 2
    bidirectional = True
    embed_size = 300


class RNNClassifier(nn.Module):
    def __init__(self,config):
        super(RNNClassifier, self).__init__()
        if RNNConfig.pretrain_embeddings:
            self.embeddings = nn.Embedding.from_pretrained(RNNConfig.pretrain_embeddings_path)
        else:
            self.embeddings = nn.Embedding(RNNConfig.vocab_size, RNNConfig.embed_size)
#         self.lstm = nn.LSTM(input_size=RNNConfig.embed_size,
#                             hidden_size=RNNConfig.hidden_size,
#                             batch_first=True,
#                             num_layers=RNNConfig.num_layers,
#                             bidirectional=RNNConfig.bidirectional)
        self.gru = nn.GRU(input_size=RNNConfig.embed_size,
                          hidden_size=RNNConfig.hidden_size,
                          batch_first=True,
                          num_layers=RNNConfig.num_layers,
                          bidirectional=RNNConfig.bidirectional)
#         self.rnn = nn.RNN(input_size=RNNConfig.embed_size,
#                           hidden_size=RNNConfig.hidden_size,
#                           batch_first=True,
#                           num_layers=RNNConfig.num_layers,
#                           bidirectional=RNNConfig.bidirectional)
        self.drop_out = nn.Dropout(config.dropout)
        self.fc = nn.Linear(2*RNNConfig.hidden_size, config.num_classes)

    def forward(self, x):
        embedding = self.embeddings(x)  # [batch_size,seq_len,embed_size]
#         hidden, _ = self.lstm(embedding)  # [batch_size,seq_len,hidden_size]
        hidden, _ = self.gru(embedding)  # [batch_size,seq_len,2*hidden_size]

        hidden = self.drop_out(hidden)
        hidden = hidden[:, -1, :]  # 获取最后一层的输出
        prob = self.fc(hidden)
        return prob




class ResConfig:
    hidden_size = 256
    num_filters = 256
    n_vocab = 60000
    embedding_pretrained = False
    embed = 300

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)  
        self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
        )

    def forward(self, x):     #    torch.Size([4, 16, 1073, 300])
        out = F.relu(self.bn1(self.conv1(x)))   # torch.Size([4, 16, 1073, 300])
        out = self.bn2(self.conv2(out))  # torch.Size([4, 16, 1073, 300])
        out += self.shortcut(x)   #  torch.Size([4, 16, 1073, 300])
        out = F.relu(out)  # torch.Size([4, 16, 1073, 300])
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.embedding = nn.Embedding(Config_textcnn.n_vocab, Config_textcnn.embed, padding_idx=0)
        self.in_planes = 16
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1) 
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)   
        out = self.layer3(out)    # torch.Size([4, 64, none, 75])
        out = F.avg_pool2d(out, (out.size(2),out.size(3))) #  torch.Size([4, 64, 1, 1])
        out = out.view(out.size(0), -1)  #  torch.Size([4, 64])
        out = self.linear(out)
        return out

def resnet20(args):
    args.num_classes
    return ResNet(BasicBlock, [1, 1, 1], args.num_classes)