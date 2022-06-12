get_ipython().system('pip install -r requirements.txt')
get_ipython().system('pip install wandb')
get_ipython().system('pip install torch==1.7.1')
get_ipython().system('pip install googletrans==4.0.0-rc1')

import os
import easydict
from sklearn.preprocessing import LabelEncoder
import time
import datetime
from datetime import datetime
import random
import wandb


import sys
import torch
import numpy as np
import pandas as pd
import sklearn
print("python: ",sys.version)
print("pytorch: ", torch.__version__)
print("numpy : ", np.__version__)
print("pandas : ", pd.__version__)
print("sklearn : ", sklearn.__version__)


import torch

import torch.nn as nn
from torch.nn import Module, Embedding, LSTM, Linear, Dropout
from torch.nn.functional import one_hot, binary_cross_entropy
from sklearn import metrics
from torch.utils.data import DataLoader, random_split


class SEKT(Module):
    def __init__(self, num_q, num_s, emb_sizeq, emb_sizes, hidden_size):
        super().__init__()
        self.num_q = num_q
        self.num_s = num_s
        self.emb_sizeq = emb_sizeq
        self.emb_sizes = emb_sizes
        self.hidden_size = hidden_size

        self.interaction_embq = Embedding(self.num_q * 2, self.emb_sizeq)
        self.interaction_embs = Embedding(self.num_s * 2, self.emb_sizes)
        self.lstm_layerq = LSTM(
            self.emb_sizeq, self.hidden_size, batch_first=True
        )
        self.lstm_layers = LSTM(
            self.emb_sizes, self.hidden_size, batch_first=True
        )
        self.out_layer = Linear(self.hidden_size, self.num_q)
        self.out_layer2 = Linear(self.hidden_size, self.num_s)
        self.dropout_layer = Dropout()

    def forward(self, q, r, flag):
        if(flag == 'q'):
          x = q + self.num_q * r
          
          h, _ = self.lstm_layerq(self.interaction_embq(x))
          y = self.out_layer(h)
          y = self.dropout_layer(y)
          y = torch.sigmoid(y)

          return y


import pickle

from torch.utils.data import Dataset

from models.utils import match_seq_len_for_SEKT


DATASET_DIR = "data/"


class test(Dataset):
    def __init__(self, seq_len, sentence, dataset_dir=DATASET_DIR) -> None:
        super().__init__()

        self.dataset_dir = dataset_dir
        self.dataset_path = os.path.join(
            self.dataset_dir, "user.csv"
        )
        self.train_path = os.path.join(
            self.dataset_dir, "train_data.csv"
        )

        self.q_seqs, self.s_seqs, self.rq_seqs, self.rs_seqs, self.q_list,             self.s_list, self.u_list, self.q2idx, self.s2idx, self.u2idx = self.preprocess(sentence)

        self.num_u = 1
        self.num_q = self.q_list.shape[0]
        self.num_s = self.s_list.shape[0]

        if seq_len:
            self.q_seqs, self.s_seqs, self.rq_seqs, self.rs_seqs =                 match_seq_len_for_SEKT(self.q_seqs, self.s_seqs, self.rq_seqs, self.rs_seqs, seq_len)

        self.len = len(self.q_seqs)

    def __getitem__(self, index):
        return self.q_seqs[index], self.s_seqs[index], self.rq_seqs[index], self.rs_seqs[index]

    def __len__(self):
        return self.len

    def preprocess(self, sentence):
        df = pd.read_csv(self.dataset_path)

        train = pd.read_csv(self.train_path).dropna(subset=["pos","sen_pos"])            .drop_duplicates(subset=["days", "pos","sen_pos"])            .sort_values(by=["days"])

        u_list = [1]
        q_list = np.unique(train["pos"].values)
        s_list = np.unique(train["sen_pos"].values)


        u2idx = {u: idx for idx, u in enumerate(u_list)} #인덱스 번호지정
        q2idx = {q: idx for idx, q in enumerate(q_list)} #인덱스 번호지정
        s2idx = {s: idx for idx, s in enumerate(s_list)}

        q_seqs = []
        s_seqs = []
        rq_seqs = []
        rs_seqs = []

        q_seq = []
        s_seq = []
        rq_seq = []
        rs_seq = []
        for i in range(len(sentence)):
          q_seq.append(q2idx[sentence[i][4]]) #pos
          s_seq.append(s2idx[sentence[i][12]]) #sen_pos
          rq_seq.append(sentence[i][6]) #correct
          rs_seq.append(sentence[i][13]) #sen_correct

        q_seqs.append(q_seq)
        s_seqs.append(s_seq)
        rq_seqs.append(rq_seq)
        rs_seqs.append(rs_seq)

        return q_seqs, s_seqs, rq_seqs, rs_seqs, q_list, s_list, u_list, q2idx, s2idx, u2idx


def process_batch(batch):
    q, s, rq, rs, qshft, sshft, rqshft, rsshft, m = batch
    
    return q, rq, m


def load_model():
    model_path = os.path.join("ckpts/SEKT/DUOLINGO+", "model.ckpt")
    load_state = torch.load(model_path)
    model = SEKT(16, 1641, 20, 300, 200)

    model.load_state_dict(load_state, strict=True)

    return model


from torch import FloatTensor
from torch.nn.utils.rnn import pad_sequence
def collate_fn(batch, pad_val=-1):
    q_seqs = []
    s_seqs = []
    rq_seqs = []
    rs_seqs = []

    qshft_seqs = []
    sshft_seqs = []
    rqshft_seqs = []
    rsshft_seqs = []

    for q_seq, s_seq, rq_seq, rs_seq in batch:
        q_seqs.append(FloatTensor(q_seq[:-1]))
        s_seqs.append(FloatTensor(s_seq[:-1]))
        rq_seqs.append(FloatTensor(rq_seq[:-1]))
        rs_seqs.append(FloatTensor(rs_seq[:-1]))

        qshft_seqs.append(FloatTensor(q_seq[1:]))
        sshft_seqs.append(FloatTensor(s_seq[1:]))
        rqshft_seqs.append(FloatTensor(rq_seq[1:]))
        rsshft_seqs.append(FloatTensor(rs_seq[1:]))

    q_seqs = pad_sequence(
        q_seqs, batch_first=True, padding_value=pad_val
    )
    s_seqs = pad_sequence(
        s_seqs, batch_first=True, padding_value=pad_val
    )

    rq_seqs = pad_sequence(
        rq_seqs, batch_first=True, padding_value=pad_val
    )
    rs_seqs = pad_sequence(
        rs_seqs, batch_first=True, padding_value=pad_val
    )

    qshft_seqs = pad_sequence(
        qshft_seqs, batch_first=True, padding_value=pad_val
    )
    sshft_seqs = pad_sequence(
        sshft_seqs, batch_first=True, padding_value=pad_val
    )
    rqshft_seqs = pad_sequence(
        rqshft_seqs, batch_first=True, padding_value=pad_val
    )
    rsshft_seqs = pad_sequence(
        rsshft_seqs, batch_first=True, padding_value=pad_val
    )

    mask_seqs = (q_seqs != pad_val)

    q_seqs, s_seqs, rq_seqs, rs_seqs, qshft_seqs, sshft_seqs, rqshft_seqs, rsshft_seqs =         q_seqs * mask_seqs, s_seqs * mask_seqs,         rq_seqs * mask_seqs, rs_seqs * mask_seqs,         qshft_seqs * mask_seqs, sshft_seqs * mask_seqs,         rqshft_seqs * mask_seqs, rsshft_seqs * mask_seqs

    return q_seqs, s_seqs, rq_seqs, rs_seqs, qshft_seqs, sshft_seqs, rqshft_seqs, rsshft_seqs, mask_seqs


from torch.utils.data import DataLoader, random_split
def inference(datas):
   model = load_model() ##dkt
   model.eval()
   test_loader = DataLoader(
          datas, batch_size=len(datas), shuffle=False,
          collate_fn=collate_fn)
  
   total_preds = []

   for data in test_loader:
     pos_dict = ['형용사', '부치사', '부사', '조동사', '접속사', '한정사', '감탄사', '명사', '수사', '불변화사', '대명사', '고유명사', '구두점', '종속접속사', '동사', '주석불가']
     q, rq, m = process_batch(data)
     preds = model(q.long(),rq.long(),'q')

     q_exist = []
     for i in range(len(q[0])): #100
        if(m[0][i]):
          q_exist.append(int(q[0][i].item()))
      
     pred = preds[:,len(q_exist)-1] ## 마지막 단어를 연습한 후 예측되는 품사별 확률들 

     avg_prob = 0
     for j in range(len(pred[0])):
        avg_prob += pred[0][j].item()
     avg_prob = avg_prob/16 * 100
     return avg_prob