import csv
import numpy as np
import pandas as pd
import os
import torch
import torch.nn.functional as F
from torch import nn, tensor
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import random
import time

class Data:
    def __init__(self, file, length, concept_num, rows_num, is_train=False, is_test=False, index_split=None):
        csv_file = csv.reader(file, delimiter=',')
        rows = [[int(e) for e in i if i != ''] for i in csv_file]
        q_rows, r_rows = [], []
        student_num = 0
        if rows_num == 5:
            rows_slice_q, rows_slice_r = rows[2::rows_num], rows[(rows_num-1)::rows_num]
            rows_slice_q = [[x - 1 for x in sublist] for sublist in rows_slice_q]
        else:
            rows_slice_q, rows_slice_r = rows[1::rows_num], rows[(rows_num-1)::rows_num]


        zipped = zip(rows_slice_q, rows_slice_r)

        if is_test:
            for q_row, r_row in zipped:
                num = len(q_row)
                for i in range(n + 1):
                    q_rows.append(q_row[i * length:(i + 1) * length])
                    r_rows.append(r_row[i * length:(i + 1) * length])
        else:
            if is_train:
                for q_row, r_row in zipped:
                    if student_num not in index_split:
                        num = len(q_row)
                        n = num // length
                        for i in range(n + 1):
                            q_rows.append(q_row[i * length:(i + 1) * length])
                            r_rows.append(r_row[i * length:(i + 1) * length])
                    student_num += 1
            else:
                for q_row, r_row in zipped:
                    if student_num in index_split:
                        num = len(q_row)
                        n = num // length
                        for i in range(n + 1):
                            q_rows.append(q_row[i * length:(i + 1) * length])
                            r_rows.append(r_row[i * length:(i + 1) * length])
                    student_num += 1
        q_rows = [row for row in q_rows if len(row) > 3]
        r_rows = [row for row in r_rows if len(row) > 3]

        self.q_rows = q_rows
        self.r_rows = r_rows
        self.concept_num = concept_num


    def __getitem__(self, index):

        return list(zip(self.q_rows[index], self.r_rows[index]))

    def __len__(self):
        return len(self.q_rows)


def collate(batch, seq_len, window_size):
    lens = [len(row) for row in batch]
    
    max_len = max(lens)
    padded_data = []
    stride = 1  
    for row in batch:
        lenth = len(row)
        padded_data.append([[0, 0, 0]] * (seq_len - lenth + window_size - 1) + [[*e, 1] for e in row] ) 

    padded_and_windowed_data = []

    for padded_row in padded_data:
        lenth = len(padded_row)
        for t in range(0, lenth - window_size + 1, stride):
            window_data = padded_row[t:t + window_size]
            padded_and_windowed_data.append(window_data)

    batch = torch.tensor(padded_and_windowed_data).cuda()
    Q, Y, S = batch.T 
    Q, Y, S = Q.T, Y.T, S.T
    return Y, S, Q



def load_dataset(file_path, batch_size, seq_len, window_size, concept_num, rows_num, student_len, val_ratio, seed = 0, shuffle=True):
    t0 = time.time()
    torch.manual_seed(seed)
    test_data = Data(open('%s/test.csv' % file_path, 'r'), seq_len, concept_num,rows_num, is_test=True)
    
    origin_list = [i for i in range(student_len)]
    index_split = random.sample(origin_list, int(val_ratio * len(origin_list)))  # 设置0.1的valid
  
    train_data = Data(open('%s/train_valid.csv' % file_path, 'r'), seq_len, concept_num, rows_num,
                          is_train=True, index_split=index_split,is_test=False)
    valid_data = Data(open('%s/train_valid.csv' % file_path, 'r'), seq_len, concept_num, rows_num,
                          is_train=False,index_split=index_split,is_test=False)
    # Step 1.2 - Remove users with a single answer
    t1 = time.time()
    print('data',t1-t0)
    # torch.save({'q_rows': train_data.q_rows, 'r_rows': train_data.r_rows}, 'train_data.pt')
    # torch.save({'q_rows': valid_data.q_rows, 'r_rows': valid_data.r_rows}, 'valid_data.pt')
    # torch.save({'q_rows': test_data.q_rows, 'r_rows': test_data.r_rows}, 'test_data.pt')
    # Step 4 - Convert to a sequence per user id and shift features 1 timestep 
   
    window_size = 20
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda batch: collate(batch,seq_len, window_size), drop_last=True)  
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda batch: collate(batch,seq_len, window_size), drop_last=True)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda batch: collate(batch,seq_len, window_size), drop_last=True)
    t2 = time.time()
    print('dataloader',t2-t1)

    return concept_num, train_data_loader, valid_data_loader, test_data_loader
