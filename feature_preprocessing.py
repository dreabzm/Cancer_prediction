#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 16:18:46 2018

@author: buyun
"""
import csv
import numpy as np
import random

class preproc:
    def __init__(self, sparse_field, shuffle=True):
        self.sparse_field = sparse_field
        self.shuffle = shuffle
    def get_data(self):
        Xd = dense_feature_preproc()
        Xs = sparse_feature_preproc(feature_fields = self.sparse_field)
        label = label_preproc()
        
        if (self.shuffle == True):
            random_seed = random.randint(0, 100)
            random.seed(random_seed)
            random.shuffle(Xd)
            for key in Xs.keys():
                random.seed(random_seed)
                random.shuffle(Xs[key])
            random.seed(random_seed)
            random.shuffle(label)
        
        return Xd, Xs, label

def dense_feature_preproc(
        csv_table='/home/buyun/sparseNN/tables/all_table_dense_encoded.csv',
    ):
    with open(csv_table, 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows= [row for row in reader]
        
    Xv = np.array(rows).astype(np.float32)#rows是数据类型是‘list',转化为数组类型好处理
    return Xv

def sparse_feature_preproc(
        csv_table='/home/buyun/sparseNN/tables/all_table_sparse.csv',
        feature_fields = 2,
    ):
    sparse_features = {}
    with open(csv_table, 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows= [row for row in reader]
        
    feature_len = len(rows[0]) / feature_fields
    for i in range(len(rows)):
        for j in range(len(rows[i])):
            if (rows[i][j] == "NA"):
                rows[i][j] = 0
    
    Xv = np.array(rows).astype(np.float32)
            
    for i in range(feature_fields):
        temp = Xv[:, i*feature_len: (i + 1) * feature_len]
        sparse_features[i] = temp

    return sparse_features

def label_preproc(
        csv_table='/home/buyun/sparseNN/tables/label.csv',
    ):
    with open(csv_table, 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows_label= [row for row in reader]
    
    Y = np.array(rows_label).astype(np.int32)
    
    return Y

dense_feature_preproc()
