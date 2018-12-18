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
        self.Xd, self.Xs, self.label = self.get_data()
    def get_data(self):
        Xd = dense_feature_preproc()
        Xs = sparse_feature_preproc(feature_fields = self.sparse_field)
        label = label_preproc()
        
        if (self.shuffle):
            state = np.random.get_state()
            np.random.shuffle(Xd)
            for key in Xs.keys():
                np.random.set_state(state)
                np.random.shuffle(Xs[key])
            np.random.set_state(state)
            np.random.shuffle(label)
        
        return Xd, Xs, label
    def get_pairwise_data(self, train, per):
#        Xd, Xs, label = self.get_data()
        n = len(self.Xd)
        if train:
            self.Xd.tolist()
            Xd = self.Xd[:int(per * n)]
            Xs = {}
            for key in self.Xs.keys():
                self.Xs[key].tolist()
                Xs[key] = self.Xs[key][:int(per * n)]
            self.label.tolist()
            label = self.label[:int(per * n)]
            Xd_pos_temp = []
            Xs_pos_temp = {}
            for key in Xs.keys():
                Xs_pos_temp[key] = []  
            Xd_neg_temp = []
            Xs_neg_temp = {}
            for key in Xs.keys():
                Xs_neg_temp[key] = []  
            for i in range(len(label)):
                if (label[i] == 1):
                    Xd_pos_temp.append(Xd[i])
                    for key in Xs.keys():
                        Xs_pos_temp[key].append(Xs[key][i])
                else:
                    Xd_neg_temp.append(Xd[i])
                    for key in Xs.keys():
                        Xs_neg_temp[key].append(Xs[key][i])
            
            Xd_anchor = []
            Xs_anchor = {}
            for key in Xs.keys():
                Xs_anchor[key] = []  
            Xd_mirror = []
            Xs_mirror = {}
            for key in Xs.keys():
                Xs_mirror[key] = []
            plabel = []
            for j in range(len(Xd_pos_temp)):
                for i in range(len(Xd_neg_temp)):
                    if (random.random() < 0.5):
                        Xd_anchor.append(Xd_pos_temp[j])
                        for key in Xs.keys():
                            Xs_anchor[key].append(Xs_pos_temp[key][j])
                        Xd_mirror.append(Xd_neg_temp[i])
                        for key in Xs.keys():
                            Xs_mirror[key].append(Xs_neg_temp[key][i])
                        plabel.append([1.0])
                    else:
                        Xd_anchor.append(Xd_neg_temp[i])
                        for key in Xs.keys():
                            Xs_anchor[key].append(Xs_neg_temp[key][i])
                        Xd_mirror.append(Xd_pos_temp[j])
                        for key in Xs.keys():
                            Xs_mirror[key].append(Xs_pos_temp[key][j])
                        plabel.append([-1.0])
            Xd_anchor = np.array(Xd_anchor).astype(np.float32)
            for key in Xs.keys():
                Xs_anchor[key] = np.array(Xs_anchor[key]).astype(np.float32)
            Xd_mirror = np.array(Xd_mirror).astype(np.float32)
            for key in Xs.keys():
                Xs_mirror[key] = np.array(Xs_mirror[key]).astype(np.float32)
            plabel = np.array(plabel).astype(np.float32)
            
            
            state = np.random.get_state()
            np.random.shuffle(Xd_anchor)
            for key in Xs.keys():
                np.random.set_state(state)
                np.random.shuffle(Xs_anchor[key])
            np.random.set_state(state)
            np.random.shuffle(Xd_mirror)
            for key in Xs.keys():
                np.random.set_state(state)
                np.random.shuffle(Xs_mirror[key])
            np.random.set_state(state)
            np.random.shuffle(plabel)
            
            return Xd_anchor, Xs_anchor, Xd_mirror, Xs_mirror, plabel
        else:
            self.Xd.tolist()
            Xd = self.Xd[int(per * n):]
            Xs = {}
            for key in self.Xs.keys():
                self.Xs[key].tolist()
                Xs[key] = self.Xs[key][int(per * n):]
            self.label.tolist()
            label = self.label[int(per * n):]
            Xd_pos_temp = []
            Xs_pos_temp = {}
            for key in Xs.keys():
                Xs_pos_temp[key] = []  
            Xd_neg_temp = []
            Xs_neg_temp = {}
            for key in Xs.keys():
                Xs_neg_temp[key] = []  
            for i in range(len(label)):
                if (label[i] == 1):
                    Xd_pos_temp.append(Xd[i])
                    for key in Xs.keys():
                        Xs_pos_temp[key].append(Xs[key][i])
                else:
                    Xd_neg_temp.append(Xd[i])
                    for key in Xs.keys():
                        Xs_neg_temp[key].append(Xs[key][i])
            
            Xd_anchor = []
            Xs_anchor = {}
            for key in Xs.keys():
                Xs_anchor[key] = []  
            Xd_mirror = []
            Xs_mirror = {}
            for key in Xs.keys():
                Xs_mirror[key] = []
            plabel = []
            for j in range(len(Xd_pos_temp)):
                for i in range(len(Xd_neg_temp)):
                    if (random.random() < 0.5):
                        Xd_anchor.append(Xd_pos_temp[j])
                        for key in Xs.keys():
                            Xs_anchor[key].append(Xs_pos_temp[key][j])
                        Xd_mirror.append(Xd_neg_temp[i])
                        for key in Xs.keys():
                            Xs_mirror[key].append(Xs_neg_temp[key][i])
                        plabel.append([1.0])
                    else:
                        Xd_anchor.append(Xd_neg_temp[i])
                        for key in Xs.keys():
                            Xs_anchor[key].append(Xs_neg_temp[key][i])
                        Xd_mirror.append(Xd_pos_temp[j])
                        for key in Xs.keys():
                            Xs_mirror[key].append(Xs_pos_temp[key][j])
                        plabel.append([-1.0])
            Xd_anchor = np.array(Xd_anchor).astype(np.float32)
            for key in Xs.keys():
                Xs_anchor[key] = np.array(Xs_anchor[key]).astype(np.float32)
            Xd_mirror = np.array(Xd_mirror).astype(np.float32)
            for key in Xs.keys():
                Xs_mirror[key] = np.array(Xs_mirror[key]).astype(np.float32)
            plabel = np.array(plabel).astype(np.float32)
            
            
            state = np.random.get_state()
            np.random.shuffle(Xd_anchor)
            for key in Xs.keys():
                np.random.set_state(state)
                np.random.shuffle(Xs_anchor[key])
            np.random.set_state(state)
            np.random.shuffle(Xd_mirror)
            for key in Xs.keys():
                np.random.set_state(state)
                np.random.shuffle(Xs_mirror[key])
            np.random.set_state(state)
            np.random.shuffle(plabel)
            
            return Xd_anchor, Xs_anchor, Xd_mirror, Xs_mirror, plabel

def dense_feature_preproc(
        csv_table='all_table_dense_encoded.csv',
    ):
    with open(csv_table, 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows= [row for row in reader]
        
    Xv = np.array(rows).astype(np.float32)#rows是数据类型是‘list',转化为数组类型好处理
    return Xv

def sparse_feature_preproc(
        csv_table='all_table_sparse.csv',
        feature_fields = 2,
    ):
    sparse_features = {}
    with open(csv_table, 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows= [row for row in reader]
        
    feature_len = len(rows[0]) // feature_fields
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
        csv_table='label.csv',
    ):
    with open(csv_table, 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows_label= [row for row in reader]
    
    Y = np.array(rows_label).astype(np.int32)
    
    return Y
#
#aa = preproc(6)
#a, b, c, d, e = aa.get_pairwise_data(train=True, per=0.8)
