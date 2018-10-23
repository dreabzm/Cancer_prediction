#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 22:36:06 2018

@author: buyun
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from Feature_preprocessing import preproc
from sklearn import metrics

def embedding_layer(input, dim_in, dim_out, name="embedding"):
    with tf.name_scope(name):
        w = tf.Variable(tf.zeros([dim_in, dim_out]), name="W")
        b = tf.Variable(tf.zeros([dim_out]), name="B")
        out_put = tf.matmul(input, w) + b
        return out_put

def fc_layer(input, dim_in, dim_out, name="fc"):
    with tf.name_scope(name):
        w = tf.Variable(tf.zeros([dim_in, dim_out]), name="W")
        b = tf.Variable(tf.zeros([dim_out]), name="B")
        act = tf.nn.relu(tf.matmul(input, w) + b)
        return act

def build_model(dense_arc, sparse_arc, sparse_field, arc, batch_size=8000):
    assert dense_arc[len(dense_arc) - 1] == sparse_arc[len(sparse_arc) - 1], \
        "last dimention of dense arch and sparse arch must be the same."
    
    y = tf.placeholder(tf.float32, shape=[None, 1], name="labels")
    #y = tf.reshape(y, [-1, 1])
    
    dense_input = tf.placeholder(tf.float32, shape=[None, dense_arc[0]], name="denseInput")
    #dense_input = tf.reshape(dense_input, [-1, dense_arc[0]])
    sparse_input = {}
    for i in range(sparse_field):
        sparse_input[i] = tf.placeholder(tf.float32, shape=[None, sparse_arc[0]], name="sparseInput{}".format(i))
       # sparse_input[i] = tf.reshape(sparse_input[i], [-1, sparse_arc[0]])
    
    dense_net = []
    for i in range(len(dense_arc) - 1):
        if (i == 0):
            dense_net.append(fc_layer(dense_input, dense_arc[i], dense_arc[i + 1], name="denseFc{}".format(i)))
        else:
            dense_net.append(fc_layer(dense_net[i - 1], dense_arc[i], dense_arc[i + 1], name="denseFc{}".format(i)))
    
    with tf.name_scope("sparse_embedding"):
        sparse_embedding = []        
        for i in range(sparse_field):
            sparse_embedding.append(embedding_layer(sparse_input[i], sparse_arc[0], sparse_arc[1], name="sparseEmbedding{}".format(i)))
        
    #initialize over all input
    product = [dense_net[-1]]
    #pairwsise dot
    sparse_embedding.append(dense_net[len(dense_net) - 1])
    with tf.name_scope("pairwiseDot"):
        count = 1
        for i in range(sparse_field + 1):
            for j in range(i + 1, sparse_field + 1):
                product.append(tf.multiply(sparse_embedding[i], sparse_embedding[j], name="pairwiseDot{}".format(count)))
                #fc_all = tf.concat([dot_product, fc_all], -1, name="concat{}".format(count))
                count = count + 1
    
    with tf.name_scope("Concat"):
        fc_all = tf.concat(product, -1, name="concat")
        
    fc = []
    for i in range(len(arc)):
        if (i == 0):
            fc.append(fc_layer(fc_all, count * dense_arc[len(dense_arc) - 1], arc[i], name="overAllFc{}".format(i)))
        else:
            fc.append(fc_layer(fc[i - 1], arc[i - 1], arc[i], name="overAllFc{}".format(i)))
            
    out = fc[-1]
    #out = tf.convert_to_tensor(out)
    #y = tf.convert_to_tensor(y)
    #out = tf.nn.sigmoid(out, name="sigmoid")
    with tf.name_scope("crossEntropy"):
        #cost = tf.losses.log_loss(y, out)
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=out,labels=y))
    
    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(0.1).minimize(cost) 
        
    with tf.name_scope("auc"):
        auc_value, auc_op = tf.metrics.auc(y, out)
    #auc = tf.metrics.auc(y, out)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    writer = tf.summary.FileWriter("/home/buyun/sparseNN/tables/board2")
    writer.add_graph(sess.graph)
    
    data = preproc(sparse_field=sparse_field)   
    Xd, Xs, label = data.get_data()
    n = len(Xd)
    
    for i in range(2000):
        Xd_batch = Xd[: ((i + 1) * batch_size) % n,:]
        Xs_batch = {}
        for key in Xs.keys():
            Xs_batch[key] = Xs[key][: ((i + 1) * batch_size) % n,:]
        label_batch = label[: ((i + 1) * batch_size) % n,:]
        
        feed_dict = {}
        feed_dict[y] = label_batch
        feed_dict[dense_input] = Xd_batch
        for j in range(sparse_field):
            feed_dict[sparse_input[j]] = Xs_batch[j]
        if i % 50 == 0:
            sess.run(auc_op, feed_dict=feed_dict)
            value = sess.run(auc_value, feed_dict=feed_dict)
            print("step %d, training auc %g" % (i, value))

        sess.run(train_step, feed_dict=feed_dict)
        
def build_deep_model(arc, useEmbedding=False, useSparse=False, batch_size=200, lr=0.01, iter=20000):
    if (useSparse == False):
        dense_input = tf.placeholder(tf.float32, shape=[None, 25], name="dense_input")
        y = tf.placeholder(tf.float32, shape=[None, 1], name="label")
        
        layer = {}
        for i in range(len(arc)):
            if (i == 0):
                layer[i] = fc_layer(dense_input, 25, arc[i], name="layer{}".format(i))
            else:
                layer[i] = fc_layer(layer[i - 1], arc[i - 1], arc[i], name="layer{}".format(i))
        
        pre_out = layer[len(arc) - 1]
        out = tf.sigmoid(pre_out)
        
        return train(dense_input, out, y, batch_size, model_type="deep_model_dense", lr=lr, iter=iter)
    else:
        if (useEmbedding):
            dense_input = tf.placeholder(tf.float32, shape=[None, 252], name="dense_input")
            y = tf.placeholder(tf.float32, shape=[None, 1], name="label")
            
            layer = {}
            for i in range(len(arc) - 1):
                if (i == 0):
                    layer[i] = fc_layer(dense_input, 252, arc[i], name="layer{}".format(i))
                else:
                    layer[i] = fc_layer(layer[i - 1], arc[i - 1], arc[i], name="layer{}".format(i))
            
            last_layer = layer[len(arc) - 1]
            pre_out = embedding_layer(last_layer, arc[-1], 1, name="preoutput")
    
            out = tf.sigmoid(pre_out)
            
            return train(dense_input, out, y, batch_size, model_type="deep_model_dense_sparse", lr=lr, iter=iter)
            

def build_wide_model(useSparse=False, batch_size=200, lr=0.01, iter=20000):
    if (useSparse == False):
        dense_input = tf.placeholder(tf.float32, shape=[None, 25], name="dense_input")
        y = tf.placeholder(tf.float32, shape=[None, 1], name="label")
        
        pre_out = embedding_layer(dense_input, 25, 1, name="pre_out")
        out = tf.sigmoid(pre_out)
        
        return train(dense_input, out, y, batch_size, model_type="wide_model_dense", lr=lr, iter=iter)
    else:
        dense_input = tf.placeholder(tf.float32, shape=[None, 252], name="dense_input")
        y = tf.placeholder(tf.float32, shape=[None, 1], name="label")
        
        pre_out = embedding_layer(dense_input, 252, 1, name="pre_out")
        out = tf.sigmoid(pre_out)
        
        return train(dense_input, out, y, batch_size, model_type="wide_model_dense_sparse", lr=lr, iter=iter)
        
def train(input, output, y, batch_size, model_type, lr=0.01, iter=20000):
    
    with tf.name_scope("crossEntropy"):
        cost = tf.losses.log_loss(y, output)
        #cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output,labels=y))
    
    with tf.name_scope("auc"):
        auc_value, auc_op = tf.metrics.auc(y, output)    
    
    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(lr).minimize(cost) 
        

    #auc = tf.metrics.auc(y, out)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    writer = tf.summary.FileWriter("/home/buyun/sparseNN/tables/board2")
    writer.add_graph(sess.graph)
    
    data = preproc(sparse_field=1)   
    Xd, Xs, label = data.get_data()
    n = len(Xd)
    
    if (model_type == "wide_model_dense" or model_type == "deep_model_dense"):    
        for i in range(iter):
            start = (i * batch_size) % n
            end = ((i + 1) * batch_size) % n
            Xd_batch = Xd[start: end,:]
            
            label_batch = label[start: end,:]
            
            feed_dict = {}
            feed_dict[y] = label_batch
            feed_dict[input] = Xd_batch

            if i % 500 == 0:
                sess.run(auc_op, feed_dict=feed_dict)
                value = sess.run(auc_value, feed_dict=feed_dict)
                print("step %d, training auc %g" % (i, value))
    
            sess.run(train_step, feed_dict=feed_dict)
            
    if (model_type == "wide_model_dense_sparse" or model_type == "deep_model_dense_sparse"):
        X = np.hstack((Xd, Xs[0]))
        #dataset=make_dataset(X,label,10)
        eval_auc=[]
        #for X_train,label_train,X_test,label_test in dataset:
            #sess.run(tf.global_variables_initializer())
            #sess.run(tf.local_variables_initializer())
        X_test = X[: len(X)//10]
        label_test = label[: len(X)//10]
        X_train = X[len(X)//10:]
        label_train = label[len(X)//10:]
        
        for i in range(iter):
            start = (i * batch_size) % n
            end = ((i + 1) * batch_size) % n
            if (start < end):
                X_batch = X_train[start: end,:]
                
                label_batch = label_train[start: end,:]
            else:
                X_batch = np.vstack((X_train[start:,:], X_train[: end,:]))
                label_batch = np.vstack((label_train[start:,:], label_train[: end,:]))
                
            feed_dict = {}
            feed_dict[y] = label_batch
            feed_dict[input] = X_batch

            if i % 500 == 0:
                sess.run(auc_op, feed_dict=feed_dict)
                value = sess.run(auc_value, feed_dict=feed_dict)
                print("step %d, training auc %g" % (i, value))
    
            sess.run(train_step, feed_dict=feed_dict)
        feed_dict = {}
        feed_dict[y] = label_test
        feed_dict[input] = X_test
        predict = sess.run(output, feed_dict=feed_dict)
        predict = np.ndarray.flatten(predict)
        label_test = np.ndarray.flatten(label_test)
        fpr, tpr, thresholds = metrics.roc_curve(label_test, predict, pos_label=1)
        eval_auc.append(metrics.auc(fpr, tpr))
        print(np.mean(eval_auc))
    return predict, label_test

def make_dataset(X_data,y_data,n_splits):

    n = len(X_data)
    dataset = []
    for i in range(n_splits):
        X_test = X_data[i * n // n_splits: (i + 1) * n // n_splits, :]
        y_test = y_data[i * n // n_splits: (i + 1) * n // n_splits, :]
        X_train = np.vstack((X_data[0: i * n // n_splits, :], X_data[(i + 1) * n // n_splits:, :]))
        y_train = np.vstack((y_data[0: i * n // n_splits, :], y_data[(i + 1) * n // n_splits:, :]))
        dataset.append((X_train, y_train, X_test, y_test))
    
    return dataset

#build_model(dense_arc=[25, 1], sparse_arc=[227, 1], sparse_field=1, arc=[16, 16, 1])
pred, label_t = build_wide_model(useSparse=True, batch_size=4000, lr=0.01, iter=10000)
            
#data = preproc(sparse_field=1)   
#Xd, Xs, label = data.get_data()
#print(label)
