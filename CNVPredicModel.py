#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 22:36:06 2018
restart kernel when rerun the program
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
        w = tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev = 0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape = [dim_out]), name="B")
        out_put = tf.matmul(input, w) + b
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("embeddings", out_put)
        return out_put

def fc_layer(input, dim_in, dim_out, name="fc"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev = 0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape = [dim_out]), name="B")
        act = tf.nn.relu(tf.matmul(input, w) + b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act

def build_denseSparse_model(dense_arc, sparse_arc, sparse_field, arc, setdiv=10, lr=0.01, iteration=2000, batch_size=8000):
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
    #with tf.name_scope("pairwiseDot"):
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
            
    last_layer = fc[-1]
    pre_out = embedding_layer(last_layer, arc[-1], 1, name="preoutput")
    out = tf.sigmoid(pre_out)
    
    return dense_input, sparse_input, out, y
    
    
    #out = tf.convert_to_tensor(out)
    #y = tf.convert_to_tensor(y)
    #out = tf.nn.sigmoid(out, name="sigmoid")
def sparseNN_train(d_input, s_input, output, y, batch_size, sparse_field, setdiv=10, lr=0.01, iteration=2000, l2regu=[False, 0.01]):
    with tf.name_scope("crossEntropy"):
        cost = tf.losses.log_loss(y, output)
        #cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output,labels=y))
        if (l2regu[0]):
            tv = tf.trainable_variables()
            regularization_cost = l2regu[1]* tf.reduce_sum([ tf.nn.l2_loss(v) for v in tv ]) 
            cost = cost + regularization_cost
        tf.summary.scalar("cross_entropy", cost)
    
    with tf.name_scope("auc"):
        auc_value, auc_op = tf.metrics.auc(y, output)
        tf.summary.scalar("aucValue", auc_op)
        tf.summary.scalar("aucValue", auc_value)
    
    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(lr).minimize(cost) 
        
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("/home/buyun/sparseNN/tables/board_sparseNN")
    writer.add_graph(sess.graph)
    
    data = preproc(sparse_field=sparse_field)   
    Xd, Xs, label = data.get_data()
    n = len(Xd)
    
        #dataset=make_dataset(X,label,10)
    eval_auc=[]
    #for X_train,label_train,X_test,label_test in dataset:
        #sess.run(tf.global_variables_initializer())
        #sess.run(tf.local_variables_initializer())
    Xd_test = Xd[: n//setdiv]
    label_test = label[: n//setdiv]
    Xs_train = {}
    Xs_test = {}
    for key in Xs.keys():
        Xs_test[key] = Xs[key][: n//setdiv]
        Xs_train[key] = Xs[key][n//setdiv:]
    Xd_train = Xd[n//setdiv:]
    label_train = label[n//setdiv:]
        
    for i in range(iteration):
        start = (i * batch_size) % n
        end = ((i + 1) * batch_size) % n
        Xs_batch = {}
        if (start < end):
            Xd_batch = Xd_train[start: end]
            for key in Xs_train.keys():
                Xs_batch[key] = Xs_train[key][start: end]
            label_batch = label_train[start: end]
        else:
            Xd_batch = np.vstack((Xd_train[start:], Xd_train[: end]))
            for key in Xs_train.keys():
                Xs_batch[key] = np.vstack((Xs_train[key][start:], Xs_train[key][: end]))
            label_batch = np.vstack((label_train[start:], label_train[: end]))
            
        feed_dict = {}
        feed_dict[y] = label_batch
        feed_dict[d_input] = Xd_batch
        for key in Xs_batch.keys():
            feed_dict[s_input[key]] = Xs_batch[key]

        if i % 5 == 0:
            s = sess.run(merged_summary, feed_dict=feed_dict)
            writer.add_summary(s, i)
#            if i % 500 == 0:
#                sess.run(auc_op, feed_dict=feed_dict)
#                value = sess.run(auc_value, feed_dict=feed_dict)
#                print("step %d, training auc %g" % (i, value))

        sess.run(train_step, feed_dict=feed_dict)
    feed_dict = {}
    feed_dict[y] = label_test
    feed_dict[d_input] = Xd_test
    for key in Xs_test.keys():
        feed_dict[s_input[key]] = Xs_test[key]
    predict = sess.run(output, feed_dict=feed_dict)
    predict = np.ndarray.flatten(predict)
    label_test = np.ndarray.flatten(label_test)
    fpr, tpr, thresholds = metrics.roc_curve(label_test, predict, pos_label=1)
    eval_auc.append(metrics.auc(fpr, tpr))
    print(np.mean(eval_auc))
    return predict, label_test

def build_deep_model(arc, useEmbedding=True, useSparse=True, batch_size=200, lr=0.01, iter=20000):
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
            for i in range(len(arc)):
                if (i == 0):
                    layer[i] = fc_layer(dense_input, 252, arc[i], name="layer{}".format(i))
                else:
                    layer[i] = fc_layer(layer[i - 1], arc[i - 1], arc[i], name="layer{}".format(i))
            
            last_layer = layer[len(arc) - 1]
            pre_out = embedding_layer(last_layer, arc[-1], 1, name="preoutput")
    
            out = tf.sigmoid(pre_out)
            
            return train(dense_input, out, y, batch_size, model_type="deep_model_dense_sparse", lr=lr, iter=iter)
            

def build_wide_model(useSparse=False, batch_size=200, lr=0.01, iter=2000, l2regu=[False, 0.01]):
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
        
def train(input, output, y, batch_size, model_type, lr=0.01, iter=2000, l2regu=[False, 0.01]):
    
    with tf.name_scope("crossEntropy"):
        cost = tf.losses.log_loss(y, output)
        tf.summary.scalar("loss", cost)
        if (l2regu[0]):
            tv = tf.trainable_variables()
            regularization_cost = l2regu[1]* tf.reduce_sum([ tf.nn.l2_loss(v) for v in tv ]) 
            cost = cost + regularization_cost
        #cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output,labels=y))
    
    with tf.name_scope("auc"):
        auc_value, auc_op = tf.metrics.auc(y, output)
        tf.summary.scalar("aucValue", auc_op)
        tf.summary.scalar("aucValue", auc_value)
    
    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(lr).minimize(cost) 
        

    #auc = tf.metrics.auc(y, out)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("/home/buyun/sparseNN/tables/board_deep")
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

            if i % 5 == 0:
                s = sess.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, i)
#            if i % 500 == 0:
#                sess.run(auc_op, feed_dict=feed_dict)
#                value = sess.run(auc_value, feed_dict=feed_dict)
#                print("step %d, training auc %g" % (i, value))
    
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
    return predict, label_test, np.mean(eval_auc)

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

def parwise_sparseNN_model(dense_arc, sparse_arc, sparse_field, arc):
    
    with tf.variable_scope("anchor"):
        dense_input_anchor, sparse_input_anchor, out_anchor, y_anchor = build_denseSparse_model(dense_arc=dense_arc, sparse_arc=sparse_arc, sparse_field=sparse_field, arc=arc)
    with tf.variable_scope("anchor", reuse=True):
        dense_input_mirror, sparse_input_mirror, out_mirror, y_mirror = build_denseSparse_model(dense_arc=dense_arc, sparse_arc=sparse_arc, sparse_field=sparse_field, arc=arc)
    
    pairwise_sparseNN_train(
            dense_input_anchor, 
            sparse_input_anchor, 
            dense_input_mirror, 
            sparse_input_mirror, 
            out_anchor, 
            out_mirror, 
            y_anchor, 
            batch_size=100, 
            sparse_field=sparse_field, 
            setdiv=5, 
            lr=0.01, 
            iteration=4000, 
            l2regu=[False, 0.01]
        )

def pairwise_sparseNN_train(
        d_input_anchor, 
        s_input_anchor, 
        d_input_mirror, 
        s_input_mirror,
        output_anchor,
        output_mirror,
        y, 
        batch_size,
        sparse_field, 
        per=0.8,
        margin=0.3,
        setdiv=10, 
        lr=0.1, 
        iteration=4000,
        l2regu=[False, 0.01]
    ):
    with tf.name_scope("rankLoss"):
        sub = tf.subtract(output_anchor, output_mirror) 
        temp1 = tf.subtract(margin, sub)
        temp2 = tf.multiply(y, temp1)
        rankingLoss = tf.maximum(0.0, temp2)
        cost = tf.reduce_mean(rankingLoss)
        #cost = tf.losses.log_loss(y, output)
        #cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output,labels=y))
#        if (l2regu[0]):
#            tv = tf.trainable_variables()
#            regularization_cost = l2regu[1]* tf.reduce_sum([ tf.nn.l2_loss(v) for v in tv ]) 
#            cost = cost + regularization_cost
        tf.summary.scalar("cross_entropy", cost)
    
    with tf.name_scope("auc"):
        auc_value, auc_op = tf.metrics.auc(y, output_anchor)
        tf.summary.scalar("aucValue", auc_op)
        tf.summary.scalar("aucValue", auc_value)
    
    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(lr).minimize(cost) 
        
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("/home/buyun/sparseNN/tables/board_sparseNN")
    writer.add_graph(sess.graph)
    
    
    data = preproc(sparse_field=sparse_field) 
    Xd_anchor_train, Xs_anchor_train, Xd_mirror_train, Xs_mirror_train, label_train = data.get_pairwise_data(train=True, per=per)
    Xd_anchor_test, Xs_anchor_test, Xd_mirror_test, Xs_mirror_test, label_test = data.get_pairwise_data(train=False, per=per)
    eval_auc=[]
#    Xd_anchor, Xs_anchor, Xd_mirror, Xs_mirror, label = data.get_pairwise_data()
#    n = len(Xd_anchor)
#    
#        #dataset=make_dataset(X,label,10)

#    #for X_train,label_train,X_test,label_test in dataset:
#        #sess.run(tf.global_variables_initializer())
#        #sess.run(tf.local_variables_initializer())
#    Xd_anchor_test = Xd_anchor[: n//setdiv]
#    Xs_anchor_train = {}
#    Xs_anchor_test = {}
#    for key in Xs_anchor.keys():
#        Xs_anchor_test[key] = Xs_anchor[key][: n//setdiv]
#        Xs_anchor_train[key] = Xs_anchor[key][n//setdiv:]
#    Xd_anchor_train = Xd_anchor[n//setdiv:]
#    
#    Xd_mirror_test = Xd_mirror[: n//setdiv]
#    Xs_mirror_train = {}
#    Xs_mirror_test = {}
#    for key in Xs_mirror.keys():
#        Xs_mirror_test[key] = Xs_mirror[key][: n//setdiv]
#        Xs_mirror_train[key] = Xs_mirror[key][n//setdiv:]
#    Xd_mirror_train = Xd_mirror[n//setdiv:]
#    
#    label_test = label[: n//setdiv]
#    label_train = label[n//setdiv:]
    n = len(Xd_anchor_train)    
    for i in range(iteration):
        start = (i * batch_size) % n
        end = ((i + 1) * batch_size) % n
        Xs_anchor_batch = {}
        Xs_mirror_batch = {}
        if (start < end):
            Xd_anchor_batch = Xd_anchor_train[start: end]
            Xd_mirror_batch = Xd_mirror_train[start: end]
            for key in Xs_anchor_train.keys():
                Xs_anchor_batch[key] = Xs_anchor_train[key][start: end]
                Xs_mirror_batch[key] = Xs_mirror_train[key][start: end]
            label_batch = label_train[start: end]
        else:
            Xd_anchor_batch = np.vstack((Xd_anchor_train[start:], Xd_anchor_train[: end]))
            Xd_mirror_batch = np.vstack((Xd_mirror_train[start:], Xd_mirror_train[: end]))
            for key in Xs_anchor_train.keys():
                Xs_anchor_batch[key] = np.vstack((Xs_anchor_train[key][start:], Xs_anchor_train[key][: end]))
                Xs_mirror_batch[key] = np.vstack((Xs_mirror_train[key][start:], Xs_mirror_train[key][: end]))
            label_batch = np.vstack((label_train[start:], label_train[: end]))
            
        feed_dict = {}
        feed_dict[y] = label_batch
        feed_dict[d_input_anchor] = Xd_anchor_batch
        feed_dict[d_input_mirror] = Xd_mirror_batch

        for key in Xs_anchor_batch.keys():
            feed_dict[s_input_anchor[key]] = Xs_anchor_batch[key]
            feed_dict[s_input_mirror[key]] = Xs_mirror_batch[key]

        if i % 5 == 0:
            s = sess.run(merged_summary, feed_dict=feed_dict)
            writer.add_summary(s, i)
#            if i % 500 == 0:
#                sess.run(auc_op, feed_dict=feed_dict)
#                value = sess.run(auc_value, feed_dict=feed_dict)
#                print("step %d, training auc %g" % (i, value))

        sess.run(train_step, feed_dict=feed_dict)
    feed_dict = {}
    feed_dict[y] = label_test
    feed_dict[d_input_anchor] = Xd_anchor_test
    feed_dict[d_input_mirror] = Xd_mirror_test
    for key in Xs_anchor_test.keys():
        feed_dict[s_input_anchor[key]] = Xs_anchor_test[key]
        feed_dict[s_input_mirror[key]] = Xs_mirror_test[key]
    predict = sess.run(output_anchor, feed_dict=feed_dict)
    predict = np.ndarray.flatten(predict)
    label_test = np.ndarray.flatten(label_test)
    fpr, tpr, thresholds = metrics.roc_curve(label_test, predict, pos_label=1)
    eval_auc.append(metrics.auc(fpr, tpr))
    print(np.mean(eval_auc))
    return predict, label_test, np.mean(eval_auc)
   
#dense_input, sparse_input, out, y = build_denseSparse_model(dense_arc=[25, 16], sparse_arc=[32, 16], sparse_field=7, arc=[128, 64]) 
#sparseNN_train(dense_input, sparse_input, out, y, batch_size=500, sparse_field=7, setdiv=10, lr=0.01, iteration=1000, l2regu=[False, 0.01])

#pred, label_t, auc_ = build_wide_model(useSparse=True, batch_size=100, lr=0.01, iter=10000, l2regu=[True, 0.05])


parwise_sparseNN_model(dense_arc=[25, 16], sparse_arc=[37, 16], sparse_field=6, arc=[128, 64]) 
#pred, label_t, auc = build_deep_model(arc=[256,128,32], useSparse=True, batch_size=100, lr=0.01, iter=3000)
#data = preproc(sparse_field=1)   
#Xd, Xs, label = data.get_data()
#print(label)
