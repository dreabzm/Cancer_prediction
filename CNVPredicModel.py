#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 22:36:06 2018

@author: buyun
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import numpy as np
import tensorflow as tf
from Feature_preprocessing import preproc

def embedding_layer(input, dim_in, dim_out):
    w = tf.Variable(tf.zeros([dim_in, dim_out]))
    b = tf.Variable(tf.zeros([dim_out]))
    out_put = tf.matmul(input, w) + b
    return out_put

def fc_layer(input, dim_in, dim_out):
    w = tf.Variable(tf.zeros([dim_in, dim_out]))
    b = tf.Variable(tf.zeros([dim_out]))
    act = tf.nn.relu(tf.matmul(input, w) + b)
    return act

def build_model(dense_arc, sparse_arc, sparse_field, arc, batch_size=500):
    assert dense_arc[len(dense_arc) - 1] == sparse_arc[len(sparse_arc) - 1], \
        "last dimention of dense arch and sparse arch must be the same."
    
    y = tf.placeholder(tf.float32, shape=[None, 1])
    y = tf.reshape(y, [-1, 1])
    
    dense_input = tf.placeholder(tf.float32, shape=[None, dense_arc[0]])
    dense_input = tf.reshape(dense_input, [-1, dense_arc[0]])
    sparse_input = {}
    for i in range(sparse_field):
        sparse_input[i] = tf.placeholder(tf.float32, shape=[None, sparse_arc[0]])
        sparse_input[i] = tf.reshape(sparse_input[i], [-1, sparse_arc[0]])
    
    dense_net = []
    for i in range(len(dense_arc) - 1):
        if (i == 0):
            dense_net.append(fc_layer(dense_input, dense_arc[i], dense_arc[i + 1]))
        else:
            dense_net.append(fc_layer(dense_net[i - 1], dense_arc[i], dense_arc[i + 1]))
    
    sparse_embedding = []        
    for i in range(sparse_field):
        sparse_embedding.append(embedding_layer(sparse_input[i], sparse_arc[0], sparse_arc[1]))
        
    #initialize over all input
    fc_all = dense_net[len(dense_net) - 1]
    #pairwsise dot
    sparse_embedding.append(dense_net[len(dense_net) - 1])
    count = 1
    for i in range(sparse_field + 1):
        for j in range(i + 1, sparse_field + 1):
            dot_product = tf.multiply(sparse_embedding[i], sparse_embedding[j])
            fc_all = tf.concat([dot_product, fc_all], -1)
            count = count + 1
    
    fc = []
    for i in range(len(arc)):
        if (i == 0):
            fc.append(fc_layer(fc_all, count * dense_arc[len(dense_arc) - 1], arc[i]))
        else:
            fc.append(fc_layer(fc[i - 1], arc[i - 1], arc[i]))
            
    out = fc[-1]
    out = tf.convert_to_tensor(out)
    y = tf.convert_to_tensor(y)
    out = tf.nn.sigmoid(out)
    cost = tf.losses.log_loss(y, out)
    #cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=out,labels=y))
    train_step = tf.train.AdamOptimizer(0.1).minimize(cost)
    auc_value, auc_op = tf.metrics.auc(y, out)
    #auc = tf.metrics.auc(y, out)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    writer = tf.summary.FileWriter("/home/buyun/sparseNN/tables/board")
    writer.add_graph(sess.graph)
    
    data = preproc(sparse_field=sparse_field)   
    Xd, Xs, label = data.get_data()
    n = len(Xd)
    
    for i in range(100):
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
        if i % 500 == 0:
            sess.run(auc_op, feed_dict=feed_dict)
            value = sess.run(auc_value, feed_dict=feed_dict)
            print("step %d, training auc %g" % (i, value))

        sess.run(train_step, feed_dict=feed_dict)

build_model(dense_arc=[25, 16], sparse_arc=[227, 16], sparse_field=1, arc=[64, 64, 1])
