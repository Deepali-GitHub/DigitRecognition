# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 22:49:25 2017

@author: dgarg
"""
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


#nearest neighbor classifier .
Xtrain, Ytrain= mnist.train.next_batch(50000)
Xtest, Ytest = mnist.test.next_batch(200)

#model
xinput = tf.placeholder(tf.float32, [None, 784])
xtest = tf.placeholder(tf.float32, [784])

distance = tf.reduce_sum(tf.abs(tf.add(xinput, tf.negative(xtest))), reduction_indices=1)
pred = tf.argmin(distance, 0)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
accuracy =0
for i in range(len(Xtest)):
    idx_pred = sess.run(pred, {xinput:Xtrain, xtest:Xtest[i, :]})
    ylabel = np.argmax(Ytest[i])
    ypred = np.argmax(Ytrain[idx_pred])
    print ('Test', i, 'ypred', ypred, 'ylabel', ylabel )
    if(ylabel == ypred):
        accuracy +=1./len(Xtest)

print ('Accuracy: ', accuracy)