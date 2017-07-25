# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 14:54:41 2017

@author: dgarg
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 12:32:15 2017

@author: dgarg
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])


#ANN -- artificial neural network

#First layer
w1 = tf.Variable(tf.truncated_normal([784, 784], stddev=0.1))
b1 = tf.Variable(tf.truncated_normal([784], stddev = 0.1))

z1 = tf.nn.sigmoid(tf.matmul(x, w1) +b1)

#Second layer
w2 = tf.Variable(tf.truncated_normal([784, 784], stddev=0.1))
b2 = tf.Variable(tf.truncated_normal([784], stddev = 0.1))

z2 = tf.nn.sigmoid(tf.matmul(z1, w2) +b2)

#Output layer
w3 = tf.Variable(tf.truncated_normal([784, 10], stddev=0.1))
b3 = tf.Variable(tf.truncated_normal([10], stddev = 0.1))

z3 = tf.nn.sigmoid(tf.matmul(z2, w3) +b3)

#evaluate
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=z3))

reguralization = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3) 
cross_entropy = cross_entropy + 0.00001*reguralization

train_step = tf.train.AdagradOptimizer(0.01).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(z3, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
      train_accuracy = accuracy.eval(feed_dict={
          x: batch[0], y_: batch[1]})
      print('step %d, training accuracy %g' % (i, train_accuracy))
      print sess.run(w1)
      
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

  print('test accuracy %g' % accuracy.eval(feed_dict={
      x: mnist.test.images, y_: mnist.test.labels}))