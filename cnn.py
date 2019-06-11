#! /usr/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf 
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
x = tf.placeholder(tf.float32, [None, 6084])
y_actual = tf.placeholder(tf.float32, shape=[None, 20])


X_train = pd.read_csv(
    'mfcctrainnl.csv'
).as_matrix()

Y_train= pd.read_csv(       #注意读取数据时第一行没有，所以处理数据文档时把第一行空出
    'trainlabelnet.csv'
).as_matrix()


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

with tf.name_scope('layer1') as scope:
    x_image = tf.reshape(x, [-1,78,78,1])                                             #转换输入数据shape,大小20*20，1表示颜色通道数目
    W_conv1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 6],dtype=tf.float32, stddev=0.1),name='W_conv1')                                      #按照[5,5,输入通道=1,输出通道=6]生成一组随机变量
    b_conv1 = tf.Variable(tf.constant(0.1,shape=[6],dtype=tf.float32),trainable=True,name='b_conv1')
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)                          #第一个卷积层
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')                                                       #池化输出size 10*10*6

with tf.name_scope('layer2') as scope:
    W_conv2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 6, 16],dtype=tf.float32, stddev=0.1),name='W_conv2')                                          #把h_pool1的厚度由6增加到16
    b_conv2 = tf.Variable(tf.constant(0.1,shape=[16],dtype=tf.float32),trainable=True,name='b_conv2')
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)                          #第二卷积层
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

with tf.name_scope('layer3') as scope:
    W_conv3 = tf.Variable(tf.truncated_normal(shape=[5, 5, 16, 40],dtype=tf.float32, stddev=0.1),name='W_conv3')                                          #把h_pool1的厚度由6增加到16
    b_conv3 = tf.Variable(tf.constant(0.1,shape=[40],dtype=tf.float32),trainable=True,name='b_conv3')
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)                          #第二卷积层
    h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

with tf.name_scope('layer4') as scope:
    W_fc1 = tf.Variable(tf.truncated_normal(shape=[7 * 7 * 40, 1040],dtype=tf.float32, stddev=0.1),name='W_fc1')                                          #把h_pool1的厚度由6增加到16
    b_fc1 = tf.Variable(tf.constant(0.1,shape=[1040],dtype=tf.float32),trainable=True,name='b_fc1')
    h_conv2_flat = tf.reshape(h_pool3, [-1, 7 * 7 * 40])                                  #把h_conv2变成向量形式
    h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)                                      #dropout层按照keep_prob的概率扔掉一些，为了减少过拟合

with tf.name_scope('layer5') as scope:
    W_fc2 = tf.Variable(tf.truncated_normal(shape=[1040, 600],dtype=tf.float32, stddev=0.1),name='W_fc2')                                          #把h_pool1的厚度由6增加到16
    b_fc2 = tf.Variable(tf.constant(0.1,shape=[600],dtype=tf.float32),trainable=True,name='b_fc2')
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

with tf.name_scope('layer6') as scope:
    W_fc3 = tf.Variable(tf.truncated_normal(shape=[600, 20],dtype=tf.float32, stddev=0.1),name='W_fc3')                                          #把h_pool1的厚度由6增加到16
    b_fc3 = tf.Variable(tf.constant(0.1,shape=[20],dtype=tf.float32),trainable=True,name='b_fc3')
    y_predict=tf.nn.softmax(tf.matmul(h_fc2, W_fc3) + b_fc3)

cross_entropy = -tf.reduce_sum(y_actual*tf.log(y_predict))                        #交叉熵
train_step1 = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)      #梯度下降法
correct_prediction = tf.equal(tf.argmax(y_predict,1), tf.argmax(y_actual,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))                   #精确度计算

           
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
for i in range(1000):
    trainnum = np.random.randint(0, 24502, 100)
    valinnum = np.random.randint(0, 24502, 100)
    trainbatch = np.zeros([100, 156 * 39])
    labelbatch = np.zeros([100, 20])
    for j in range(0, 100):
        trainbatch[j] = X_train[trainnum[j]]
        labelbatch[j] = Y_train[trainnum[j]]

    train_acc = accuracy.eval(feed_dict={x: trainbatch, y_actual: labelbatch, keep_prob: 1.0})
    # if train_acc<0.6:
    #     train_step1.run(feed_dict={x: trainbatch, y_actual: labelbatch, keep_prob: 0.8})
    # else:
    #     train_step2.run(feed_dict={x: trainbatch, y_actual: labelbatch, keep_prob: 0.8})
    train_step1.run(feed_dict={x: trainbatch, y_actual: labelbatch, keep_prob: 0.8})
    if i % 100 == 0:
        print ('step %d, Training Accuracy %g' % (i, train_acc))

X_test = pd.read_csv(
    'mfcctestnl.csv'
).as_matrix()

Y_test= pd.read_csv(       #注意读取数据时第一行没有，所以处理数据文档时把第一行空出
    'testlabelnet.csv'
).as_matrix()

test_acc=accuracy.eval(feed_dict={x: X_test, y_actual: Y_test, keep_prob: 1.0})
print ("Testing Accuracy %g"%test_acc)

#saver = tf.train.Saver({'W_conv1':W_conv1,'W_conv2':W_conv2,'W_conv3':W_conv3,'b_conv1':b_conv1,'b_conv2':b_conv2,'b_conv3':b_conv3})
saver = tf.train.Saver()

saver.save(sess, 'module/cnn.ckpt')


# with open('CNN_x_decode.csv', 'w') as f:
#   data1 = pd.DataFrame(testdata)
#   data1.to_csv(f)