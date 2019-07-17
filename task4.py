# -*- coding: utf-8 -*-
import math
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from sklearn.preprocessing import OneHotEncoder

# GRADED FUNCTION: read_in_dataset
def read_in_dataset(datadir):
    """
    从路径下读取训练数据和训练标签。
    参数:
    datadir -- 包含训练数据和训练标签文件的目录
    返回:
    train_x -- 训练数据
    train_y -- 训练标签
    """
    ohe = OneHotEncoder()
    train_x = np.load(datadir + 'train_data.npy')
    train_x = np.array(train_x,dtype='float32')
    test_x = np.load(datadir + 'test_data.npy')
    test_x = np.array(test_x,dtype='float32')
    train_y = np.load(datadir + 'train_label.npy')
    test_y = np.load(datadir + 'test_label.npy')
    ohe.fit(np.unique(train_y).reshape(-1,1))
    train_y = ohe.transform(train_y).toarray()
    test_y = ohe.transform(test_y).toarray()
    return train_x,train_y,test_x,test_y  

# GRADED FUNCTION: random_mini_batches
def random_mini_batches(X, Y, mini_batch_size=10, seed=0):
    """
    从 (X, Y) 创建随机的minibatch。
    参数:
    X -- 输入数据
    Y -- 输入标签
    mini_batch_size - 每个minibatch样本数量
    seed -- 仅为模型调优而设，确保每次模型得到的minibatch是一样的
    返回:
    mini_batches -- 数组，每个元素是 (mini_batch_X, mini_batch_Y)这样的元组形式
    """
    m = X.shape[0]#
    mini_batches = []
    np.random.seed(seed)
    # Step 1:打乱顺序
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :, :]
    shuffled_Y = Y[permutation, :]
    # Step 2:取得minibatch
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  #x 
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches

#Step 1:读取文件，创建placeholder
ops.reset_default_graph()
tf.set_random_seed(1)#固定初始化
global_step=tf.Variable(0, trainable=False)

workdir = 'C:/Users/May-Walon/Desktop/'
(train_x, train_y, test_x, test_y) = read_in_dataset(workdir)
input_x_org=tf.placeholder(tf.float32,[None,32,32],name = 'input_x_org')
output_y=tf.placeholder(tf.float32,[None,10],name = 'output_y')
input_x=tf.reshape(input_x_org,[-1,32,32,1],name = 'input_x')
#Step 2:初始化参数
W1 = tf.Variable(tf.random_normal([5,5,1,6]),name = 'W1')
W2 = tf.Variable(tf.random_normal([5,5,6,16]),name = 'W2')
b1 = tf.Variable(tf.random_normal([6]),name = 'b1')
b2 = tf.Variable(tf.random_normal([16]),name = 'b2')
keep_prob = tf.placeholder(tf.float32,name = 'KP')
#Step 3:构造cnn网络
# 前项迭代：
# CONV2D + RELU
Z1 = tf.nn.conv2d(input_x,W1,strides = [1,1,1,1],padding = 'VALID')
A1 = tf.nn.relu(Z1 + b1)
# MAXPOOL
P1 = tf.nn.max_pool(A1,ksize = [1,2,2,1],strides = [1,2,2,1],padding = 'SAME')
# CONV2D + RELU
Z2 = tf.nn.conv2d(P1,W2,strides=[1,1,1,1],padding='VALID')
A2 = tf.nn.relu(Z2 + b2)
# MAXPOOL
P2 = tf.nn.max_pool(A2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
# FLATTEN
P2 = tf.contrib.layers.flatten(P2)
# FULLY-CONNECTED
F1 = tf.layers.dense(inputs=P2,units=120,activation=tf.nn.relu)
D1 = tf.layers.dropout(inputs=F1,rate=keep_prob)
F2 = tf.layers.dense(inputs=D1,units=84,activation=tf.nn.relu)
D2 = tf.layers.dropout(inputs=F2,rate=keep_prob)
F3 = tf.layers.dense(inputs=D2,units=10)
SM = tf.nn.softmax(F3,name = 'SM')

# 损失函数
cost = tf.losses.softmax_cross_entropy(onehot_labels=output_y,logits=F3)
optimizer = tf.train.AdamOptimizer(1e-4).minimize(cost,global_step=global_step)
# 精确度
correct_prediction = tf.equal(tf.argmax(output_y,1),tf.argmax(F3,1))
acc = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#Step 4:运行Session,保存网络
saver = tf.train.Saver()
with tf.Session() as sess:
    init=tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    sess.run(init)
    seed = 3
    #断点续训
    ckpt = tf.train.get_checkpoint_state(workdir + 'model/') #获取checkpoints对象
    if ckpt and ckpt.model_checkpoint_path:##判断ckpt是否为空，若不为空，才进行模型的加载，否则从头开始训练  
        saver.restore(sess,ckpt.model_checkpoint_path)#恢复保存的神经网络结构，实现断点续训 
    step = 0
    while step < 40*48:
        minibatch_cost = 0.
        seed = seed + 1
        minibatches = random_mini_batches(train_x, train_y, 100, seed)
        for minibatch in minibatches:
            # 选择一个minibatch
            (minibatch_x, minibatch_y) = minibatch
            temp_cost,train_opt=sess.run([cost,optimizer],{input_x_org:minibatch_x,output_y:minibatch_y,keep_prob:0.5})
            minibatch_cost = temp_cost / 48
            train_accuracy,step = sess.run([acc,global_step],{input_x_org:train_x,output_y:train_y,keep_prob:0.5})
        print("Step=%d, Train loss=%e,[Train accuracy=%.5f]"%(step/48,minibatch_cost,train_accuracy))
        saver.save(sess,workdir + 'model/train_model',global_step=global_step)
    test_accuracy = sess.run(acc,{input_x_org:test_x,output_y:test_y,keep_prob:1})
    cor = tf.argmax(F3)
    print("Test accuracy=%.5f" % test_accuracy)