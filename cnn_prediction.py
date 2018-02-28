# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 19:57:41 2018

@author: Administrator
"""

import tensorflow as tf
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

matlab_train_data = sio.loadmat('F:\image_train.mat')
train_data = matlab_train_data['image_train']
#训练数据集
with tf.name_scope('input_image'):
    train = np.array(train_data,dtype=np.float32)
#train_slice = tf.reshape(train,[-1,128*5,128*6,1])
#    train_slice = tf.slice(train,[0,0,0,0],[-1,-1,-1,3])
#train_slice_image = tf.summary.image('image',train_slice,1)
#matlab_label_data = sio.loadmat('image_train_label.mat')
    train_label_data = matlab_train_data['image_train_label']

#训练数据集标签
train_label = np.array(train_label_data,dtype=np.float64)
matlab_test_data = sio.loadmat('F:\image_test.mat')
test_data = matlab_test_data['image_test']
#测试数据集
test = np.array(test_data,dtype=np.float32)
#matlab_label_data = sio.loadmat('image_test_label')
test_label_data = matlab_test_data['image_test_label']
#测试数据集标签
test_label = np.array(test_label_data,dtype=np.float64)
#第一层卷积层
#输入数据大小[128x128x3]
#核函数大小[5x5x3],共有64个卷积核

def batch_process(train1,train_label1,num,flag):
    if flag == True:
        idx = np.random.randint(0,700,num)
    else:
        idx = np.random.randint(0,104,num)
    tmp_train = []
    tmp_train_label = []
    for i in idx:
        tmp_train.append(train[i])
        tmp_train_label.append(train_label[i])
    return np.array(tmp_train,dtype=np.float32),\
           np.array(tmp_train_label,dtype=np.float32)
           
sess = tf.Session()
imported_meta = tf.train.import_meta_graph('c:\\cnn\\model-5000.meta')
imported_meta.restore(sess,tf.train.latest_checkpoint('c:\\cnn'))
#all_vars = tf.trainable_variables()
sess.run('Convolution_layer1/W_conv1_:0')
graph = tf.get_default_graph()
x = graph.get_tensor_by_name('place_holder/x_input:0')
#y_ = graph.get_tensor_by_name('place_holder/y__input:0')
keep_prob = graph.get_tensor_by_name('Dropout_layer/keep_prob_:0')
fc21 = graph.get_tensor_by_name('Fully_connected_layer2/h_fc21_output:0')
predict_data,_ = batch_process(train,train_label,1,True)
predict_show = np.reshape(np.array(predict_data),[128,128,3])
plt.imshow(predict_show)
plt.show()
rst = sess.run(fc21,feed_dict={x:predict_data,keep_prob:1})
rst1 = rst.tolist()[0]
if rst1.index(max(rst1))==0:
    print('------airplanes------')
elif rst1.index(max(rst1))==1:
    print('------ferry------')
else:
    print('------laptop------')