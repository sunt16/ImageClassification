# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 09:41:23 2018

@author: Administrator
"""
import scipy.io as sio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#导入训练数据
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
           
with tf.name_scope('place_holder'):
    x = tf.placeholder(tf.float32,[None,128,128,3],'x_input')
    y_ = tf.placeholder(tf.float64,[None,3],'y__input')
    
with tf.name_scope('Convolution_layer1'):
    W_conv1 = tf.Variable(tf.truncated_normal(shape=[5,5,3,32],stddev=0.1),\
                          name='W_conv1_')
#    W_conv1_slice = tf.slice(W_conv1,[0,0,0,0],[-1,-1,1,1])
#    W_conv1_slice1 = tf.reshape(W_conv1_slice,[1,5,5,1])
#    W_conv1_slice1_image = \
#    tf.summary.image('Weight',W_conv1_slice1,max_outputs=1)
    b_conv1 = tf.Variable(tf.constant(0.1,shape=[32]),name='b_conv1_')
    conv1 = tf.nn.conv2d(x,W_conv1,strides=[1,1,1,1],padding='SAME')
    h_conv1 = tf.nn.relu(conv1+b_conv1)
    #最大池化
    h_pool1 = tf.nn.max_pool(h_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],\
                         padding='SAME')
#第二层卷积层
#核函数大小为[5x5x32],共有64个卷积核
with tf.name_scope('Convolution_layer2'):
    W_conv2 = tf.Variable(tf.truncated_normal(shape=[5,5,32,64],stddev=0.1),\
                          name='W_conv2_')
    b_conv2 = tf.Variable(tf.constant(0.1,shape=[64]),name='b_conv2_')
    conv2 = tf.nn.conv2d(h_pool1,W_conv2,strides=[1,1,1,1],padding='SAME')
    h_conv2 = tf.nn.relu(conv2+b_conv2)
    #最大池化
    h_pool2 = tf.nn.max_pool(h_conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#第三层全连接层
with tf.name_scope('Fully_connected_layer1'):
    h_pool2_flat = tf.reshape(h_pool2,shape=[-1,32*32*64])
    W_fc1 = tf.Variable(tf.truncated_normal(shape=[32*32*64,1024]),name='W_fc1_')
    b_fc1 = tf.Variable(tf.constant(0.1,shape=[1024]),name='b_fc1_')
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

#dropout层，防止过拟合
with tf.name_scope('Dropout_layer'):
    keep_prob = tf.placeholder(dtype=tf.float32,name='keep_prob_')
    keep_prob_scalar = tf.summary.scalar('dropout',keep_prob)
    h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob=keep_prob)

#第四层全连接层
with tf.name_scope('Fully_connected_layer2'):
    W_fc2 = tf.Variable(tf.truncated_normal(shape=[1024,3],stddev=0.1),name='W_fc2_')
    b_fc2 = tf.Variable(tf.constant(0.1,shape=[3]),name='b_fc2_')
    h_fc2 = tf.matmul(h_fc1_drop,W_fc2)+b_fc2
    h_fc21 = tf.cast(h_fc2,tf.float64,'h_fc21_output')
#输入数据的大小为[128x128x3]


#定义损失函数

with tf.name_scope('Loss'):
#    delta_square = tf.square(y_-h_fc21)
#    loss = tf.reduce_mean(delta_square)
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=h_fc21)
    cross_entropy_loss = tf.reduce_mean(loss)
    loss_scalar = tf.summary.scalar('loss',cross_entropy_loss)
    
#train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(\
#                                              cross_entropy_loss)
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy_loss)
with tf.name_scope('Accuracy'):
    correct_accuracy = tf.equal(tf.argmax(h_fc21,1),tf.argmax(y_,1))
    correct_accuracy = tf.cast(correct_accuracy,tf.float32)
    accuracy = tf.reduce_mean(correct_accuracy)
merged = tf.summary.merge([keep_prob_scalar,loss_scalar])
sess = tf.Session()
train_writer = tf.summary.FileWriter('c:/Deep_learning1/log',sess.graph)
#train_writer.add_graph(tf.get_default_graph())

#with tf.Session() as sess:alizer())
#sess.run()
init_op = tf.initialize_all_variables()
saver = tf.train.Saver()
sess.run(init_op)
for i in range(5000):
    print('Step %s' % i)
#sess.run(tf.global_variables_initi
#        train_step.run(feed_dict={x:train,y_:train_label,keep_prob:0.5})
    train_s,train_label_s = batch_process(train,train_label,10,True)
    summary,_=sess.run([merged,train_step],feed_dict={x:train_s,y_:train_label_s,keep_prob:0.5})
    train_writer.add_summary(summary,i)
    saver.save(sess,'c:\\cnn\\model',global_step=i+1)
    plt.imshow(train_s[1])
    plt.show()
    if i%10==0:
        test_s,test_label_s = batch_process(test,test_label,50,False)
        train_accuray = accuracy.eval(feed_dict={x:test_s,y_:test_label_s,keep_prob:1.0},session=sess)
        print('train accuracy %.6f' % train_accuray)
sess.close()