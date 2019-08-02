#!/usr/bin/env python
# -*- coding:utf-8 -*-

# 客户端
import socket
import threading
import time

import tensorflow as tf
from sklearn.utils import shuffle  # 随机打乱工具，将原有序列打乱，返回一个全新的顺序错乱的值
import numpy as np
import json
import random

import binascii
import os

import configparser
conf = configparser.ConfigParser()
conf.read("../config.cfg")

os.environ['CUDA_VISIBLE_DEVICES'] = conf.get("syn", "client1_CUDA_VISIBLE_DEVICES")


def load_data():
    data_train_images = np.load("../data/MNIST/train_images.npy")
    data_train_labels = np.load("../data/MNIST/train_labels.npy")
    # data_test_images = np.load("data/MNIST/test_images.npy")
    # data_test_labels = np.load("data/MNIST/test_labels.npy")
    data_validation_images = np.load("../data/MNIST/validation_images.npy")
    data_validation_labels = np.load("../data/MNIST/validation_labels.npy")

    # 根据label的值分组，list中有10个数组，分别代表0-9十个数字，每个数组中存着该数字对应的下标
    train_index = np.load("../data/MNIST/train_index.npy")
    # test_index = np.load("data/MNIST/test_index.npy")
    return data_train_images, data_train_labels, data_validation_images, data_validation_labels, train_index


# def load_cifar10():
#     data_train_images = np.load("../data/cifar10_data/image_train_reshape.npy")
#     data_train_labels = np.load("../data/cifar10_data/label_train.npy")
#     return data_train_images, data_train_labels


def data_sample(sample_size_local):
    m = data_train_labels.shape[0]
    if sample_type == 1:  # 按照数据的index进行随机抽样
        sample_index = random.sample(list(range(0, m)), sample_size_local)
        # data_test_x = data_test_images
        # data_test_y = data_test_labels
    # elif sample_type == 2:  # 按照label进行抽样
    #     sample_index = random.sample(train_index[sample_label], sample_size_local)
    #     # sample_index_test = random.sample(test_index[sample_label], sample_size_local)
    #     # data_test_x = data_test_images[sample_index_test]
    #     # data_test_y = data_test_labels[sample_index_test]
    data_train_x = data_train_images[sample_index]
    data_train_y = data_train_labels[sample_index]
    return data_train_x, data_train_y


def sample_sequence_no_duplication(total, i, sample_size):
    train_size = data_train_labels.shape[0]
    thread_size = int(train_size / total)
    index_start = (i-1) * thread_size
    index_end = index_start + thread_size
    data_train_x_thread = data_train_images[index_start:index_end]
    data_train_y_thread = data_train_labels[index_start:index_end]
    sample_index = random.sample(list(range(0, thread_size)), sample_size)
    data_train_x = data_train_x_thread[sample_index]
    data_train_y = data_train_y_thread[sample_index]
    return data_train_x, data_train_y


def sample_sequence_half_duplication(total, i, sample_size):
    train_size = data_train_labels.shape[0]
    thread_size = int(train_size / (total+1)) * 2
    index_start = (i-1) * int(thread_size / 2)
    index_end = index_start + thread_size
    data_train_x_thread = data_train_images[index_start:index_end]
    data_train_y_thread = data_train_labels[index_start:index_end]
    sample_index = random.sample(list(range(0, thread_size)), sample_size)
    data_train_x = data_train_x_thread[sample_index]
    data_train_y = data_train_y_thread[sample_index]
    return data_train_x, data_train_y


# def data_sample_sequence(sample_size_local, thread_name): # 按数据集原始顺序，从前往后不均匀抽样
#     if thread_name == "t1":
#         data_train_x = data_train_images[0:9999]
#         data_train_y = data_train_labels[0:9999]
#     elif thread_name == "t2":
#         data_train_x = data_train_images[10000:19999]
#         data_train_y = data_train_labels[10000:19999]
#     elif thread_name == "t3":
#         data_train_x = data_train_images[20000:29999]
#         data_train_y = data_train_labels[20000:29999]
#     elif thread_name == "t4":
#         data_train_x = data_train_images[30000:]
#         data_train_y = data_train_labels[30000:]
#     m = data_train_y.shape[0]
#     sample_index = random.sample(list(range(0, m)), sample_size_local)
#     data_train_x = data_train_x[sample_index]
#     data_train_y = data_train_y[sample_index]
#     return data_train_x, data_train_y


# def data_sample_number(sample_size_local, thread_name): # 根据标签值进行抽样
#     if thread_name == "t1":
#         index_thread = train_index[0]
#         index_thread.extend(train_index[1])
#     elif thread_name == "t2":
#         index_thread = train_index[2]
#         index_thread.extend(train_index[3])
#     elif thread_name == "t3":
#         index_thread = train_index[4]
#         index_thread.extend(train_index[5])
#     elif thread_name == "t4":
#         index_thread = train_index[6]
#         index_thread.extend(train_index[7])
#         index_thread.extend(train_index[8])
#         index_thread.extend(train_index[9])
#     data_train_x = data_train_images[index_thread]
#     data_train_y = data_train_labels[index_thread]
#     m = data_train_y.shape[0]
#     sample_index = random.sample(list(range(0, m)), sample_size_local)
#     data_train_x = data_train_x[sample_index]
#     data_train_y = data_train_y[sample_index]
#     return data_train_x, data_train_y


def socket_util():
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((HOST, PORT))
    return client


def receive_util(client):
    buffer = ""
    has_received_length = 0
    # 收到的数据包中，前六个字节固定是包的长度，根据这个长度循环接收整个包
    # pack_length = int(client.recv(6).decode('utf-8'))
    pack_length = int(binascii.hexlify(client.recv(4)), 16)
    while has_received_length < pack_length:
        d = client.recv(1024)
        buffer = buffer + d.decode('utf-8')
        has_received_length = has_received_length + len(d)
    return buffer


# 封装从client发送到第一层server的参数, dict---->json
# client <-----> server-1 server-2: 1
# server <-----> server-1 server-2: 0
# {direction_flag: , w: , b: , acc: , loss: , sample_size, local_epoch: , local_duration: }
def message_cs(direction_flag, thread_name, w_send, b_send, train_error, loss, sample_size_local, local_epoch, local_duration):
    message_dict = {}
    message_dict['direction_flag'] = direction_flag
    message_dict['thread_name'] = thread_name
    message_dict['w'] = w_send
    message_dict['b'] = b_send
    message_dict['train_error'] = train_error
    message_dict['loss'] = loss
    message_dict['sample_size_local'] = sample_size_local
    message_dict['local_epoch'] = local_epoch
    message_dict['local_duration'] = local_duration
    message_json_bytes = json.dumps(message_dict).encode('utf-8')
    return message_json_bytes


def time_util(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    duration = str(int(hours)) + ":" + str(int(minutes)) + ":" + str(int(round(seconds)))
    return duration


def tf_util(w0temp, b0temp, acc, loss):
    loss = float(loss)
    loss_list.append(loss)

    train_error = 1 - float(acc)
    train_error_list.append(train_error)

    b0temp = b0temp.tolist()  # 上一步拿到的b0temp是numpy的float类型，必须转成python的float类型才能用json转换
    b_list.append(b0temp)  # 每全局迭代一轮添加一次

    w0temp = w0temp.tolist()  # 上一步拿到的w0temp是numpy.ndarray类型，必须转成python的list才能用json转换
    w_list.append(w0temp)  # 每全局迭代一轮添加一次

    return w0temp, b0temp, train_error, loss


def send_message(thread_name):
    global_start_time = time.time()

    x = tf.placeholder(tf.float32, [None, 784])
    # w表示每一个特征值（像素点）会影响结果的权重
    w = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    # 根据线性函数预测的值
    y = tf.matmul(x, w) + b
    # 是图片实际对应的值
    y_ = tf.placeholder(tf.float32, [None, 10])

    # x = tf.placeholder(tf.float32, [128, 1728])
    # # w表示每一个特征值（像素点）会影响结果的权重
    # w = tf.Variable(tf.zeros([1728,1]))
    # b = tf.Variable(tf.zeros([128,1]))
    # # 根据线性函数预测的值
    # y = tf.matmul(x, w) + b
    # # 是图片实际对应的值
    # y_ = tf.placeholder(tf.float32, [128,1])

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # 迭代训练
    local_epoch = int(conf.get("syn", "local_epoch"))
    sample_size_local = int(conf.get("syn", "sample_size_local"))
    num_node = int(conf.get("syn", "num_node"))
    client1_sample_number = int(conf.get("syn", "client1_sample_number"))
    for g_epoch in range(global_epoch):

        local_start_time = time.time()
        # local_duration_real = local_start_time - local_start_time_real
        # if g_epoch != 0:
        #     print("g_epoch: ", g_epoch, "train_transport_time: ", local_duration_real)

        for l_epoch in range(local_epoch):
            data_train_x, data_train_y = data_sample(sample_size_local)
            # data_train_x, data_train_y = sample_sequence_no_duplication(num_node, client1_sample_number, sample_size_local)
            # data_train_x, data_train_y = sample_sequence_half_duplication(num_node, client1_sample_number, sample_size_local)

            # data_train_y = data_train_y.reshape(-1, 1)
            
            _, loss = sess.run([train_step, cross_entropy], feed_dict={x: data_train_x, y_: data_train_y})
            # 打乱数据顺序，防止按原次序假性训练输出
            # data_train_x, data_train_y = shuffle(data_train_x, data_train_y)

        local_end_time = time.time()
        local_duration = local_end_time - local_start_time
        # print("train_time: ", local_duration)

        # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # # 取得y得最大概率对应的数组索引来和y_的数组索引对比，如果索引相同，则表示预测正确
        # acc_train = sess.run(accuracy, feed_dict={x: data_validation_images, y_: data_validation_labels})


        b0temp = b.eval(session=sess)  # 训练中当前变量b值
        w0temp = w.eval(session=sess)  # 训练中当前权重w值
        acc_train = 0.5
        w_send, b_send, train_error, loss = tf_util(w0temp, b0temp, acc_train, loss)

        print("thread name:", thread_name, "before send global epoch:", g_epoch + 1,
              # "w:", w0temp, "b:", b0temp,
              "train_error:", train_error, "loss:", loss,
              "sample_size_local:", sample_size_local, "local_epoch:", local_epoch, "local_duration:", local_duration)
        # client 主动向 server发消息
        message_send = message_cs(1, thread_name, w_send, b_send, train_error, loss, sample_size_local,
                                  local_epoch, local_duration)
        bytes_len = len(message_send)
        print(bytes_len)
        bytes_len_send = binascii.unhexlify(f"{bytes_len:0{8}x}")
        # 前六个字节是这次消息中参数的长度
        message_send_length = bytes_len_send + message_send

        # if thread_name == "t3":
        #     time.sleep(0.5)
        # elif thread_name == "t4":
        #     time.sleep(0.5)

        client = socket_util()
        client.sendall(message_send_length)
        string = receive_util(client)
        client.close()

        # server向client发消息的时候会触发这个方法
        message_sc_dict = json.loads(string)

        sess.run(tf.assign(w, message_sc_dict['w']))
        sess.run(tf.assign(b, message_sc_dict['b']))

    global_end_time = time.time()
    global_duration = time_util(global_start_time, global_end_time)
    print("global_duration:", global_duration)
    sess.close()


if __name__ == '__main__':
    # 设置训练超参数
    # 迭代轮次
    global_epoch = int(conf.get("syn", "global_epoch"))
    # local_epoch = 10

    # sample_size = 10  # 抽样数据集的大小
    sample_type = 1  # 抽样的方式
    sample_label = 0  # 抽样的手写字体的数字

    # 学习率
    learning_rate = float(conf.get("syn", "learning_rate"))

    # server 的ip和port
    # HOST, PORT = '127.0.0.1', 65433
    # HOST, PORT = '47.102.41.81', 65433
    HOST, PORT = conf.get("syn", "server_ip"), int(conf.get("syn", "server_port"))
    # 存储每轮全局迭代的中间结果
    loss_list = []  # 用于保存loss值的列表
    b_list = []  # 用于保存b值
    w_list = []  # 用于保存w值
    train_error_list = []  # 用于保存准确率
    gradient_w_list = []
    gradient_b_list = []

    # 从npy文件中快速加载数据
    data_train_images, data_train_labels, data_validation_images, data_validation_labels, train_index = load_data()
    # data_train_images, data_train_labels = load_cifar10()

    # lock_socket = threading.Lock()
    # lock_tf = threading.Lock()

    # num_node_syn = int(conf.get("syn", "num_node_syn"))
    # for i in range(0, num_node_syn):
    #     # 创建线程
    #     try:
    #         t = threading.Thread(target=send_message, args=("t1-" + str(i+1),))
    #         t.start()
    #     except:
    #         print("Error: unable to start thread")

    t1 = threading.Thread(target=send_message, args=('t1', ))
    t1.start()

    # t2 = threading.Thread(target=send_message, args=('t2', ))
    # t2.start()
    #
    # t3 = threading.Thread(target=send_message, args=('t3',))
    # t3.start()
    #
    # t4 = threading.Thread(target=send_message, args=('t4',))
    # t4.start()
    #
    t1.join()
    # t2.join()
    # t3.join()
    # t4.join()

    print("main finish")


