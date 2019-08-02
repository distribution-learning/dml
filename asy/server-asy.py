import socket  # 导入 socket 模块
from threading import Thread
import threading
import numpy as np
import json
import time
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
import tensorflow as tf
import os

import binascii

import configparser
conf = configparser.ConfigParser()
conf.read("../config.cfg")

server_ip = conf.get("asy", "server_ip")
server_port = int(conf.get("asy", "server_port"))
ADDRESS = (server_ip, server_port)  # 绑定地址
os.environ['CUDA_VISIBLE_DEVICES'] = conf.get("asy", "server_CUDA_VISIBLE_DEVICES")

start_time = 0
global_epoch = int(conf.get("asy", "global_epoch"))
end_time = 0

g_socket_server = None  # 负责监听的socket

g_conn_pool = []  # 连接池

num_nodes = int(conf.get("asy", "num_nodes"))   # client 数量

w_list_global = []

b_list_global = []

loss_global = []

acc_global = []

w_list_client = []

b_list_client = []

loss_client = []

acc_client = []


def load_data():
    data_test_images = np.load("../data/MNIST/test_images.npy")
    data_test_labels = np.load("../data/MNIST/test_labels.npy")
    return data_test_images, data_test_labels


def message_util_client_to_server(client_to_server_json):
    client_to_server_dict = json.loads(client_to_server_json)
    gradient_w = client_to_server_dict['gradient_w']
    gradient_b = client_to_server_dict['gradient_b']
    train_error = client_to_server_dict['train_error']
    loss = client_to_server_dict['loss']
    thread_name = client_to_server_dict['thread_name']
    send_time = client_to_server_dict['local_duration']
    return gradient_w, gradient_b, thread_name, send_time, loss


def message_util_server_to_client(w_update, b_update, flag):
    message_dict = {}
    message_dict['direction_flag'] = flag
    message_dict['w'] = w_update
    message_dict['b'] = b_update
    message_dict_json = json.dumps(message_dict)
    return message_dict_json


def receive_util(client):
    buffer = ""
    has_received_length = 0
    # pack_length = int(client.recv(6).decode('utf-8'))
    pack_length = int(binascii.hexlify(client.recv(4)), 16)
    while has_received_length < pack_length:
        d = client.recv(1024)
        buffer = buffer + d.decode('utf-8')
        has_received_length = has_received_length + len(d)
    return buffer


def init_server():
    """
    初始化服务端
    """
    global g_socket_server
    g_socket_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 创建 socket 对象
    g_socket_server.bind(ADDRESS)
    g_socket_server.listen(5)  # 最大等待数（有很多人理解为最大连接数，其实是错误的）
    print("服务端已启动，等待客户端连接...")


def accept_client():
    """
    接收新连接
    """
    while True:
        client, _ = g_socket_server.accept()  # 阻塞，等待客户端连接
        # 加入连接池
        g_conn_pool.append(client)
        # 给每个客户端创建一个独立的线程进行管理
        thread_accept = Thread(target=message_handle, args=(client, ))
        # 设置成守护线程
        thread_accept.setDaemon(True)
        thread_accept.start()


def message_handle(client):
    """
    消息处理
    """
    global w_global, b_global, start_time, end_time  # 全局变量w,b

    # 接收client发来的消息
    client_to_server_json = receive_util(client)

    # 解析client发来的消息，获得梯度
    gradient_w, gradient_b, thread_name, send_time, loss = message_util_client_to_server(client_to_server_json)
    loss_global.append(loss)
    # recv_time = time.time()
    # recv_duration = recv_time - send_time
    # print("recv_duration: ", recv_duration)
    # 根据异步SGD算法更新server端的参数w,b
    w_global = (np.asarray(w_global) - learning_rate * np.asarray(gradient_w)).tolist()
    b_global = (np.asarray(b_global) - learning_rate * np.asarray(gradient_b)).tolist()

    # server端根据跟新后的参数w,b还有测试数据集计算测试误差
    # 是否还需要计算训练误差，即验证集的问题，如果每个client数据集一样，那么验证集可以一样
    # 如果每个client的数据集不一样，验证集也应该从哥哥数据集中抽取
    w = tf.convert_to_tensor(w_global)
    b = tf.convert_to_tensor(b_global)
    x = tf.placeholder(tf.float32, [None, 784])
    # 根据线性函数预测的值
    y = tf.matmul(x, w) + b
    # 是图片实际对应的值
    y_ = tf.placeholder(tf.float32, [None, 10])
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 取得y得最大概率对应的数组索引来和y_的数组索引对比，如果索引相同，则表示预测正确
    data_test_images, data_test_labels = load_data()

    acc_test = sess.run(accuracy, feed_dict={x: data_test_images, y_: data_test_labels})

    if len(test_error_list) == 0:
        start_time = time.time()
        error_duration_list.append(0)
    else:
        error_duration_list.append(int(round(time.time() - start_time)))

    test_error_list.append(1-float(acc_test))
    print("thread name: ", thread_name, "test_error: ", 1-float(acc_test))

    if len(test_error_list) == global_epoch:
        end_time = time.time()
        total_duration = int(round(end_time - start_time))
        print("total_duration: ", total_duration)

    # test_time = time.time()
    # test_duration = test_time - recv_time
    # print("test_duration: ", test_duration)

    message_server_to_client = message_util_server_to_client(w_global, b_global, 0)
    # 对消息长度进行处理，填充冗余
    bytes_len = len(message_server_to_client.encode('utf-8'))
    bytes_len_send = binascii.unhexlify(f"{bytes_len:0{8}x}")
    message_send = bytes_len_send + message_server_to_client.encode('utf-8')
    # 发送消息给client与计算训练误差、测试误差的先后顺序
    client.sendall(message_send)
    client.close()


def plot_util(list_x, list_y):
    font = FontProperties(fname="/usr/share/fonts/truetype/arphic/uming.ttc", size=20)     # 解决windows环境下画图汉字乱码问题
    # x = error_duration_list
    plt.plot(list_x, list_y, color='green', label='asy')
    # plt.plot(x, hybrid, color='red', label='hybrid')
    # plt.plot(x, asyn, color='blue', label='asyn')
    plt.xlabel(u"time", fontproperties=font)  # 注意指定字体，要不然出现乱码问题
    plt.ylabel(u"test error", fontproperties=font)
    plt.title(u"测试误差随时间的变化", fontproperties=font)
    plt.legend()  # 显示图例
    plt.show()


if __name__ == '__main__':

    # w表示每一个特征值（像素点）会影响结果的权重
    w_global = (np.zeros((784, 10), dtype=float)).tolist()
    b_global = (np.zeros(10, dtype=float)).tolist()

    learning_rate = float(conf.get("syn", "learning_rate"))

    test_error_list = []
    error_duration_list = []

    init_server()
    # 新开一个线程，用于接收新连接
    lock_receive = threading.Lock()
    lock_calculate = threading.Lock()
    thread = Thread(target=accept_client)
    thread.setDaemon(True)
    thread.start()
    # 主线程逻辑
    while True:
        cmd = input("""--------------------------
                输入1:关闭服务端
                输入2:画损失曲线
                """)
        if cmd == '1':
            exit()
        elif cmd == '2':
            plot_util(error_duration_list, test_error_list)
            np.save("../npyfile/asy/test_error_asy.npy", test_error_list)
            np.save("../npyfile/asy/error_duration_asy.npy", error_duration_list)
            np.save("../npyfile/asy/train_loss_asy.npy", loss_global)