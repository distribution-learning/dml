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

server_ip = conf.get("syn", "server_ip")
server_port = int(conf.get("syn", "server_port"))

os.environ['CUDA_VISIBLE_DEVICES'] = conf.get("syn", "server_CUDA_VISIBLE_DEVICES")
ADDRESS = (server_ip, server_port)  # 绑定地址

start_time = 0
global_epoch = int(conf.get("syn", "global_epoch"))
end_time = 0

g_socket_server = None  # 负责监听的socket

g_conn_pool = []  # 连接池

num_nodes = int(conf.get("syn", "num_nodes"))  # client 数量
client_conn_num = 0  # 当前链接上的client的数量，作为判断是否要聚合所有client参数的标志
send_flag = 0  # 广播的标志， 当所有参数聚合完成以后，这个值变为1,代表server可以控制线程各自向client发消息
send_flag_not = 0  # 当本次server已经发送完所有client的消息之后，这个值变为0,跳出广播消息的线程

learning_rate = float(conf.get("syn", "learning_rate"))

test_error_list = []
error_duration_list = []

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


# def load_cifar10():
#     data_test_images = np.load("../data/cifar10_data/image_test_reshape.npy")
#     data_test_labels = np.load("../data/cifar10_data/label_test.npy")
#     return data_test_images, data_test_labels


def message_util_client_to_server(client_to_server_json):
    client_to_server_dict = json.loads(client_to_server_json)
    w = client_to_server_dict['w']
    b = client_to_server_dict['b']
    train_error = client_to_server_dict['train_error']
    loss = client_to_server_dict['loss']
    thread_name = client_to_server_dict['thread_name']
    return w, b, thread_name, loss


def mean_w(w):
    w_np = np.array(w)
    w_np_mean = np.mean(w_np, axis=0)
    w_list_mean = w_np_mean.tolist()
    return w_list_mean


def aggregate_w_b_loss(w, b, loss):
    w = mean_w(w)
    b = mean_w(b)
    loss = sum(loss_index for loss_index in loss) / num_nodes
    w_list_global.append(w)
    b_list_global.append(b)
    loss_global.append(loss)


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


# 聚合参数，并根据新的参数计算测试误差
def calculate():
    while True:
        global client_conn_num, send_flag, end_time, start_time
        if client_conn_num == num_nodes:
            aggregate_w_b_loss(w_list_client, b_list_client, loss_client)
            w_list_client.clear()
            b_list_client.clear()
            loss_client.clear()

            w = tf.convert_to_tensor(w_list_global[-1])
            b = tf.convert_to_tensor(b_list_global[-1])

            x = tf.placeholder(tf.float32, [None, 784])
            # 根据线性函数预测的值
            y = tf.matmul(x, w) + b
            # 是图片实际对应的值
            y_ = tf.placeholder(tf.float32, [None, 10])

            # x = tf.placeholder(tf.float32, [128, 1728])
            # # 根据线性函数预测的值
            # y = tf.matmul(x, w) + b
            # # 是图片实际对应的值
            # y_ = tf.placeholder(tf.float32, [128, 1])

            sess = tf.InteractiveSession()
            tf.global_variables_initializer().run()

            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            # 取得y得最大概率对应的数组索引来和y_的数组索引对比，如果索引相同，则表示预测正确

            data_test_images, data_test_labels = load_data()
            # data_test_images, data_test_labels = load_cifar10()
            # data_test_labels = data_test_labels.reshape(-1, 1)

            acc_test = sess.run(accuracy, feed_dict={x: data_test_images, y_: data_test_labels})
            test_error_list.append(1 - float(acc_test))

            error_duration_list.append(int(round(time.time() - start_time)))

            if len(test_error_list) == global_epoch:
                end_time = time.time()
                total_duration = int(round(end_time - start_time))
                print("total_duration: ", total_duration)

            client_conn_num = 0  # 聚合完成以后继续监听，直到当前又有两个client连接上
            send_flag = 1  # 聚合完成以后可以广播发送消息了


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
    global client_conn_num, send_flag_not, send_flag, start_time

    # 接收消息并解析出相应的值
    client_to_server_json = receive_util(client)

    if len(test_error_list) == 0:
        start_time = time.time()

    w_recv, b_recv, thread_name, loss = message_util_client_to_server(client_to_server_json)

    # 互斥更新全局变量
    lock_update_global.acquire()
    w_list_client.append(w_recv)
    b_list_client.append(b_recv)
    loss_client.append(loss)
    client_conn_num += 1
    send_flag_not += 1
    lock_update_global.release()

    while True:
        if send_flag == 1:

            # 封装消息广播给client
            message_server_to_client = message_util_server_to_client(w_list_global[-1], b_list_global[-1], 0)
            bytes_len = len(message_server_to_client.encode('utf-8'))
            bytes_len_send = binascii.unhexlify(f"{bytes_len:0{8}x}")
            message_send = bytes_len_send + message_server_to_client.encode('utf-8')

            # 判断当前socket是否关闭，如果关闭代表当前线程的消息发送完
            if client.fileno() == -1:
                break
            client.sendall(message_send)
            client.close()
            print("thread name: ", thread_name, "test_error: ", test_error_list[-1])

            # 互斥更新
            lock_update_global.acquire()
            send_flag_not -= 1
            lock_update_global.release()

            # 如果两个线程都发送完
            if send_flag_not == 0:
                send_flag = 0
                break


def plot_util(list_x, list_y):
    font = FontProperties(fname="/usr/share/fonts/truetype/arphic/uming.ttc", size=20)     # 解决windows环境下画图汉字乱码问题
    plt.plot(list_x, list_y, color='green', label='syn')
    # plt.plot(x, hybrid, color='red', label='hybrid')
    # plt.plot(x, asyn, color='blue', label='asyn')
    plt.xlabel(u"time", fontproperties=font)  # 注意指定字体，要不然出现乱码问题
    plt.ylabel(u"test error", fontproperties=font)
    plt.title(u"测试误差随时间的变化", fontproperties=font)
    plt.legend()  # 显示图例
    plt.show()


if __name__ == '__main__':

    lock_update_global = threading.Lock()

    init_server()

    thread_update = Thread(target=calculate, args=())
    thread_update.setDaemon(True)
    thread_update.start()
    # 新开一个线程，用于接收新连接

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
            np.save("../npyfile/syn/test_error_syn.npy", test_error_list)
            np.save("../npyfile/syn/error_duration_syn.npy", error_duration_list)
            np.save("../npyfile/syn/train_loss_syn.npy", loss_global)