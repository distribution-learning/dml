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

server_ip = conf.get("hybrid", "server_ip_syn1")
server_port = int(conf.get("hybrid", "server_port_syn1"))

os.environ['CUDA_VISIBLE_DEVICES'] = conf.get("hybrid", "server_CUDA_VISIBLE_DEVICES")
ADDRESS = (server_ip, server_port)  # 绑定地址
# ADDRESS = ('127.0.0.1', 65431)  # 绑定地址
# HOST, PORT = '127.0.0.1', 65433  # 上层server的地址
HOST, PORT = conf.get("hybrid", "server_ip_asy"), int(conf.get("hybrid", "server_port_asy"))  # 上层server的地址

g_socket_server = None  # 负责监听的socket

g_conn_pool = []  # 连接池

num_nodes = int(conf.get("hybrid", "num_node_syn1"))  # client 数量
client_conn_num = 0  # 当前链接上的client的数量，作为判断是否要聚合所有client参数的标志
send_flag = 0  # 广播的标志， 当所有参数聚合完成以后，这个值变为1,代表server可以控制线程各自向client发消息
send_flag_not = 0  # 当本次server已经发送完所有client的消息之后，这个值变为0,跳出广播消息的线程
message_global = ""

learning_rate = float(conf.get("hybrid", "learning_rate"))

test_error_list = []

w_list_global = []

b_list_global = []

loss_global = []

acc_global = []

w_list_client = []

b_list_client = []

loss_client = []

acc_client = []


def load_data():
    data_test_images = np.load("data/MNIST/test_images.npy")
    data_test_labels = np.load("data/MNIST/test_labels.npy")
    return data_test_images, data_test_labels


def message_util_client_to_server(client_to_server_json):
    client_to_server_dict = json.loads(client_to_server_json)
    w_update = client_to_server_dict['w']
    b_update = client_to_server_dict['b']
    train_error = client_to_server_dict['train_error']
    loss = client_to_server_dict['loss']
    thread_name = client_to_server_dict['thread_name']
    return w_update, b_update, thread_name, loss


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


def message_util_server_to_client(w_update, b_update, loss, flag, server_no):
    message_dict = {}
    message_dict['server_no'] = server_no
    message_dict['direction_flag'] = flag
    message_dict['w'] = w_update
    message_dict['b'] = b_update
    message_dict['loss'] = loss
    message_dict_json = json.dumps(message_dict)
    return message_dict_json


def socket_util():
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((HOST, PORT))
    return client


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


# 聚合参数，将参数发送到上层server
def calculate():
    while True:
        global client_conn_num, send_flag, message_global
        if client_conn_num == num_nodes:
            aggregate_w_b_loss(w_list_client, b_list_client, loss_client)
            w_list_client.clear()
            b_list_client.clear()
            loss_client.clear()

            # 封装消息发送给上层server
            server_no = 1
            message_server_to_client = message_util_server_to_client(w_list_global[-1],
                                                                     b_list_global[-1], loss_global[-1], 0, server_no)
            bytes_len = len(message_server_to_client.encode('utf-8'))
            bytes_len_send = binascii.unhexlify(f"{bytes_len:0{8}x}")
            message_send = bytes_len_send + message_server_to_client.encode('utf-8')

            # 当同步server聚合完参数后，将消息发往上层server
            server_server_client = socket_util()
            server_server_client.sendall(message_send)
            # 同步server收到上层server发来的消息，直接广播给client
            lock_message.acquire()
            message_global = receive_util(server_server_client)
            lock_message.release()
            server_server_client.close()

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
    global client_conn_num, send_flag_not, send_flag

    # 接收消息并解析出相应的值
    client_to_server_json = receive_util(client)
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
            # 判断当前socket是否关闭，如果关闭代表当前线程的消息发送完
            if client.fileno() == -1:
                break

            bytes_len = len(message_global.encode('utf-8'))
            bytes_len_send = binascii.unhexlify(f"{bytes_len:0{8}x}")
            message_send = bytes_len_send + message_global.encode('utf-8')
            client.sendall(message_send)
            client.close()
            # print("thread name: ", thread_name, "test_error: ", test_error_list[-1])

            # 互斥更新
            lock_update_global.acquire()
            send_flag_not -= 1
            lock_update_global.release()

            # 如果两个线程都发送完
            if send_flag_not == 0:
                send_flag = 0
                break


if __name__ == '__main__':

    lock_update_global = threading.Lock()
    lock_message = threading.Lock()

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