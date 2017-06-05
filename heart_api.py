#!/usr/bin/env python
# -- coding: UTF-8 --
from keras.models import Sequential  # 导入神经网络初始化函数
from keras.layers.core import Dense, Activation  # 导入神经网络层函数、激活函数


class Heart:
    def __init__(self, netfile):
        self.netfile = netfile

    def predict(self, data):
        net = Sequential()  # 建立神经网络
        net.add(Dense(30, input_dim=13))  # 添加输入层（13节点）到隐藏层（10节点）的连接
        net.add(Activation('relu'))  # 隐藏层使用relu激活函数
        net.add(Dense(1, input_dim=30))  # 添加隐藏层（10节点）到输出层（1节点）的连接
        net.add(Activation('sigmoid'))  # 输出层使用sigmoid激活函数
        net.load_weights(self.netfile)
        return {"isSick": net.predict_classes(data), "sickRat": net.predict(data)}
