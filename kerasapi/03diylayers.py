import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# 自定义Dense层 必须继承到 layers.Layer，自定义的层函数中必须有两个方法：__init__初始化方法和 call方法
class MyDense(layers.Layer):  # 继承到layers.Layer
    def __init__(self, inp_dim, outp_dim):
        super(MyDense, self).__init__()  # 调用母类初始化，为必须有的。

        # 以下为DIY的部分 自定义要用到的参数
        self.kernel = self.add_variable('w', [inp_dim, outp_dim])
        self.bias = self.add_variable('b', [outp_dim])

    def call(self, inputs, training=None): # training 定义跑training的函数 而不是testing的函数。
        # 以往使用系统函数 使用model(x)函数调用的时候 等价于 model.__call__(x)

        out = inputs @ self.kernel + self.bias  # diy函数公式

        return out


# 自定义Model模型 也是必须有__init__ 和 call 函数，必须继承到keras.Model
class MyModel(keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = MyDense(28*28, 256)  # 使用了自定义的layer函数
        self.fc2 = MyDense(256, 128)
        self.fc3 = MyDense(128, 64)
        self.fc4 = MyDense(64, 32)
        self.fc5 = MyDense(32, 10)
        # 五层函数降低到纬度为10

    def call(self, inputs, training=None):
        x = tf.reshape(inputs, [-1, 32*32*3])
        x = self.fc1(x)
        x = tf.nn.relu(x)
        x = self.fc2(x)
        x = tf.nn.relu(x)
        x = self.fc3(x)
        x = tf.nn.relu(x)
        x = self.fc4(x)
        x = tf.nn.relu(x)
        x = self.fc5(x)
        return x




