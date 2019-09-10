from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from tensorflow.keras import layers
# MaxPool2D()缩小图片 max_pooling2d ,还存在一个avera_pooling
x = tf.ones([1, 14, 14, 4])
pool = layers.MaxPool2D(2, strides=2)
out = pool(x)
print(out)
pool = layers.MaxPool2D(3, strides=2)
out = pool(x)
print(out)
out = tf.nn.max_pool2d(x, 2, strides=2, padding='VALID')
print(out.shape)

# 图片放大UpSampling2D
x = tf.random.normal([1, 17, 17, 4])
layers = layers.UpSampling2D(size=3)
out = layers(x)
print(out.shape)

# 使用relu函数 在两个工具包都存在relu函数 去掉特征
x = tf.random.normal([1, 7, 7, 3])
print(x)
x = tf.nn.relu(x)
print(x)


class MyModel(Model):

  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(10, activation='softmax')

  def call(self, x):

    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)


model = MyModel()