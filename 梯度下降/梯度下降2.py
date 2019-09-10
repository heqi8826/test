import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
# auther:wangqi
# email:heqi8826@qq.com

'''
mean squared error （MSE）均方差
loss = n个[y-(wx+b)]^2 的和
l2 - norm=
cross entropy loss
典型降低损失函数
binaery multi-class softmax leave it to logistic regression part
'''

x = tf.random.normal([2, 4])
w = tf.random.normal([4, 3])
b = tf.zeros([3])
y = tf.constant([2, 0])
# print(x, w, b, y)

# 求导公式设计计算
with tf.GradientTape() as tape:
    tape.watch([w, b])
    prediction = tf.nn.softmax(x@w + b, axis=1)  # 为何axis设置
    loss = tf.reduce_sum(tf.losses.mse(tf.one_hot(y, depth=3), prediction))

# 梯度变化率
grad = tape.gradient(loss, [w, b])
print(grad)
# grad[0] 和 grad[1]分别指什么 尝试计算 看下结果区别
# 64视频 函数求导过程 2个例子 笔算


