# tf工具包语法练习
import tensorflow as tf
import numpy
# 创建tensor 数据
a = tf.constant(1)
print(a)


b = tf.constant(1.)
print(b)

d = tf.constant(2., dtype=tf.double)
print(d)

e = tf.constant([True, False])
print(e)

f = tf.constant('hello,world')
print(f)

with tf.device('cpu'):
    a = tf.constant(123)
    print(a.device)

g = tf.ones([5, 9])
print('g为：', g)

h = tf.random.uniform([5, 9])



i = tf.random.uniform([5, 9])


j = tf.keras.losses.mse(h, i)
print('均方误差为：', h)
