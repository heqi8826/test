import tensorflow as tf
import numpy as np

# from numpy, list
# zeros , ones
# fill
# random
# constant
# Application


# 从一个numpy数据转换为tensor数据
mytf = tf.convert_to_tensor(np.ones([2, 3]), dtype=tf.int32)
print('mytf is:', mytf)
mytf1 = tf.convert_to_tensor([
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1]
])
print('mytf1 is:', mytf1)

mytf2 = tf.convert_to_tensor([
    [1],
    [2.3]
])
print('mytf2 is:', mytf2)
# print(mytf2.name)

mytf3 = tf.zeros([3, 5], dtype=tf.float32, name='input_name')
print('mytf3 is:', mytf3)
print(tf.Variable(mytf3))
print(tf.Variable(mytf3).name)
print(tf.Variable(mytf3).trainable)

mytf4 = tf.ones([4, 3], dtype=tf.float32, name='input_labels')
print('mytf4 is:', mytf4)
print(mytf4.dtype)
print(type(mytf4))