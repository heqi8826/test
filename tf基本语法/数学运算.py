import tensorflow as tf
from tensorflow import keras
'''
+ - * / %(取余数） // 取整
@（矩阵相乘）matmul
reduce_mean
max
min
sum
'''
a = tf.fill(dims=[2, 2], value=2.0)
b = tf.ones([2, 2])
print(a + b)
print(a - b)
print(a * b)
print(a / b)
print(a // b)
print(a % b)

# tf.math.log  与 tf.exp

