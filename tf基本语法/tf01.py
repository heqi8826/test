# tf工具包语法练习
import tensorflow as tf
import numpy as np
# 创建tensor 数据类型 整型、浮点型、双精度浮点型、bool型、字符串型
# constant 创建一个常量，理解为一个普通的tensor数据类型即可
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

#tensor属性 device：tensor程序所在的设备名称
with tf.device('cpu'):
    a = tf.constant(123)
    print(a.device)

# a如果在cpu上 转移到gpu上运行的话如下：
with tf.device('cpu'):
    a = tf.constant([[1, 2, 3], [2, 5, 6], [8, 9, 10]])
# a = a.gpu()
# print(a.device)

# tensor数据类型转换为numpy类型
print(a.numpy())
# 查看tensor数据类型的行列数，形状shape属性
print(a.shape)
# 查看tensor数据类型的纬度ndim属性 或者 tf.rank(tensor)方法
print(a.ndim)
print(tf.rank(a))
# 举例多维的打印纬度
mytf = tf.ones([4, 5, 3], dtype=tf.int32)
print(tf.rank(mytf))
print(mytf)

# 是否为tensor数据方法：tf.is_tensor(tensor) 或 isinstance(tensor,tf.Tensor)
print(tf.is_tensor(mytf))
print(isinstance(mytf, tf.Tensor))
print('---------')
# dtype 属性判断tensor数据类型
print(mytf.dtype)

# tensor 数据类型 转换 方法
mytf = np.arange(5)  # 区别标准的python中的range(10)函数
print(mytf)
print(mytf.dtype)
# np.arange()函数默认为64位整型，转化为32位 可以使用如下：
mytf1 = tf.convert_to_tensor(mytf)  # 将np存储数据类型转化为tensor类型打印出来为int32位
mytf2 = tf.convert_to_tensor(mytf, dtype=tf.int32)  # 将tensor的64位转化位32位
print(mytf2)
print(mytf1)
# another method to change datatype:cast函数
mytf3 = tf.cast(mytf, dtype=tf.float32)  # 转化为浮点型的tensor数据
print(mytf3)
mytf4 = tf.cast(mytf3, dtype=tf.double)  # 浮点型转化为浮点双精度型
print(mytf4)
print(tf.cast(mytf4, dtype=tf.int32))  # 将浮点双精度数据转化整型数据int32
# 转换为bool型数据 与 还原为int32位数据
mybool = tf.constant([0, 1], dtype=tf.int64)
print(mybool)
mybool1 = tf.cast(mybool, dtype=tf.bool)
print(mybool1)
mybool2 = tf.cast(mybool, dtype=tf.int32)
print(mybool2)
print('-------')

