import tensorflow as tf
from tensorflow import keras


# Vector
net = keras.layers.Dense(10)
net.build((4, 8))  # 随机创建了4*8的tensor类型的矩阵 用于放到net降低纬度计算层
# 该句命令等同于：
# test_tensor = tf.random.normal([4,8])
# net(test_tensor)
print(net.kernel)  # net的内核因子 ，打印出来是一个 [8,10]的矩阵，
# 也就是 net.build((4,8))引入了一个随机创建的矩阵与该内核[8,10]相乘
print(net.bias)  # tensor.bias 属性 初始化元素为0的tensor矩阵
# 如果不理解，查阅 matrix.py文件例子。
