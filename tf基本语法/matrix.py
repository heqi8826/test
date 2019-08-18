import tensorflow as tf
from tensorflow import keras


# 随机创建一个 4个图片 784列的一维度图片集
x = tf.random.normal(shape=(4, 784))
print(x)
print('\n')
# 创建神经网络层layers
net = keras.layers.Dense(10)
net.build((4, 784))  # 个人理解 为Dense构建了一个模版为[4,784]tensor向[1,10]的层降低纬度的方法

# 将x图片集 放入 net 函数降低梯度输出4个图片
print('-----------\n')
print(net(x))

# kernel 和 bias
print('-----------\n')
print(net.kernel.shape)
print(net.bias.shape)
