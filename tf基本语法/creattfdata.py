import tensorflow as tf
import numpy as np

# from numpy, list
# zeros , ones
# fill
# random
# constant
# Application


# 从一个numpy数据创建tensor数据
mytf = tf.convert_to_tensor(np.ones([2, 3]), dtype=tf.int32)
print('mytf is:', mytf)
# 从一个list创建tensor数据
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
# 使用ones() 或者 zeros() 或者 ones_like() zeros_like()创建tensor数据
mytf3 = tf.zeros([3, 5], dtype=tf.float32, name='input_name')
print('mytf3 is:', mytf3)
print(tf.Variable(mytf3))
print(tf.Variable(mytf3).name)
print(tf.Variable(mytf3).trainable)

mytf4 = tf.ones([4, 3], dtype=tf.float32, name='input_labels')
print('mytf4 is:', mytf4)
print(mytf4.dtype)
print(type(mytf4))

mytf5 = tf.zeros_like(mytf4)  # mytf5 与 mytf4 shape一样创建 即：4 * 3 的矩阵
print('mytf5 is:', mytf5)  # 不同之处是 mytf4 中的标量为1 而 mytf5的标量是 0 。

# fill使用方法 生成的tensor里的每一个元素均相同，因为value为一个固定的标量
mytf6 = tf.fill(dims=[2, 2], value=1.8, name='mytes_fill')
print('mytf6 is:', mytf6)

# 使用随机函数 random :normal正态分布，uniform均匀分布，truncated_normal 截短正态分布
mytf7 = tf.random.normal(shape=[2, 3], mean=1, stddev=1.0)  # mean平均值为1 stddev标准偏差为1
print(mytf7)

mytf8 = tf.random.truncated_normal([2, 2], mean=1.0, stddev=1.0)
print(mytf8)

mytf9 = tf.random.uniform([2, 3], minval=0, maxval=1)
print(mytf9)

# random 随机打散排序shuffle,后面作为图片集的序列用
idx = tf.range(10)
print(tf.random.shuffle(idx))

# 创建10张1行784列的图片组 img_x 及 10个标签img_y
img_x = tf.random.normal([10, 784])
img_y = tf.random.uniform([10], minval=0, maxval=10, dtype=tf.int32)

# 使用tf.gather函数 组合idx 和 img_x img_y
img_x = tf.gather(img_x, idx)
img_y = tf.gather(img_y, idx)
print(img_x)
print(img_y)


print('使用constan 创建tensor的举例说明：')
print('创建标量scalar：', tf.constant(2.35))
print('创建向量：', tf.constant([1, 2]))
print('创建shape为2*3的tensor数据集：', tf.constant(value=2, shape=[2, 3]))

# tf.convert_to_tensor 与 constan 一样用法
# compute loss 创建out 和 y（下面的my_one_hot)
pre = tf.random.uniform([4, 10])
# 独热编码 one_hot 方法
my_one_hot = tf.range(4)
my_one_hot = tf.one_hot(my_one_hot, depth=10)
loss = tf.keras.losses.mse(my_one_hot, pre)
my_reduce_loss = tf.reduce_mean(loss)
print(loss)
print(my_reduce_loss)

# Vector
net = tf.keras.layers.Dense(10)
net.built((4, 8))

print(net.kernel)
print(net.bias)