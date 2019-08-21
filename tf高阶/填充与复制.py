import tensorflow as tf


# pad 填充 tile 复制 broadcast_to
a = tf.reshape(tf.range(9), [3, 3])
print(a)
'''
tf.Tensor(
[[0 1 2]
 [3 4 5]
 [6 7 8]], shape=(3, 3), dtype=int32)
'''
# 第一行上面添加一行
aa = tf.pad(a, [[1, 0], [0, 0]])
print(aa)
aaa = tf.pad(a, [[1, 1], [0, 0]])
print(aaa)
aaaa = tf.pad(a, [[1, 1], [1, 0]])
print(aaaa)
aaaaa = tf.pad(a, [[1, 1], [1, 1]])
print(aaaaa)
'''
tf.Tensor(
[[0 0 0]
 [0 1 2]
 [3 4 5]
 [6 7 8]], shape=(4, 3), dtype=int32)
tf.Tensor(
[[0 0 0]
 [0 1 2]
 [3 4 5]
 [6 7 8]
 [0 0 0]], shape=(5, 3), dtype=int32)
tf.Tensor(
[[0 0 0 0]
 [0 0 1 2]
 [0 3 4 5]
 [0 6 7 8]
 [0 0 0 0]], shape=(5, 4), dtype=int32)
tf.Tensor(
[[0 0 0 0 0]
 [0 0 1 2 0]
 [0 3 4 5 0]
 [0 6 7 8 0]
 [0 0 0 0 0]], shape=(5, 5), dtype=int32)
'''
b = tf.pad(a, [[2, 2], [3, 3]])
print(b)

# image paddings
a = tf.random.normal([4, 28, 28, 3])
b = tf.pad(a, paddings=[[1, 2], [2, 3], [2, 2], [1, 1]])
print(b.shape)

# tf.tile 复制tensor数据 横向 纵向
a = tf.range(9)
# 变化矩阵纬度
a = tf.reshape(a, [3, 3])
print(a)

aa = tf.tile(a, multiples=[1, 2])
print(aa)

aaa = tf.tile(a, multiples=[2, 2])
print(aaa)

