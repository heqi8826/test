import tensorflow as tf
from tensorflow import keras

# indexing
# Basic indexing:[idx][idx][idx]
a = tf.ones([2, 4, 5, 3])
print(a[0][2][1][2])  # 取最底的标量 需要四次[idx]
# same with Nyumpy 下面举例numpy风格的取法
print(a[0].shape)
print(a[0, 1].shape)
print(a[0, 1, 2].shape)
print(a[0, 1, 3, 1].shape)
'''
打印结果如下：
(4, 5, 3)
(5, 3)
(3,)
()
'''
# tensor切片方法：start:end
b = tf.ones((4, 28, 28, 3))
#
print(b[0, :, :, :].shape)
print(b[0, 1, :, :].shape)
print(b[:, :, :, 0].shape)
print(b[:, :, :, 2].shape)
print(b[:, 0, :, :].shape)
'''
(28, 28, 3)
(28, 3)
(4, 28, 28)
(4, 28, 28)
(4, 28, 3)
'''

# 步阶取样
c = tf.ones((4, 28, 28, 3))
c1 = c[0:2, :, :, :].shape
print('c1的矩阵为', c1)
c2 = c[:, 0:28:2, 0:28:2, :].shape
print('c2的矩阵为', c2)
c3 = c[:, :14, :14, :].shape  # 前14行前14列
print('c3的矩阵为', c3)
c4 = c[:, 14:, 14:, :].shape  # 后14行后14列
print('c4的矩阵为', c4)
c5 = c[:, ::2, ::2, :].shape
print('c5的矩阵为', c5)
c6 = c[:, 2:26:2, 2:26:2, :].shape  # [3,4,5....27] 每2步提取一个:[4,6,8,10,12,14,16,18,20,22,24,26]
print('c6的矩阵为', c6)
'''
c1的矩阵为 (2, 28, 28, 3)
c2的矩阵为 (4, 14, 14, 3)
c3的矩阵为 (4, 14, 14, 3)
c4的矩阵为 (4, 14, 14, 3)
c5的矩阵为 (4, 14, 14, 3)
c6的矩阵为 (4, 12, 12, 3)
'''

# tensor 倒序：::-1
d = tf.range(10)
print(d)
d1 = d[::-1]  # :之后没有','号了，区分切片。从倒序取值,步阶为1
print(d1)
d2 = d[::-2]  # 从倒序取值,步阶为2
print(d2)
d3 = d[2::-1]  # 从索引为2的第三个元素value=2倒序提取 2 1 0 如果为-2 则从 2开始提取 第一个 0 提取第二个
print(d3)
print(d[2::-2])
'''
tf.Tensor([0 1 2 3 4 5 6 7 8 9], shape=(10,), dtype=int32)
tf.Tensor([9 8 7 6 5 4 3 2 1 0], shape=(10,), dtype=int32)
tf.Tensor([9 7 5 3 1], shape=(5,), dtype=int32)
tf.Tensor([2 1 0], shape=(3,), dtype=int32)
tf.Tensor([2 0], shape=(2,), dtype=int32)
'''

# ...用法
e = tf.ones((3, 4, 28, 28, 3))  # 创建一个dim为5的tensor数据
e1 = e[0, ...]
print(e1.shape)
e2 = e[1, ..., 2]
print(e2.shape)




