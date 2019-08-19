import tensorflow as tf
from tensorflow import keras

# shape ndim reshape expand_dims squeeze transpose
a = tf.random.normal([4, 28, 28, 3])
print(a.shape)
print(a.ndim)
print(tf.reshape(a, [4, 784, 3]).shape)
# -1 的特殊含义 ，一个tensor数据类型 以shape形式表示的时候
# 仅可以出现一个-1，系统会自动计算size （几个元素相乘再除以已知数，
# 就等于-1 代表的数
aa = tf.reshape(a, [4, -1, 3])
print(aa.shape)
'''
(4, 784, 3)
'''
# 恢复原来的shape reshape套用
a_original = tf.reshape(tf.reshape(a, [4, -1]), [4, 28, 28, 3])
print(a_original.shape)

# tf.transpose 转置 位置 矩阵横纵变换
b = tf.random.normal((4, 3, 2, 1))
bb = tf.transpose(b)
print(bb.shape)

bbb = tf.transpose(b, perm=[0, 1, 3, 2])  # 将原来的0纬度放在第第一维度
# 原来的1纬度放在第二纬度 原来的三维度放在第三纬度 原来的二纬度放在四纬度
# 可以这样理解：perm内的 为原来的索引 放在新的tensor数据的所在纬度。
print(bbb.shape)
print('-------')
# 增加维度 减少维度
# expand dim 增加纬度
a = tf.random.normal([4, 35, 8])  # 班级 学生 科目
# 增加一个dim 学校 [1, 4, 35, 8]
aa = tf.expand_dims(a, axis=0)
print(aa.shape)
# 增加在第4个纬度 也就是索引为3 的位置 则将axis赋值为3
aaa = tf.expand_dims(a, axis=3)
print(aaa.shape)

# 如果给负轴 则注意如下规则
aaaa = tf.expand_dims(a, axis=-1)  # 最右边-1
print(aaaa.shape)

aaaaa = tf.expand_dims(a, axis=-4)  # 最右边-1 依次向左 -2 -3 -4
print(aaaaa.shape)

# squeeze  减少某个纬度 只能减少 dim为1的。
b = tf.squeeze(aaaa, axis=-1)
print(b.shape)
