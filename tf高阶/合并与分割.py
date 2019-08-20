import tensorflow as tf
from tensorflow import keras
'''
tf.concat 合并  tf.split 切割 tf.stack 堆叠  tf.unstak
'''

a = tf.ones([4, 35, 8])
b = tf.ones([2, 35, 8])
# tf.concat拼接合并两个tensor a 与 b ,concat 内有两个参数 合并的两个tensor 和 合并的纬度索引值
c = tf.concat([a, b], axis=0)
print(c.shape)
a = tf.ones([4, 32, 8])
b = tf.ones([4, 3, 8])
d = tf.concat([a, b], 1)
print(d.shape)

# stack 创建新dim纬度，增加纬度 把两个tensor堆叠起来
a = tf.ones([4, 35, 8])
b = tf.ones([4, 35, 8])
e = tf.stack([a, b], 0)  # a,b 所有纬度相同
print(e.shape)

# unstak 指定 一个纬度 在此纬度上 全部打散
a = tf.ones([4, 35, 8])
aa = tf.unstack(a, axis=2)
# 在 索引为2的纬度上 打散数据 也就是生成了8个【4，35】的列表
print(len(aa))
# 打印数字为 8
print(aa[0].shape, '\n', aa[7].shape)
b = tf.ones([3, 4, 5, 3])
bb = tf.unstack(b, num=3, axis=3)
print(b[0].shape, b[0])


# diy式的打散数据 在某个纬度上 使用split
c = tf.ones([2, 4, 35, 8])
cc = tf.split(c, num_or_size_splits=2)  # 均分
print(len(cc))
ccc = tf.split(c, num_or_size_splits=[2, 2, 4], axis=3)  # 按照2：2：4
print(len(ccc), '\n', ccc[0].shape, '\n', ccc[1].shape, '\n', ccc[2].shape)
'''
3 
 (2, 4, 35, 2) 
 (2, 4, 35, 2) 
 (2, 4, 35, 4)
'''

