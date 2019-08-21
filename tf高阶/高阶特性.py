import tensorflow as tf
'''
where（）接收一个参数（tensor）时候，里面为bool类型的坐标。
根据坐标中True的横向和纵向索引 一一定位。参考 yuque 笔记
 scatter_nd meshgrid
'''
a = tf.random.normal([3, 3])
mask = a > 0
print(a, mask)
'''
tf.Tensor(
[[False False  True]
 [False  True False]
 [False  True False]], shape=(3, 3), dtype=bool)
'''
b = tf.boolean_mask(a, mask)
print(b)
'''
tf.Tensor([2.5953548 1.1503891 1.2916565], shape=(3,), dtype=float32)
'''
indices = tf.where(mask)
print(indices)
'''
tf.Tensor(
[[0 2]
 [1 1]
 [2 1]], shape=(3, 2), dtype=int64)
'''

aa = tf.gather_nd(a, indices)
print(aa)
'''
tf.Tensor([0.29008022 1.2823507  0.23192722 0.17014457], shape=(4,), dtype=float32)
'''
# where(cond,a,b) 三个参数时候 cond为bool值类型的tensor ，true时候从a里取样，反之从b取样。
a = tf.fill([3, 3], 1)
b = tf.fill([3, 3], 0)
print(a, b)
ab = tf.where(mask, a, b)
print(ab)

# scatter_nd
indices = tf.constant([[4], [3], [1], [7]])  # 提供索引值的作用
update = tf.constant([9, 10, 11, 12])  # value作用
shape = tf.constant([8])  # 提供取样数据放置的模版作用
res = tf.scatter_nd(indices, update, shape)  # 将索引为4的shape是9，索引为3的shape上为10 以此类推。
print(res)