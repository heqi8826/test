import  tensorflow as tf
'''
sort（元素排序） argsort（元素位置排序） Topk Top-5 Acc.
'''
a = tf.constant([30, 32, 64, 3])
# print(tf.sort(a, direction='DESCENDING'), tf.argsort(a))  # 降序排列
# argsort 也支持升降序列排序

b = tf.random.uniform([3, 3], maxval=10, dtype=tf.int32)
bb = tf.sort(b)
# print(b, '\n', bb)

a = tf.constant([
    [4, 6, 8], [9, 4, 7], [4, 5, 1]
])
res = tf.math.top_k(a, 2)
# print(res)

pro = tf.constant([[0.1, 0.2, 0.7], [0.2, 0.7, 0.1]])
target = tf.constant([2, 0])
k_b = tf.math.top_k(pro, 3).indices
print(k_b)
print(tf.transpose(k_b, [1, 0]))
print(tf.broadcast_to(target, [3, 2]))

