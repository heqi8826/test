import tensorflow as tf


# clip_by_value 限幅 限制tensor的最大值和最小值 (tensor,a,b)比a小的赋值为a，比b大的赋值为b
# relu
# clip_by_norm
# gradient clipping
a = tf.range(10)
print(tf.maximum(a, 3))
print(tf.minimum(a, 6))
print(tf.clip_by_value(a, 2, 7))
print('-----')
a = a - 5
print(a)
a = tf.nn.relu(a)
print(a)
a = tf.maximum(a, 0)
print(a)

# clip_by_norm
a = tf.random.normal([2, 2], mean=10)
print(a)
print(tf.norm(a))  # 平方和再开方 三次运算 平方 加和 开方
print('----')
a = tf.clip_by_norm(a, 15)
print(a)
print(tf.norm(a))


