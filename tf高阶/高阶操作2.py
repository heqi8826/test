import tensorflow as tf


update = tf.random.normal([2, 4, 4])
shape = tf.constant([4, 4, 4])
print(shape.shape)
indices = tf.constant([[0], [2]])

res = tf.scatter_nd(indices, update, shape)
print(res)