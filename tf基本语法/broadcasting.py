import tensorflow as tf
from tensorflow import keras

a = tf.ones([4, 32, 32, 3])
b = tf.random.normal([4, 1, 1, 1])
b = tf.broadcast_to(b, shape=[4, 32, 32, 3])
print(b.shape)