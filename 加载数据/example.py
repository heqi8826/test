import tensorflow as tf


def prepare_features_and_labels(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255
    y = tf.cast(y, dtype=tf.int64)
    return x, y


def mnist_dataset():
    (x, y), (x_val, y_val) = tf.keras.datasets.mnist.load_data()
    y = tf.one_hot(y, depth=10)
    y_val = tf.one_hot(y_val, depth=10)

    