import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, datasets, Sequential, optimizers
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(22)
np.random.seed(22)
assert tf.__version__.startswith('2.')

num_words = 10000
batchsz = 128
(x, y), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=num_words)
db = tf.convert_to_tensor((x, y)).shuffle().batch(batchsz)
db_test = tf.convert_to_tensor((x_test, y_test)).
