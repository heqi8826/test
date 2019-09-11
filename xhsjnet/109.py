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
max_review_len = 80
(x, y), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=num_words)
print(x[0:5])
print(type(x), x_test.shape)
# 预处理数据preprocessing
x = tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=max_review_len)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_review_len)
print(x.shape, x_test.shape, x.dtype, x_test.dtype)
# 转换为tensor类型

db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.shuffle(10000).batch(batchsz, drop_remainder=True)

db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.batch(batchsz, drop_remainder=True)

print(x.shape, x_test.shape, type(db), type(db_test))

db_item = iter(db)
db_test_item = iter(db_test)
db_item = next(db_item)
db_test_item = next(db_test_item)
print(db_item, db_test_item)
