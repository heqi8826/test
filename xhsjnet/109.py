import tensorflow as tf
import numpy as np
from tensorflow import keras
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(22)
np.random.seed(22)
assert tf.__version__.startswith('2.')


def proprecess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255
    y = tf.cast(y, dtype=tf.int32)
    return x, y


embedding_len = 100
num_words = 10000
batchsz = 80
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
db = db.shuffle(1000).batch(batchsz, drop_remainder=True)

db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.batch(batchsz, drop_remainder=True)


class MyRNN(keras.Model):  # 构建函数模型

    def __init__(self, units):
        super(MyRNN, self).__init__(self)
        self.state = [tf.zeros((batchsz, units))]
        self.state1 = [tf.zeros([batchsz, units])]
        # 数据转换，将[b,80]的语句转换为：[b,80,100]的数据模型
        self.embedding = tf.keras.layers.Embedding(num_words, embedding_len, input_length=max_review_len)
        # 将语句按照时间轴展开：SimpleRNNCell
        self.rnn_cell0 = tf.keras.layers.SimpleRNNCell(units, dropout=0.2)
        self.rnn_cell1 = tf.keras.layers.SimpleRNNCell(units, dropout=0.2)
        # 将语句输入到全连阶层
        self.outlayer = tf.keras.layers.Dense(1)

    def call(self, inputs, training=None):
        # [b,80]
        x = inputs
        # embedding : [b,80] => [b, 80, 100]
        x = self.embedding(x)
        # 降低纬度 100 => 64
        state = self.state
        state1 = self.state1
        for word in tf.unstack(x, axis=1):  # unstack 提取axis为1的单词个数纬度数据分析词特征[b,100]
            out, state = self.rnn_cell0(x, state, training)
            out1, state1 = self.rnn_cell1(out, state1, training)

        x = self.outlayer(out1)
        prob = tf.nn.sigmoid(x)
        return prob


def main():
    units = 64
    epochs = 4
    model = MyRNN(units)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.losses.BinaryCrossentropy(),
                  metrics=['accuracy']
                  )
    model.fit(db, epochs=epochs, validation_data=db_test)
    model.evaluate(db_test)


if __name__ == '__main__':
    main()