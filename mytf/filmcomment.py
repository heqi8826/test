# 导入tf 和 keras 库
import tensorflow as tf
from tensorflow import keras
import numpy as np

# 将影评数据集赋值给参数：imdb,可有可无，为了简便些赋值给参数，也可以直接使用tf.keras.datasets.imdb
imdb = tf.keras.datasets.imdb

# 加载影评数据，并将imdb影评数据赋值给 训练集 和 测试集,训练 和 数据 集 中的 参数均为：影评词句 与 标签
(train_words, train_labels), (test_words, test_labels) = imdb.load_data(num_words=10000)

# 打印一下影评数据的数量规模 可有可无。
# print("Training entries: {}, labels: {}".format(len(train_words), len(train_labels)))

# 探索数据了解数据使用tf.keras.datasets.imdb 下的 get_word_index()函数
# film_words = imdb.get_word_index()
# 看下film_words 参数获得的数据类型 。
# print(type(film_words))
# 获得的数据为字典类型 根据字典的每个数据的 键 和 值 的对应关系进行打印 获得影评数据。
# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k:(v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])
# print(decode_review(train_words[0]))
# 准备数据，使得测试和训练数据的矩阵长度标准化


train_words = keras.preprocessing.sequence.pad_sequences(train_words,
                                                         value=word_index['<PAD>'],
                                                         padding='post',
                                                         maxlen=256
                                                         )
test_words = keras.preprocessing.sequence.pad_sequences(test_words,
                                                        value=word_index['<PAD>'],
                                                        padding='post',
                                                        maxlen=256
                                                        )
# 使用keras库构建序列机器学习模型 并设置 堆砌层
# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
# model.summary()

# 配置模型以使用优化器和损失函数：
# model.compile(optimizer=tf.train.AdamOptimizer(),
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 创建验证集
x_val = train_words[:10000]
partial_x_train = train_words[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# 训练模型
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

# 使用测试数据，评估训练模型
results = model.evaluate(test_words, test_labels)
print(results)

