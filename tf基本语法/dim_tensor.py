import tensorflow as tf
from tensorflow import keras

# 词汇dim = 3
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=10000)
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=80)
print(x_train.shape)
# 添加纬度 使用 embedding()函数
emb = keras.layers.Embedding(x_train, output_dim=100)
print(type(emb))

out = keras.layers.RNN(emb[:4])
print(out.shape)
# 图片 dim = 4
# image：[b,h,w,3]
# feature maps:[b,h,w,3]
imgs = tf.random.normal((4, 32, 32, 3))
net = keras.layers.Conv2D(16,  kernel_size=3, padding='valid')
print(net(imgs).shape)

