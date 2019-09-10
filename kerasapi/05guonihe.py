import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def preprocess(x, y):
    x = 2 * tf.cast(x, dtype=tf.float32)/255 - 1
    y = tf.cast(y, dtype=tf.int32)
    return x, y


batchsz = 128
(x, y), (x_val, y_val) = keras.datasets.cifar10.load_data()
y = tf.squeeze(y)
y_val = tf.squeeze(y_val)
y = tf.one_hot(y, depth=10)
y_val = tf.one_hot(y_val, depth=10)
print('datasets:', x.shape, y.shape, x_val.shape, y_val.shape, x.min(), x.max())

db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(preprocess).shuffle(10000).batch(batchsz)
db_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
db_val = db_val.map(preprocess).batch(batchsz)

sample = next(iter(db))
print('每一个样本的数据为：', sample[0].shape, sample[1].shape)


class MyDense(layers.Layer):

    def __init__(self, in_dim, out_dim):
        super(MyDense, self).__init__()

        self.kernel = self.add_variable('w', [in_dim, out_dim])
        # self.bias = self.add_weight('b', [out_dim])

    def call(self, inputs, training=None):

        x = inputs @ self.kernel
        return x


class MyNetwork(keras.Model):

    def __init__(self):
        super(MyNetwork, self).__init__()

        self.fc1 = MyDense(32*32*3, 256)
        self.fc2 = MyDense(256, 128)
        self.fc3 = MyDense(128, 64)
        self.fc4 = MyDense(64, 32)
        self.fc5 = MyDense(32, 10)

    def call(self, inputs, training=None):
        x = tf.reshape(inputs, [-1, 32 * 32 * 3])
        x = self.fc1(x)
        x = tf.nn.relu(x)
        x = self.fc2(x)
        x = tf.nn.relu(x)
        x = self.fc3(x)
        x = tf.nn.relu(x)
        x = self.fc4(x)
        x = tf.nn.relu(x)
        x = self.fc5(x)
        # x = tf.nn.relu(x)  # 最后一层一般不用激活函数的 屏蔽掉删除即可
        return x


network = MyNetwork()
network.compile(optimizer='Adam', loss=tf.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
network.fit(db, epochs=15, validation_data=db_val, validation_freq=1)

network.evaluate(db_val)
network.save_weights('./method/wangqi.tf')
del network
print('保存完毕')

network = MyNetwork()
network.compile(optimizer='Adam', loss=tf.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
network.fit(db, epochs=15, validation_data=db_val, validation_freq=1)

network.load_weights('./method/wangqi.tf')
print('加载完毕')






