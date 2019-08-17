import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers

# 下载数据集
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
# 将numpy数据转换为tensor数据
x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)/255
# 将转化的x_train的tensor数据与y_train 重新生成新的数据
db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
for step, (x, y) in enumerate(db):
    print(step, x.shape, y, y.shape)

# 以上为训练数据准备，以下构建模型model
model = keras.Sequential()
model.add(layers.Dense(input_shape=(28, 28)))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(10))

# 优化函数
optimizer = optimizers.SGD(learning_rate=0.01)
# 配置模型
model.compile(optimizer, loss='mean_squared_error', metrics=['accuracy'])
