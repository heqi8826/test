import  os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 下载数据集
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
# 将numpy数据转换为tensor数据
x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)/255
y_train = tf.convert_to_tensor(y_train,dtype=tf.int32)
y_train = tf.one_hot(y_train, depth=10)
# 将转化的x_train的tensor数据与y_train 重新生成新的tensor格式数据 训练用
db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
db = db.batch(200)

# 以上为训练数据准备，以下构建模型model
model = keras.Sequential()
model.add(layers.Dense(512, activation='relu'))#降低纬度为512列
model.add(layers.Dense(256, activation='relu'))#降低纬度为256列
model.add(layers.Dense(10))#降低纬度为10列目标结果

# 优化函数
optimizer = optimizers.SGD(learning_rate=0.001)
optimizer = optimizers.SGD(learning_rate=0.001)
# 配置模型
# model.compile(optimizer, loss='mean_squared_error', metrics=['accuracy'])


def train_epoch(epoch):
    for step, (x_train, y_train) in enumerate(db):
        with tf.GradientTape() as tape:
            x_train = tf.reshape(x_train, (-1, 28*28))
#            计算out
            out = model(x_train)
#            计算loss
            loss = tf.reduce_sum((out-y_train)**2)/len(x_train)
        # 优化 更新w、b
        grads = tape.gradient(loss, model.trainable_variables)
        # w = w-lr*loss
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if step % 100 == 0:
            print(epoch, step, 'loss:', loss.numpy())


def train():
    for epoch in range(30):
        train_epoch(epoch)


if __name__ == '__main__':
    train()