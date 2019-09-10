import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import datetime
import tensorboard
import matplotlib.pyplot as plt


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)/255.
    y = tf.cast(y, dtype=tf.int32)

    return x, y


(x, y), (x_test, y_test) = datasets.fashion_mnist.load_data()
# 构造数据集from_tensor_slices
batchsz = 128
db = tf.data.Dataset.from_tensor_slices((x, y))
# 预处理数据 使用map函数 map内的参数为函数名 而不是函数调用方法。
db = db.map(preprocess).shuffle(10000).batch(batchsz)

db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.map(preprocess).shuffle(10000).batch(batchsz)

db_iter = iter(db)
sample = next(db_iter)
print(sample[0].shape, sample[1].shape)

# 构建模型model
model = Sequential()
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(10))

model.build(input_shape=[None, 28*28])
model.summary()  # summary（）调试功能 查看model模型的网络层信息

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


def main():

    for epoch in range(30):
        current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        log_dir = 'logs/' + current_time
        summary_writer = tf.summary.create_file_writer(log_dir)
        sample_img = next(iter(db))[0]
        sample_img = tf.reshape(sample_img[0], [1, 28, 28, 1])
        with summary_writer.as_default():
            tf.summary.image(sample_img, step=0)
        for step, (x, y) in enumerate(db):

            x = tf.reshape(x, [-1, 28*28])

            with tf.GradientTape() as tape:
                tape.watch(x)
                logits = model(x)
                y = tf.one_hot(y, depth=10)
                loss_mse = tf.reduce_mean(tf.losses.mse(y, logits))
                loss_ce = tf.losses.categorical_crossentropy(y, logits, from_logits=True)
                loss_ce = tf.reduce_mean(loss_ce)

            grads = tape.gradient(loss_ce, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 100 == 0:
                print(epoch, step, 'loss:', float(loss_ce), float(loss_mse))
        # test测试数据环节
        total_correct = 0
        total_num = 0
        for (x, y) in db_test:
            x = tf.reshape(x, [-1, 28*28])
            logits = model(x)  # 相当于y值
            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(prob, axis=1)  # 求最大的prob预测值
            y = tf.cast(y, dtype=tf.int64)
            correct = tf.equal(pred, y)  # true fasle 的bool型
            correct = tf.reduce_sum(tf.cast(correct, dtype=tf.int32))

            total_correct += int(correct)
            total_num += x.shape[0]
        acc = total_correct / total_num
        print(acc)



    pass


if __name__ == '__main__':
    main()