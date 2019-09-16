import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import datetime

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = tf.cast(x_test, dtype=tf.float32) / 255.0
x_train = tf.cast(x_train, dtype=tf.float32) / 255.0
model = tf.keras.Sequential(
    [layers.Flatten(input_shape=(28, 28)), layers.Dense(128, activation='relu'), layers.Dropout(0.2),
     layers.Dense(10, activation='softmax')])
# 可视化页面监听数据创建
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs/' + current_time
summary_writer = tf.summary.create_file_writer(log_dir)
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=150, epochs=4, validation_data=(x_test, y_test), validation_freq=2)
with summary_writer.as_default():
    tf.summary.scalar('loss', float(model.compile.sparse_categorical_crossentropy), step=epochs)
    tf.summary.scalar('accuracy', float(metrics), step=epochs)


def get_images(x_test):
    images_print = []
    for i in range(len(x_test)):
        image = next(iter(x_test))
        model.predict(image)
        plt.matshow(image)
        images_print += plt.show()
        images_print = tf.reshape(images_print, [-1, 50, 50, 1])
        return images_print


with summary_writer.as_default():
    tf.summary.image('image', get_images(x_test), step=0)
