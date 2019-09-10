import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential


class BasicBlock(layers.Layer):

    def __init__(self, filter_num, stride=1):  # filter_num 通道数量
        super(BasicBlock, self).__init__(self)

        self.conv1 = tf.keras.layers.Conv2D(filter_num, kernel_size=(3, 3), strides=stride, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.Activation('relu')

        self.conv2 = tf.keras.layers.Conv2D(filter_num, kernel_size=(3, 3), strides=stride, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()

        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num, (1, 1), strides=stride))
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None):

        output = self.conv1(inputs)


