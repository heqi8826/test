import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import datetime
import imageloss.py
'''
tensorBoard
安装、loss和acc监听、可视化
'''
loss = 0.1
epoch = 100
train_accuracy = 0.56
current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
log_dir = 'logs/'+ current_time
summary_writer = tf.summary.create_file_writer(log_dir)

with summary_writer.as_default():
    tf.summary.scalar('loss', float(loss), step=epoch)
    tf.summary.scalar('accuracy', float(train_accuracy), step=epoch)
# 以上为传递 标量 到 可视化页面 也可以传递 图片数据
sample_img = next(iter(db))[0]
sample_img = tf.reshape(sample_img[0], [1, 28, 28, 1])
with summary_writer.as_default():
    tf.summary.image(sample_img, step=0)

