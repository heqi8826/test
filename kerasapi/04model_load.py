import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
'''
save/load weight  只保存参数 需要对代码逻辑 层次有个清晰的把我
save/load entire model 简单粗暴的一宗 把所有的状态保存 后续恢复
saved_model
'''
# save/load weight
model.save_weight('./checkpoints/mycheckpoint')
# restore the weights
model = create_model()  # model = Squential()
model.load_weight('./checkpoints/mycheckpoint')

loss, acc = model.evaluate(test_images, test_labels)
print('Restored model, accuracy:{5.2f}%'.format(100*acc))


# 整个model保存
network.save('model.h5')
print('saved total model.')
del network

print('load model from file')
network = tf.keras.models.load_model('model.h5')

network.evaluate(x_val, y_val)


# 工业生产
tf.saved_model.save(m, '存储路径path')

imported = tf.saved_model.load(path)
f =imported.signatures['serving_defult']
print(f(x=tf.ones([1, 28, 28, 3])))
