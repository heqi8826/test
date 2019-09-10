import tensorflow as tf
from tensorflow import keras
from keras import optimizers, datasets
'''
compile 比之前的训练更简洁更方便 代码量减少 它将优化器、loss计算、衡量的标准（比如精准度accuracy）三个参数通过compile函数一步实现
fit
evaluate
predict 
'''
# 假设network为创建好的网络连接层，通过compile函数将优化器 loss 衡量标准参数传入实现快速计算,db为测试的数据集
network.compile(optimizers.Adam(learing_rate=0.01), loss=tf.losses.categorical_crossentropy(from_logits=True),metrics=['accuracy'])
network.fit(db, epoch=10,validation_data=ds_val,validation_frequent=2)
network.evaluate(ds_val)  # 与上述validation_frequent=2工作一样
