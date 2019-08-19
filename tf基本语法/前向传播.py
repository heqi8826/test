import tensorflow as tf
from tensorflow import keras
import os
# 特别鸣谢@龙良曲老师
# 回顾：
# 创建tensor数据方法
# 索引与切片
# reshape重构tensor的shape便于操作计算，broadcast_to
# math工具包的函数方法 加减乘除取余取整对数相乘等

# 回顾线性回归模型实例
# 加载数据集 影评的

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# 数据转换为tensor类型 ，定义数据为tf的浮点型32位数据
x_train = tf.convert_to_tensor(x_train, dtype=tf.float32) / 255.  # 除以255. 为了方便计算 将[0,255]范围转换为[0,1]范围
y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)
print(x_train.shape, y_train.shape, x_train.dtype, y_train.dtype)
# keras方法 查看最大值 最小值 reduce_min reduce_max
print(tf.reduce_max(x_train), '\n',  tf.reduce_min(x_train), '\n', tf.reduce_min(y_train), '\n', tf.reduce_max(y_train))