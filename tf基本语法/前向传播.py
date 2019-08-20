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
# print(x_train.shape, y_train.shape, x_train.dtype, y_train.dtype)
# keras方法 查看最大值 最小值 reduce_min reduce_max
# print(tf.reduce_max(x_train),'\n',tf.reduce_min(x_train),'\n',tf.reduce_min(y_train),'\n',tf.reduce_max(y_train))
# 因原始数据较大，将其进行切片 创建数据集 以每次一个小批量（128）个进行训练计算。
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(128)
# 迭代 属性类型为：tensorflow.python.data.ops.iterator_ops.IteratorV2 object
train_iter = iter(train_db)
# 不断next查看 next_train的tensor数据
next_train = next(train_iter)
# 下面的构造W 和 b 的原理是 降维过程
# [b,784] => [b,256] => [b,128] => [b,10]
# 即：w：[dim_in,dim_out]  b：[dim_out]
w1 = tf.random.truncated_normal([784, 256])  # truncated 截断意思
# 第一层原始x_train的 dim_in 为784 输入，输出设计为 [b,256] 故 dim_out 为256 所以W1 构造为：[784，256]
b1 = tf.zeros([256])
w2 = tf.random.truncated_normal([256, 128])
b2 = tf.zeros([128])
w3 = tf.random.truncated_normal([128, 10])
b3 = tf.zeros([10])
# 下一步迭代训练数据（x_train，y_train）到模型层y=x@w+b
for (x_train, y_train) in train_db:
    # x_train:[128,28,28] y_train:[128]
    tf.reshape(x_train, [-1, 28*28])

# 视频课件第四章后三个视频内容。


