import tensorflow as tf
from tensorflow import keras
'''
小型数据集
keras.datasets
tf.data.Dataset.from_tensor_slices
    shuffle
    map
    batch
    repeat
常用数据集
boston housing
mnist/fashion mnist
cifar10/100 10大类 每大类对应10个小类
imdb
'''
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()  # 拿到的数据为numpy数据 非tensor数据
print(x_train.shape, '\n', y_train.shape, '\n', x_train.min(), '\n', x_train.max(), '\n', x_train.mean())
# 上述打印所使用的min max mean的方法为numpy工具包中的 ，如果为tensor数据 则使用tf.reduce_min / reduce.max /reduce.mean
'''
(60000, 28, 28) 
 (60000,)  # 是一个0-9的列表，后续编码为one-hot的tensor数据，one-hot占用资源多 前期不做转换。
 0 
 255  # 后续为方便处理 可以令x_train 除以 255 变为 0-1范围，根据预处理需求也可以变为-1～1范围 或者0-1范围等等diy范围。
 33.318421449829934
'''
print(x_test.shape, '\n', y_test.shape, '\n', x_test.max())
'''
(10000, 28, 28) 
 (10000,)
 255
'''
# 转换y_train为one_hot编码
# 首先查看下内部数据形式 ，任意取n个数据查看
test_y = y_train[0:5]  # pick five
print(test_y, test_y.dtype)  # [5 0 4 1 9]
y_train = tf.one_hot(y_train, depth=10)  # 独热编码转换，数据类型转换为了float32
print(y_train[0:5])
'''
tf.Tensor(
[[0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]], shape=(5, 10), dtype=float32)
'''

# 使用cifar10/100 加载数据集中的小像素图片
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# 转换成tensor数据类型
db = tf.data.Dataset.from_tensor_slices(x_test)

# 通过迭代器将db一个一个数据取样
one = next(iter(db)).shape
print(one)
db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
print(next(iter(db))[0].shape)

# .shuffle 方法 打散数据
db = db.shuffle(buffer_size=10000)


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)/255
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x, y


db2 = db.map(preprocess)
'''
map() 会根据提供的函数对指定序列做映射。
map(function, iterable, ...)
function -- 函数
iterable -- 一个或多个序列
Python 3.x 返回迭代器。
>>>def square(x) :            # 计算平方数
...     return x ** 2
... 
>>> map(square, [1,2,3,4,5])   # 计算列表各个元素的平方
[1, 4, 9, 16, 25]
>>> map(lambda x: x ** 2, [1, 2, 3, 4, 5])  # 使用 lambda 匿名函数
[1, 4, 9, 16, 25]
 
# 提供了两个列表，对相同位置的列表数据进行相加
>>> map(lambda x, y: x + y, [1, 3, 5, 7, 9], [2, 4, 6, 8, 10])
[3, 7, 11, 15, 19]
'''
res = next(iter(db2))
print(res[0].shape, res[1].shape)
print(res[1][:2])
# .batch 方法
db3 = db2.batch(32)
res = next(iter(db3))
print(res[0].shape, res[1].shape)

# 完整流程 见下一个文件exampl
