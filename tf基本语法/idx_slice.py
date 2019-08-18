import tensorflow as tf
from tensorflow import keras

# indexing
# Basic indexing:[idx][idx][idx]
a = tf.ones([2, 4, 5, 3])
print(a[0][2][1][2])  # 取最底的标量 需要四次[idx]
# same with Nyumpy 下面举例numpy风格的取法
print(a[0].shape)
print(a[0, 1].shape)
print(a[0, 1, 2].shape)
print(a[0, 1, 3, 1].shape)
'''
打印结果如下：
(4, 5, 3)
(5, 3)
(3,)
()
'''
# tensor切片方法：start:end

