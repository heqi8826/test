import tensorflow as tf
# 可导属性tf.Variable（）
x = tf.range(10)

y = tf.Variable(x)
print(y.dtype)
print(y.name)
print('----')
y = tf.Variable(x, name='input_data')  # 专门为神经网络计算所设计的一个参数
# 被Variable执行之后，该参数 就具备了name 和 trainable两个属性。
print(y.name)
print(y.trainable)  # return True 数据可以用于训练模型
print(y)
print('----')
print(isinstance(y, tf.Tensor))  # 与下面的is_tensor 比较下，所以不推荐isinstance
print(tf.is_tensor(y))  # 使用上面的函数 返回假使用该函数返回真。建议使用is_tensor
print(isinstance(x, tf.Tensor))
print(isinstance(y, tf.Variable))



