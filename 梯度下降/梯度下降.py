import tensorflow as tf
from tensorflow import keras


# w = tf.constant(1.)
# x = tf.constant(2.)
# y = x * w
# with tf.GradientTape() as tape:
#     tape.watch([w])
#     y2 = x * w
# grad1 = tape.gradient(y, [w])


# with tf.GradientTape(persistent=True) as tape:
#     tape.watch([w])
#     y3 = x * w
# grad2 = tape.gradient(y3, [w])
# print(grad2)
# grad2 = tape.gradient(y3, [w])  # 二次调用，必须persistent=True
# print(grad2)
# 二阶梯度
w = tf.constant(1.)
x = tf.constant(2.)
b = tf.constant(3.)
with tf.GradientTape(persistent=True) as tape:
    with tf.GradientTape(persistent=True) as tape2:
        tape.watch([x])
        y = w * x * x * x + 2 * x * x + b
grad = tape.gradient(y, [x])  # 执行外层的tape方法
print(grad)

# 以上为简单的函数 使用普通求导方法很方便求导，但是当工程中遇到较复杂的函数时候
a = tf.linspace(-10., 10., 10)  # 创建一个等差数列
with tf.GradientTape(persistent=True) as tape:
    tape.watch(a)
    y = tf.sigmoid(a)

grad = tape.gradient(y, [a])
print(a)
print(grad)

# 其他函数
'''
tf.tanh(x)
tf.nn.relu(x)
tf.nn.leaky_relu(x)
'''

