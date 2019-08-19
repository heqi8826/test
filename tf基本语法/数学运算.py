import tensorflow as tf
from tensorflow import keras
'''
+ - * / %(取余数） // 取整
@（矩阵相乘）tf.matmul(a,b): a * b
reduce_mean
max
min
sum
注意 涉及 乘方 开方 对数 数据类型为 浮点型 否则报错
'''
a = tf.fill(dims=[2, 2], value=2.0)
b = tf.ones([2, 2])
print(a + b)
print(a - b)
print(a * b)
print(a / b)
print(a // b)
print(a % b)

# tf.math.log  与 tf.exp
print(tf.math.log(b))
print(tf.exp(b))  # （y = e ^ x ）
# tf.log函数 底数为e，如果需要更换 需要转换 log以a为底以b为真数 除以 log以a为底以c为真数的对数函数
# 就得到log以c为底以b为真数的对数结果了

c = tf.math.log(8.) / tf.math.log(2.)  # log()内参数为浮点型 不能为整型
print(c)
'''
tf.Tensor(3.0, shape=(), dtype=float32)
'''

# pow sqrt 乘方 和 开方运算
d = tf.convert_to_tensor([
    [2, 3, 4],
    [3, 5, 2],
    [1, 3, 2]
], dtype=tf.float32)
print(d)
dd = tf.pow(d, 2)
print(dd)
'''
tf.Tensor(
[[2 3 4]
 [3 5 2]
 [1 3 2]], shape=(3, 3), dtype=int32)
tf.Tensor(
[[ 4.  9. 16.]
 [ 9. 25.  4.]
 [ 1.  9.  4.]], shape=(3, 3), dtype=int32)
'''

print(tf.sqrt(dd))
'''
tf.Tensor(
[[1.9999999  2.9999998  3.9999998 ]
 [2.9999998  5.         1.9999999 ]
 [0.99999994 2.9999998  2.        ]], shape=(3, 3), dtype=float32)
'''

# tf.matmul
ab = tf.matmul(a, b)
print(ab)
e = tf.random.normal([4, 28, 28], dtype=tf.float32)
f = tf.fill([4, 28, 8], 2.0)
# 能够进行matmul的两个tensor数据的特征是 第一维度相等，高宽两个参数形成的矩阵 ：符合两个向量积的规则 A矩阵列数等于B矩阵行数
g = tf.matmul(e, f)
print(g.shape)

# 如果纬度不同 两个tensor相乘 使用 bradcast_to 构造
a = tf.random.normal((4, 3, 6))
b = tf.random.normal([6, 3])
bb = tf.broadcast_to(b, [4, 6, 3])
abb = tf.matmul(a, bb)
print('abb的形状是：', abb.shape)

# 验证 y = x@W + b
x = tf.ones([4, 2])
w = tf.ones([2, 1])
c = tf.constant(0.1)  # 自动broadcast为一个[4, 1]的tensor数据与x*w相加
print(x@w + c)
out = x@w + c
out = tf.nn.relu(out)
print(out)
