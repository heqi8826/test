import tensorflow as tf
from tensorflow import keras
# tf.norm 张量范数 比如 向量范数 矩阵范数
# tf.reduce_min/max 张量最大值 最小值
# tf.argmax / argmin 最大值 最小值位置
# tf.equal 张量比较
# tf.unique 独特性
'''
范数理论：一系列数的平方和再开方 称为二范数 eukl.norm，一系列数的绝对值的和 称为一范数l1-norm
无穷范数 则是 最大的一个数的绝对值 max.norm
'''
# tf.norm举例
a = tf.ones([2, 3])
aa = tf.norm(a)
print('aa的值为：', aa)
# tf.norm(a)等价于以下计算方法
aaa = tf.sqrt(tf.reduce_sum(tf.square(a)))
print('aaa的值为：', aaa)
'''
aa的值为： tf.Tensor(2.4494898, shape=(), dtype=float32)
aaa的值为： tf.Tensor(2.4494898, shape=(), dtype=float32)
'''
# 多维度同样适用
b = tf.ones([4, 32, 32, 3])
bb = tf.norm(b)
print(bb)

# 指定某个纬度求norm
c = tf.ones([2, 3])
print(c)
'''
[[1,1,1]
 [1,1,1]]
纬度解析：
 [[A][A]] axis=0 所以第一个纬度的一范数为：A+A=1+1=2 共3列 所以结果为【2，2，2】
 [B,B,B]  axis=1 所以第二个纬度的一范数为：B+B+B=1+1+1=3 共2行，所以结果为【3，3】
'''
cc = tf.norm(c, ord=2, axis=0)  # ord=2 第二个纬度上的数据指定二范数运算

print(cc)
ccc = tf.norm(c, ord=1, axis=0)  # 第二个纬度上的数据进行一范数运算
print(ccc)

cc1 = tf.norm(c, ord=2, axis=1)  # ord=2 第二个纬度上的数据指定二范数运算
print(cc1)
ccc1 = tf.norm(c, ord=1, axis=1)  # 第二个纬度上的数据进行一范数运算
print(ccc1)

# 求reduce_min max mean
d = tf.random.normal([4, 8])
# 求第一个纬度的最大值 最小值
dd = tf.reduce_max(d, axis=0)
dd_num = tf.argmax(d, axis=0)
ddd = tf.reduce_min(d, axis=0)
ddd_num = tf.argmin(d, axis=0)
print(dd, ddd)
print('------')
print(dd_num, ddd_num)

# tf.equal 对应位置做比较 相同true 否则fasle
a = tf.constant([1, 3, 2, 2, 4])
b = tf.range(5)
print(a)
print(b)
print(tf.equal(a, b))
'''
tf.Tensor([False False  True False  True], shape=(5,), dtype=bool)
'''

# accuracy求准确率 占比 某一个特征在数据集中的占比
# 将上述打印的结果bool类型转化为int类型
c = tf.equal(a, b)
pre = tf.cast(c, dtype=tf.int32)  # cast ： bool转换为int类型
correct = tf.reduce_sum(pre)
print(correct/len(c))

# tf.unique 去除重复元素
a = tf.constant([4, 2, 2, 4, 3])
print(tf.unique(a))
'''
去除重复元素后：
Unique(y=<tf.Tensor: id=86, shape=(3,), dtype=int32, numpy=array([4, 2, 3], dtype=int32)>, idx=<tf.Tensor: id=87, shape=(5,), dtype=int32, numpy=array([0, 1, 1, 0, 2], dtype=int32)>)
解释：
[0, 1, 1, 0, 2] 代表 原来的元素 对应 现在的剔除重复元素后新tensor数据的索引值
'''
# 还原回去剔除的元素gather
aa = tf.gather(a, [0, 1, 1, 0, 2])
print(aa)