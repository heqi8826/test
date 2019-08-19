import tensorflow as tf
from tensorflow import keras
import numpy as np
'''
gather 收集
tf.gather tf.gather_nd tf.boolean_mask 三种收集数据方式 提取数据方式
'''
# 4个班级学生课程数据data：[classes, students, subjects] :[4,35,8]
# obj = tf.ones((4, 35, 8))
obj = tf.convert_to_tensor(np.ones((4, 35, 8)))
# 取出索引为2 3 的 以班级为纬度的数据
obj1 = tf.gather(obj, indices=[2, 3], axis=0)
print(obj1.shape)

# 取出索引为2、3的班级数据，即：[2:4] 针对 第一个纬度4个班级 参数切片的
print(obj[2:4].shape)
# 在班级纬度（axis=0,即第一个纬度）上收集学生课程数据，不过这次取班级索引indices=[2,1,3,0]的数据
obj2 = tf.gather(obj, indices=[2, 1, 3, 0], axis=0)
print(obj2.shape)

# 以学生为纬度，取 学生因子索引为indices=[2,3,7,9,16]的数据，学生参数的纬度axis=1 第二个纬度 索引值为1
obj3 = tf.gather(obj, indices=[2, 3, 7, 9, 16], axis=1)
print(obj3.shape)

# 使用tf.gather实现 取样4个班的7个学生的前三门课程数据,通过gather串行分两步可以实现
# 第一步 取样4个班的7个学生的8门课程
obj_one = tf.gather(obj, indices=[0, 1, 2, 3, 4, 5, 6], axis=1)  # 研究对象（取样对象为学生，纬度为第二纬度）
print(obj_one.shape)
obj_two = tf.gather(obj_one, indices=[0, 1, 2], axis=2)  # 课程为第三个纬度,第三个纬度的索引值为 2
print(obj_two.shape)
'''
打印结果：
(2, 35, 8)
(2, 35, 8)
(4, 35, 8)
(4, 5, 8)
(4, 7, 8)
(4, 7, 3)
'''

# tf.gather_nd 指定纬度
obj4 = tf.gather_nd(obj, [0])
print(obj4.shape)
obj5 = tf.gather_nd(obj, [0, 1])  # 对样本obj取样，取第索引为0纬度的班级的学生索引为1的数据
print(obj5.shape)
obj6 = tf.gather_nd(obj, [0, 1, 2])
print(obj6.shape)
obj7 = tf.gather_nd(obj, [[0, 1, 2]])
print(obj7.shape)
'''
打印出如下结果
(35, 8)
(8,)
()
(1,)
'''

