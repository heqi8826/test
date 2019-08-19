import tensorflow as tf
# tf.boolean_mask 方法使用
mybool = tf.ones((4, 28, 28, 3))  # 定义四张28*28的彩色图片集合
# 使用tf.boolean_mask取样前两张图片
mybool1 = tf.boolean_mask(mybool, mask=([True, True, False, False]), axis=0)  # mask内参数 前两个为真 则取前两个图片
print(mybool1.shape)

# 取色道 R 和G 的颜色channel
mybool2 = tf.boolean_mask(mybool, mask=([True, True, False]), axis=3)  # RGB颜色前两个通道为真
print(mybool2.shape)

# 比较难点的 mask标记多维
myeg = tf.ones((2, 3, 4))  # 2个3*4的tensor数据集
print(myeg)
'''
[[[1. 1. 1. 1.]
  [1. 1. 1. 1.]
  [1. 1. 1. 1.]]
 [[1. 1. 1. 1.]
  [1. 1. 1. 1.]
  [1. 1. 1. 1.]]]
  第一个纬度为 2 个2纬的tensor数据
  [[1. 1. 1. 1.]
  [1. 1. 1. 1.]
  [1. 1. 1. 1.]]
  第二个纬度为 3 个列表的tensor数据
  [1. 1. 1. 1.]
  第三个纬度为 4 个标量的tensor数据 即：1
'''
myeg1 = tf.boolean_mask(myeg, mask=[[True, False, False], [False, True, True]])  # 根据mask形式确定取样纬度，而不使用axis参数
# 根据mask内的tensor类型的确定纬度为 myeg的 【2，3】综合纬度
# [True, False, False] 表示 第一纬度 的第一列为真
print(myeg1)

