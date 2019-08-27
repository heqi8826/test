import tensorflow as tf
import numpy as np

update = tf.random.normal([2, 4, 4])
shape = tf.constant([4, 4, 4])
print(shape.shape)
indices = tf.constant([[0], [2]])

res = tf.scatter_nd(indices, update, shape)
print(res)

points = []
for x in np.linspace(-2, 2, 5):
    for y in np.linspace(-2, 2, 5):
        z = [x, y]
        points.append(z)
        np.array(points)

print(len(points), '\n', points)

# 用另外一种便捷方法：meshgrid
x = tf.linspace(-2., 2, 5)
x = tf.cast(x, dtype=tf.double)
y = np.linspace(-2., 2, 5)
y = tf.convert_to_tensor(y)
print('x与y的打印结果为：', x, y)
points_x, points_y = tf.meshgrid(x, y)
print(points_x.shape, points_x)
print(points_y.shape, points_y)
print('-----')
points = tf.stack([points_x, points_y], axis=0)
print('当axis=0时候拼接结果为：', points)

points = tf.stack([points_x, points_y], axis=1)
print('当axis=1时候拼接结果为：', points)

points = tf.stack([points_x, points_y], axis=2)
print('当axis=2时候拼接结果为：', points)


