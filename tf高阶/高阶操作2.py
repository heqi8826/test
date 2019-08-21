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
y = np.linspace(-2., 2, 5)
y = tf.convert_to_tensor(y)
x1, y1 = tf.meshgrid(x, y)
print(x1.shape)



