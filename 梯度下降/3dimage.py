import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
# z = (x^2+y-11)^2 + (x + 2 y - 7)^2


def myfunc(x):
    return (x[0]**2+x[1]-11)**2+(x[0]+x[1]**2-7)**2


x = np.arange(-6, 6, 0.1)
y = np.arange(-6, 6, 0.1)
print(x.shape, y.shape)
X, Y = np.meshgrid(x, y)
print(X.shape, Y.shape)
Z = myfunc([X, Y])

x = tf.constant([4., 0.])  # 作用：定义一个初始点

# 绘制图像
fig = plt.figure('myfunc')
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z)
ax.view_init(60, -30)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()


for step in range(200):
    with tf.GradientTape() as tape:
        tape.watch([x])
        y = myfunc(x)

    grads = tape.gradient(y, [x])[0]
    x -= 0.01 * grads
    if step % 20 == 0:
        print(
            "当循环step={}次时候，x={}的时候，loss={}".format(step, x, y)
        )






