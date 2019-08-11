import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd

# 导入数据集
url = './data.csv'
dataframe = pd.read_csv(url).drop(columns=['Unnamed: 0', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6'])
mydict = pd.DataFrame(dataframe)
points = mydict.to_numpy()


# 定义函数，实现计算线性模型y=wx+b的loss均方误差平均值
def myloss(points, w, b):
    losserror = 0
    for i in range(len(points)):
        x = points[i][0]
        y = points[i][1]
        losserror += ((w * x + b) - y) ** 2
        return losserror / len(points)


# 计算w' 和 b'
def my_w_b(w_current, b_current, points, learningrate):
    w = 0
    b = 0
    for i in range(len(points)):
        x = points[i][0]
        y = points[i][1]
        N = len(points)
        # w 和 b 的变化率累加
        w += 2/N * (w_current * x + b_current - y) * x
        b += 2/N * (w_current * x + b_current - y)
        return w, b
    new_w = w_current - learningrate * w
    new_b = b_current - learningrate * b
    return [new_w, new_b]

def myrun(points, b, w, num):
    b = b
    w = w
    for i in range(num):
        b, w = my_w_b(b, w, mypoints, learningrate)
    return [b, w]










