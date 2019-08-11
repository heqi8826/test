from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# 下载心脏病例数据集
URL = 'https://storage.googleapis.com/applied-dl/heart.csv'
dataframe = pd.read_csv(URL)  # 数据类型：pandas.core.frame.DataFrame
print(dataframe)
print(dataframe.index)
print(dataframe.columns)
print(dataframe.to_numpy())

# 将数据集切割为：训练集、验证集、测试集
train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples', type(train))
print(len(val), 'validation examples', type(val))
print(len(test), 'test examples', type(test))
