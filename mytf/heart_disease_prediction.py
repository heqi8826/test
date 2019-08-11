import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 下载心脏病例数据集
Url = "https://storage.googleapis.com/applied-dl/heart.csv"
heart_sill_form = pd.read_csv(Url)


# 将数据集切割为：训练集、验证集、测试集
train, test = train_test_split(heart_sill_form, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
# print(len(train), 'train examples')
# print(len(val), 'validation examples')
# print(len(test), 'test examples')


# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(heart_sill_form, shuffle=True, batch_size=32):
  heart_sill_form = heart_sill_form.copy()
  labels = heart_sill_form.pop('target')
  ds = tf.data.Dataset.from_tensor_slices((dict(heart_sill_form), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(heart_sill_form))
  ds = ds.batch(batch_size)
  return ds


batch_size = 5  # A small batch sized is used for demonstration purposes
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

for feature_batch, label_batch in train_ds.take(1):
  print('Every feature:', list(feature_batch.keys()))
  print('A batch of ages:', feature_batch['age'])
  print('A batch of targets:', label_batch)



