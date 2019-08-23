import tensorflow as tf
import numpy as np
import pandas as pd

data = pd.read_csv('./data.csv', dtype={'手机': str})
data = pd.DataFrame(data)
# print(data)
data = pd.concat([data, pd.DataFrame(columns=['院系', '学号', '毕业去向', '负责人'])])
data = data.reindex(columns=['姓名', '院系', '专业', '学号', '毕业去向', '手机', '负责人'])
# print(data)
update = pd.read_csv('./update.csv', dtype={'手机': str})
update = pd.DataFrame(update).drop(index=0)
update = [update, data]
update = pd.concat(update)
print(update)

# print(data['姓名'], '\n', data['专业'], '\n', data['手机'])
# update = pd.read_csv('./update.csv')
# update = pd.DataFrame(update)
# update = update.merge(data, columns='姓名')
# print(update)


