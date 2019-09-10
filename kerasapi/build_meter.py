import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
# auther:感恩购
# email:heqi8826@qq.com

# 创建meter数据
acc_meter = tf.metrics.Accuracy()
loss_meter = tf.metrics.Mean()
# 向meter中添加数据update()
acc_meter = acc_meter.update_state(y, pred)
loss_meter = loss_meter.update_state(loss)
# 获得数据举例如下：
print(step, 'loss', loss_meter.result().numpy())
print(step, 'acc', total_correct, acc_meter.result().numpy())
# 清楚数据 放置在以上的赋值运算中 继续添加 而不是 新的添加操作
loss_meter = loss_meter.reset_states()
acc_meter = acc_meter.reset_states()
