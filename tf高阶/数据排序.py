import  tensorflow as tf
'''
sort argsort Topk Top-5 Acc.
'''
a = tf.constant([30, 32, 64, 3])
print(tf.sort(a, direction='DESCENDING'), tf.argsort(a))
