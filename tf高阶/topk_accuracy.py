import tensorflow as tf


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.shape[0]

    pred = tf.math.top_k(output, maxk).indices
    pred = tf.transpose(pred, perm[1, 0])
    target_ = tf.broadcast_to(target, pred.shape)
    correct = tf.equal(pred, target_)

    res = []
    for k in topk:
        correct_k = tf.cast(tf.reshape(correct[:k]), [-1], dtype=tf.float32)
        correct_k = tf.reduce_sum(correct_k)
        acc = float(correct_k / batch_size)
        res.append(acc)
    return res
