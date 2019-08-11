import tensorflow as tf
import timeit

with tf.device('/cpu:0'):
    cpu_a = tf.random.normal([100, 1000])
    cpu_b = tf.random.normal([1000, 200])
    print(cpu_a.device, cpu_b.device)


def run_cpu():
    with tf.device('/cpu:0'):
        c = tf.matmul(cpu_a, cpu_b)
    return c


cpu_time = timeit.timeit(run_cpu, number=10)
print('warning up', cpu_time)

cpu_time = timeit.timeit(run_cpu, number=10)
print('run time', cpu_time)

