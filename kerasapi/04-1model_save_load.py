import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
network.save_weights('weights.ckpt')
print('saved weights.')
del network

network = Sequential([
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(10)
])

network.compile(optimizer=optimizers.Adam(lr=0.001),loss=tf.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
network.load_weight('weights.ckpt')
network.evaluate(ds_val)