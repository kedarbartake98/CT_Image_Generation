"""
Core network which predicts rewards from frames,
for gym-moving-dot and Atari games.
"""

import tensorflow as tf

from reinforcement_learning.nn_layers import dense_layer, conv_layer

def net_cnn(s, batchnorm, dropout, training, reuse):
    x = s / 255.0
    x = tf.reshape(x, (-1, 256, 256, 1))
    x = conv_layer(x, 8, 3, 1, batchnorm, training, "c1", reuse, 'relu')
    x = tf.compat.v1.layers.max_pooling2d(x, 2, 2, padding='same')
    x = conv_layer(x, 16, 3, 1, batchnorm, training, "c2", reuse, 'relu')
    x = tf.compat.v1.layers.max_pooling2d(x, 2, 2, padding='same')
    x = conv_layer(x, 16, 3, 1, batchnorm, training, "c3", reuse, 'relu')
    x = tf.compat.v1.layers.max_pooling2d(x, 2, 2, padding='same')
    x = conv_layer(x, 32, 3, 1, batchnorm, training, "c4", reuse, 'relu')
    x = tf.compat.v1.layers.max_pooling2d(x, 2, 2, padding='same')
    x = conv_layer(x, 32, 3, 1, batchnorm, training, "c5", reuse, 'relu')
    x = tf.compat.v1.layers.max_pooling2d(x, 2, 2, padding='same')
    x = conv_layer(x, 16, 3, 1, batchnorm, training, "c6", reuse, 'relu')
    x = tf.compat.v1.layers.max_pooling2d(x, 2, 2, padding='same')
    x = conv_layer(x, 8, 3, 1, batchnorm, training, "c7", reuse, 'relu')
    x = conv_layer(x, 2, 3, 1, batchnorm, training, "c8", reuse, 'relu')
    x = tf.compat.v1.layers.average_pooling2d(x, 4, 4, padding='same')

    w, h, c = x.get_shape()[1:]
    x = tf.reshape(x, [-1, int(w * h * c)])

    x = dense_layer(x, 1, "d1", reuse, activation='relu')
    x = x[:, 0]

    return x