#!/usr/bin/env python3

#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

obs_dim = 128
act_dim = 3 * 3 * 4

obs = tf.placeholder(shape=(None, obs_dim), dtype=tf.float32)
net = mlp(obs, hidden_dims=(64,64), activation=tf.tanh)
actions = tf.layers.dense(net, units=act_dim, activation=None)
