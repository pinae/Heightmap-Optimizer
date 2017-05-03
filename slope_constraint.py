#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, unicode_literals, print_function
import tensorflow as tf


def calculate_slope_constraint(heightmap, max_slope=0.2):
    filters = tf.constant([[[[1, 0, 0, 0, 0, 0, 0, 0]], [[0, 1, 0, 0, 0, 0, 0, 0]], [[0, 0, 1, 0, 0, 0, 0, 0]]],
                           [[[0, 0, 0, 1, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 1, 0, 0, 0]]],
                           [[[0, 0, 0, 0, 0, 1, 0, 0]], [[0, 0, 0, 0, 0, 0, 1, 0]], [[0, 0, 0, 0, 0, 0, 0, 1]]]],
                          dtype=tf.float32)
    surrounding_pixels = tf.expand_dims(tf.nn.conv2d(heightmap, filters, strides=[1, 1, 1, 1], padding='VALID'), axis=3)
    cropped = tf.slice(heightmap, [0, 1, 1, 0], [1, tf.shape(heightmap)[1] - 2, tf.shape(heightmap)[2] - 2, 1])
    slope_constraint_map = tf.nn.relu(
        tf.abs(tf.stack([cropped for _ in range(8)], axis=4) - surrounding_pixels) - max_slope)
    return tf.reduce_sum(slope_constraint_map)


if __name__ == "__main__":
    test_input = tf.constant([[[[0.2], [0.2], [0.2], [0.3]],
                               [[0.2], [0.0], [0.2], [0.4]],
                               [[0.2], [0.2], [0.2], [0.3]],
                               [[0.1], [0.2], [0.3], [0.4]]]], dtype=tf.float32)
    sess = tf.Session()
    print(sess.run(calculate_slope_constraint(test_input), {}))
