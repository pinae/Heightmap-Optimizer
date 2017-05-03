#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, unicode_literals, print_function
import tensorflow as tf


def calculate_lake_constraint(heightmap):
    filters = tf.constant([[[[1, 0, 0, 0, 0, 0, 0, 0]], [[0, 1, 0, 0, 0, 0, 0, 0]], [[0, 0, 1, 0, 0, 0, 0, 0]]],
                           [[[0, 0, 0, 1, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 1, 0, 0, 0]]],
                           [[[0, 0, 0, 0, 0, 1, 0, 0]], [[0, 0, 0, 0, 0, 0, 1, 0]], [[0, 0, 0, 0, 0, 0, 0, 1]]]],
                          dtype=tf.float32)
    surrounding_pixels = tf.nn.conv2d(heightmap, filters, strides=[1, 1, 1, 1], padding='VALID')
    min_surrounding_pixels = tf.expand_dims(tf.reduce_min(surrounding_pixels, axis=3), axis=3)
    cropped = tf.slice(heightmap, [0, 1, 1, 0], [1, tf.shape(heightmap)[1] - 2, tf.shape(heightmap)[2] - 2, 1])
    lake_constraint_map = tf.nn.relu(min_surrounding_pixels - cropped)
    return tf.reduce_sum(lake_constraint_map)


if __name__ == "__main__":
    test_input = tf.constant([[[[2], [2], [2], [3]],
                               [[2], [0], [2], [4]],
                               [[2], [2], [2], [3]],
                               [[1], [2], [3], [4]]]], dtype=tf.float32)
    sess = tf.Session()
    print(sess.run(calculate_lake_constraint(test_input), {}))
