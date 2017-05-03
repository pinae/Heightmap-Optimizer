#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, unicode_literals, print_function
import tensorflow as tf


def get_lake_map(heightmap):
    filters = tf.constant([[[[1, 0, 0, 0, 0, 0, 0, 0]], [[0, 1, 0, 0, 0, 0, 0, 0]], [[0, 0, 1, 0, 0, 0, 0, 0]]],
                           [[[0, 0, 0, 1, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 1, 0, 0, 0]]],
                           [[[0, 0, 0, 0, 0, 1, 0, 0]], [[0, 0, 0, 0, 0, 0, 1, 0]], [[0, 0, 0, 0, 0, 0, 0, 1]]]],
                          dtype=tf.float32)
    surrounding_pixels = tf.nn.conv2d(heightmap, filters, strides=[1, 1, 1, 1], padding='VALID')
    min_surrounding_pixels = tf.expand_dims(tf.reduce_min(surrounding_pixels, axis=3), axis=3)
    max_surrounding_pixels = tf.expand_dims(tf.reduce_max(surrounding_pixels, axis=3), axis=3)
    cropped = tf.slice(heightmap, [0, 1, 1, 0], [1, tf.shape(heightmap)[1] - 2, tf.shape(heightmap)[2] - 2, 1])
    return 10 * tf.nn.relu(cropped - max_surrounding_pixels + 0.1) + \
           10 * tf.nn.relu(min_surrounding_pixels - cropped + 0.1) + \
           100 * tf.nn.relu(min_surrounding_pixels - cropped)


if __name__ == "__main__":
    test_input = tf.constant([[[[2], [2], [2], [3]],
                               [[2], [0], [2], [4]],
                               [[2], [2], [2], [3]],
                               [[1], [2], [3], [4]]]], dtype=tf.float32)
    sess = tf.Session()
    print(sess.run(get_lake_map(test_input), {}))
