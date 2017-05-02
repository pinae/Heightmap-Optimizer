#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, unicode_literals, print_function
import tensorflow as tf


def get_lake_map(heightmap):
    filter = tf.constant([[[[1, 0, 0, 0, 0, 0, 0, 0]], [[0, 1, 0, 0, 0, 0, 0, 0]], [[0, 0, 1, 0, 0, 0, 0, 0]]],
                          [[[0, 0, 0, 1, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 1, 0, 0, 0]]],
                          [[[0, 0, 0, 0, 0, 1, 0, 0]], [[0, 0, 0, 0, 0, 0, 1, 0]], [[0, 0, 0, 0, 0, 0, 0, 1]]]],
                         dtype=tf.float32)
    cropped = tf.slice(heightmap, [0, 1, 1, 0], [1, tf.shape(heightmap)[1]-2, tf.shape(heightmap)[2]-2, 1])
    min_surrounding_pixel = tf.reduce_min(tf.nn.conv2d(heightmap, filter, strides=[1, 1, 1, 1], padding='VALID'),
                                          axis=3)
    return tf.nn.relu(min_surrounding_pixel - cropped)


if __name__ == "__main__":
    test_input = tf.constant([[[[2], [2], [2], [3]],
                               [[2], [0], [2], [4]],
                               [[2], [2], [2], [3]],
                               [[1], [2], [3], [4]]]], dtype=tf.float32)
    sess = tf.Session()
    print(sess.run(get_lake_map(test_input), {}))
