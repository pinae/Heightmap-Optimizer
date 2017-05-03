#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, unicode_literals, print_function
import tensorflow as tf


def calculate_variance_tendency(heightmap):
    return 1.0 - (tf.reduce_max(heightmap) - tf.reduce_min(heightmap))


if __name__ == "__main__":
    test_input = tf.constant([[[[0.2], [0.2], [0.2], [0.3]],
                               [[0.2], [0.0], [0.2], [0.4]],
                               [[0.2], [0.2], [0.2], [0.3]],
                               [[0.1], [0.2], [0.3], [0.4]]]], dtype=tf.float32)
    sess = tf.Session()
    print(sess.run(calculate_variance_tendency(test_input), {}))
