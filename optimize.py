#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import unicode_literals, division, print_function
import numpy as np
import tensorflow as tf
from PIL import Image
from lake_constraint import calculate_lake_constraint
from slope_constraint import calculate_slope_constraint


def create_noise_map(dimensions):
    return np.random.random(size=(1, dimensions[1], dimensions[0], 1)) * 256


class HeightmapOptimizer:
    def __init__(self, init_map):
        self.heightmap = tf.Variable(initial_value=tf.constant(init_map, dtype=tf.float32) / 256, name="Heightmap")
        self.loss = 5 * calculate_lake_constraint(self.heightmap) + calculate_slope_constraint(self.heightmap, 0.05)
        self.learning_rate = 0.1
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate,
                                                beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(self.loss)
        init_op = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init_op)

    def optimize_step(self):
        _, heightmap = self.sess.run([self.optimizer, self.heightmap * 256], feed_dict={})
        return heightmap


if __name__ == "__main__":
    initial_map = create_noise_map((640, 480))
    optimizer = HeightmapOptimizer(initial_map)
    for i in range(300):
        height_map = optimizer.optimize_step()
        x = np.clip(height_map.reshape((initial_map.shape[1], initial_map.shape[2])), 0.0, 255.0).astype(np.uint8)
        x = np.dstack((x, x, x))
        im = Image.fromarray(x)
        im.save("images/image" + str(i) + ".jpg", quality=95)
