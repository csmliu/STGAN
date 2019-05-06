from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def minmax_norm(x, epsilon=1e-12):
    x = tf.to_float(x)
    min_val = tf.reduce_min(x)
    max_val = tf.reduce_max(x)
    x_norm = (x - min_val) / tf.maximum((max_val - min_val), epsilon)
    return x_norm
