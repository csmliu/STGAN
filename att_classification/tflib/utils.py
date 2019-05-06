from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def session(graph=None,
            allow_soft_placement=True,
            log_device_placement=False,
            allow_growth=True):
    """Return a Session with simple config."""
    config = tf.ConfigProto(allow_soft_placement=allow_soft_placement,
                            log_device_placement=log_device_placement)
    config.gpu_options.allow_growth = allow_growth
    return tf.Session(graph=graph, config=config)


def print_tensor(tensors):
    if not isinstance(tensors, (list, tuple)):
        tensors = [tensors]

    for i, tensor in enumerate(tensors):
        ctype = str(type(tensor))
        if 'Tensor' in ctype:
            print('%d: %s("%s", shape=%s, dtype=%s, device=%s)' %
                  (i, 'Tensor', tensor.name, tensor.shape, tensor.dtype.name, tensor.device))
        elif 'Variable' in ctype:
            print('%d: %s("%s", shape=%s, dtype=%s, device=%s)' %
                  (i, 'Variable', tensor.name, tensor.shape, tensor.dtype.name, tensor.device))
        elif 'Operation' in ctype:
            print('%d: %s("%s", device=%s)' %
                  (i, 'Operation', tensor.name, tensor.device))
        else:
            raise Exception('Not a Tensor, Variable or Operation!')


prt = print_tensor


def shape(tensor):
    sp = tensor.get_shape().as_list()
    return [num if num is not None else -1 for num in sp]


def summary(tensor_collection,
            summary_type=['mean', 'stddev', 'max', 'min', 'sparsity', 'histogram'],
            scope=None):
    """Summary.

    Usage:
        1. summary(tensor)
        2. summary([tensor_a, tensor_b])
        3. summary({tensor_a: 'a', tensor_b: 'b})
    """
    def _summary(tensor, name, summary_type):
        """Attach a lot of summaries to a Tensor."""
        if name is None:
            name = tensor.name

        summaries = []
        if len(tensor.shape) == 0:
            summaries.append(tf.summary.scalar(name, tensor))
        else:
            if 'mean' in summary_type:
                mean = tf.reduce_mean(tensor)
                summaries.append(tf.summary.scalar(name + '/mean', mean))
            if 'stddev' in summary_type:
                mean = tf.reduce_mean(tensor)
                stddev = tf.sqrt(tf.reduce_mean(tf.square(tensor - mean)))
                summaries.append(tf.summary.scalar(name + '/stddev', stddev))
            if 'max' in summary_type:
                summaries.append(tf.summary.scalar(name + '/max', tf.reduce_max(tensor)))
            if 'min' in summary_type:
                summaries.append(tf.summary.scalar(name + '/min', tf.reduce_min(tensor)))
            if 'sparsity' in summary_type:
                summaries.append(tf.summary.scalar(name + '/sparsity', tf.nn.zero_fraction(tensor)))
            if 'histogram' in summary_type:
                summaries.append(tf.summary.histogram(name, tensor))
        return tf.summary.merge(summaries)

    if not isinstance(tensor_collection, (list, tuple, dict)):
        tensor_collection = [tensor_collection]

    with tf.name_scope(scope, 'summary'):
        summaries = []
        if isinstance(tensor_collection, (list, tuple)):
            for tensor in tensor_collection:
                summaries.append(_summary(tensor, None, summary_type))
        else:
            for tensor, name in tensor_collection.items():
                summaries.append(_summary(tensor, name, summary_type))
        return tf.summary.merge(summaries)


def counter(start=0, scope=None):
    with tf.variable_scope(scope, 'counter'):
        counter = tf.get_variable(name='counter',
                                  initializer=tf.constant_initializer(start),
                                  shape=(),
                                  dtype=tf.int64)
        update_cnt = tf.assign(counter, tf.add(counter, 1))
        return counter, update_cnt
