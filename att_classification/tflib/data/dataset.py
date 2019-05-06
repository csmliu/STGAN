from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing

import tensorflow as tf
import tensorflow.contrib.eager as tfe

from tflib.utils import session


_N_CPU = multiprocessing.cpu_count()


def batch_dataset(dataset,
                  batch_size,
                  prefetch_batch=_N_CPU + 1,
                  drop_remainder=True,
                  filter=None,
                  map_func=None,
                  num_threads=_N_CPU,
                  shuffle=True,
                  shuffle_buffer_size=None,
                  repeat=-1):
    if filter:
        dataset = dataset.filter(filter)

    if map_func:
        dataset = dataset.map(map_func, num_parallel_calls=num_threads)

    if shuffle:
        if shuffle_buffer_size is None:
            shuffle_buffer_size = batch_size * 100
        dataset = dataset.shuffle(shuffle_buffer_size)

    if drop_remainder:
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    else:
        dataset = dataset.batch(batch_size)

    dataset = dataset.repeat(repeat).prefetch(prefetch_batch)

    return dataset


class Dataset(object):

    def __init__(self):
        self._dataset = None
        self._iterator = None
        self._batch_op = None
        self._sess = None

        self._is_eager = tf.executing_eagerly()
        self._eager_iterator = None

    def __del__(self):
        if self._sess:
            self._sess.close()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            b = self.get_next()
        except:
            raise StopIteration
        else:
            return b

    next = __next__

    def get_next(self):
        if self._is_eager:
            return self._eager_iterator.get_next()
        else:
            return self._sess.run(self._batch_op)

    def reset(self, feed_dict={}):
        if self._is_eager:
            self._eager_iterator = tfe.Iterator(self._dataset)
        else:
            self._sess.run(self._iterator.initializer, feed_dict=feed_dict)

    def _bulid(self, dataset, sess=None):
        self._dataset = dataset

        if self._is_eager:
            self._eager_iterator = tfe.Iterator(dataset)
        else:
            self._iterator = dataset.make_initializable_iterator()
            self._batch_op = self._iterator.get_next()
            if sess:
                self._sess = sess
            else:
                self._sess = session()

        try:
            self.reset()
        except:
            pass

    @property
    def dataset(self):
        return self._dataset

    @property
    def iterator(self):
        return self._iterator

    @property
    def batch_op(self):
        return self._batch_op
