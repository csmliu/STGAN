from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing

import numpy as np
import tensorflow as tf

from tflib.data.dataset import batch_dataset, Dataset


_N_CPU = multiprocessing.cpu_count()


def memory_data_batch_dataset(memory_data_dict,
                              batch_size,
                              prefetch_batch=_N_CPU + 1,
                              drop_remainder=True,
                              filter=None,
                              map_func=None,
                              num_threads=_N_CPU,
                              shuffle=True,
                              shuffle_buffer_size=None,
                              repeat=-1):
    """Memory data batch dataset.

    `memory_data_dict` example:
        {'img': img_ndarray, 'label': label_ndarray} or
        {'img': img_tftensor, 'label': label_tftensor}
        * The value of each item of `memory_data_dict` is in shape of (N, ...).
    """
    dataset = tf.data.Dataset.from_tensor_slices(memory_data_dict)
    dataset = batch_dataset(dataset,
                            batch_size,
                            prefetch_batch,
                            drop_remainder,
                            filter,
                            map_func,
                            num_threads,
                            shuffle,
                            shuffle_buffer_size,
                            repeat)
    return dataset


class MemoryData(Dataset):
    """MemoryData.

    `memory_data_dict` example:
        {'img': img_ndarray, 'label': label_ndarray} or
        {'img': img_tftensor, 'label': label_tftensor}
        * The value of each item of `memory_data_dict` is in shape of (N, ...).
    """

    def __init__(self,
                 memory_data_dict,
                 batch_size,
                 prefetch_batch=_N_CPU + 1,
                 drop_remainder=True,
                 filter=None,
                 map_func=None,
                 num_threads=_N_CPU,
                 shuffle=True,
                 shuffle_buffer_size=None,
                 repeat=-1,
                 sess=None):
        super(MemoryData, self).__init__()
        dataset = memory_data_batch_dataset(memory_data_dict,
                                            batch_size,
                                            prefetch_batch,
                                            drop_remainder,
                                            filter,
                                            map_func,
                                            num_threads,
                                            shuffle,
                                            shuffle_buffer_size,
                                            repeat)
        self._bulid(dataset, sess)
        if isinstance(list(memory_data_dict.values())[0], np.ndarray):
            self._n_data = len(list(memory_data_dict.values())[0])
        else:
            self._n_data = list(memory_data_dict.values())[0].get_shape().as_list()[0]

    def __len__(self):
        return self._n_data

if __name__ == '__main__':
    data = {'a': np.array([1.0, 2, 3, 4, 5]),
            'b': np.array([[1, 2],
                           [2, 3],
                           [3, 4],
                           [4, 5],
                           [5, 6]])}

    def filter(x):
        return tf.cond(x['a'] > 2, lambda: tf.constant(True), lambda: tf.constant(False))

    def map_func(x):
        x['a'] = x['a'] * 10
        return x

    # tf.enable_eager_execution()

    s = tf.Session()

    dataset = MemoryData(data,
                         2,
                         filter=None,
                         map_func=map_func,
                         shuffle=True,
                         shuffle_buffer_size=None,
                         drop_remainder=True,
                         repeat=4,
                         sess=s)

    for i in range(5):
        print(map(dataset.get_next().__getitem__, ['b', 'a']))

    print([n.name for n in tf.get_default_graph().as_graph_def().node])
