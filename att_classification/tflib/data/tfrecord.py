from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import glob
import json
import multiprocessing
import os

import tensorflow as tf

from tflib.data.dataset import batch_dataset, Dataset


_N_CPU = multiprocessing.cpu_count()

_DEFAULT_READER_BUFFER_SIZE_BYTES = 256 * 1024

_DECODERS = {
    'png': {'decoder': tf.image.decode_png, 'decode_param': dict()},
    'jpg': {'decoder': tf.image.decode_jpeg, 'decode_param': dict()},
    'jpeg': {'decoder': tf.image.decode_jpeg, 'decode_param': dict()},
    'uint8': {'decoder': tf.decode_raw, 'decode_param': dict(out_type=tf.uint8)},
    'int64': {'decoder': tf.decode_raw, 'decode_param': dict(out_type=tf.int64)},
    'float32': {'decoder': tf.decode_raw, 'decode_param': dict(out_type=tf.float32)},
}


def tfrecord_batch_dataset(tfrecord_files,
                           infos,
                           compression_type,
                           batch_size,
                           prefetch_batch=_N_CPU + 1,
                           drop_remainder=True,
                           filter=None,
                           map_func=None,
                           num_threads=_N_CPU,
                           shuffle=True,
                           shuffle_buffer_size=None,
                           repeat=-1):
    """Tfrecord batch dataset.

    `infos` example:
        [{'name': 'img', 'decoder': tf.image.decode_png, 'decode_param': {}, 'shape': [112, 112, 1]},
         {'name': 'point', 'decoder': tf.decode_raw, 'decode_param': dict(out_type = tf.float32), 'shape':[136]}]
    """
    dataset = tf.data.TFRecordDataset(tfrecord_files,
                                      compression_type=compression_type,
                                      buffer_size=_DEFAULT_READER_BUFFER_SIZE_BYTES)

    features = {}
    for info in infos:
        features[info['name']] = tf.FixedLenFeature([], tf.string)

    def parse_func(serialized_example):
        example = tf.parse_single_example(serialized_example, features=features)

        feature_dict = {}
        for info in infos:
            name = info['name']
            decoder = info['decoder']
            decode_param = info['decode_param']
            shape = info['shape']

            feature = decoder(example[name], **decode_param)
            feature = tf.reshape(feature, shape)
            feature_dict[name] = feature

        return feature_dict

    dataset = dataset.map(parse_func, num_parallel_calls=num_threads)

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


class TfrecordData(Dataset):

    def __init__(self,
                 tfrecord_path,
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
        super(TfrecordData, self).__init__()

        info_file = os.path.join(tfrecord_path, 'info.json')
        infos, self._data_num, compression_type = self._parse_json(info_file)

        self._shapes = {info['name']: tuple(info['shape']) for info in infos}

        tfrecord_files = sorted(glob.glob(os.path.join(tfrecord_path, '*.tfrecord')))
        dataset = tfrecord_batch_dataset(tfrecord_files,
                                         infos,
                                         compression_type,
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

    def __len__(self):
        return self._data_num

    @property
    def shape(self):
        return self._shapes

    @staticmethod
    def _parse_old(json_file):
        with open(json_file.replace('info.json', 'info.txt')) as f:
            try:  # older version 1
                infos = json.load(f)
                for info in infos[0:-1]:
                    info['decoder'] = _DECODERS[info['dtype_or_format']]['decoder']
                    info['decode_param'] = _DECODERS[info['dtype_or_format']]['decode_param']
            except:  # older version 2
                f.seek(0)
                infos = ''
                for line in f.readlines():
                    infos += line.strip('\n')
                infos = eval(infos)

        data_num = infos[-1]['data_num']
        compression_type = tf.python_io.TFRecordOptions.compression_type_map[infos[-1]['compression_type']]
        infos[-1:] = []

        return infos, data_num, compression_type

    @staticmethod
    def _parse_json(json_file):
        try:
            with open(json_file) as f:
                info = json.load(f)
                infos = info['item']
                for i in infos:
                    i['decoder'] = _DECODERS[i['dtype_or_format']]['decoder']
                    i['decode_param'] = _DECODERS[i['dtype_or_format']]['decode_param']
                data_num = info['info']['data_num']
                compression_type = tf.python_io.TFRecordOptions.compression_type_map[info['info']['compression_type']]
        except:  # for older version
            infos, data_num, compression_type = TfrecordData._parse_old(json_file)

        return infos, data_num, compression_type
