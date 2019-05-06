from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import json
import os
import shutil

import numpy as np
from PIL import Image
import tensorflow as tf

__metaclass__ = type


DECODERS = {
    'png': {'decoder': tf.image.decode_png, 'decode_param': dict()},
    'jpg': {'decoder': tf.image.decode_jpeg, 'decode_param': dict()},
    'jpeg': {'decoder': tf.image.decode_jpeg, 'decode_param': dict()},
    'uint8': {'decoder': tf.decode_raw, 'decode_param': dict(out_type=tf.uint8)},
    'float32': {'decoder': tf.decode_raw, 'decode_param': dict(out_type=tf.float32)},
    'int64': {'decoder': tf.decode_raw, 'decode_param': dict(out_type=tf.int64)},
}
ALLOWED_TYPES = DECODERS.keys()


class BytesTfrecordCreator(object):
    """BytesTfrecordCreator.

    `compression_type`:
        0: NONE
        1: ZLIB
        2: GZIP
    """

    @staticmethod
    def bytes_feature(values):
        """Return a TF-Feature of bytes.

        Args:
          values: A byte string or list of byte strings.

        Returns:
          a TF-Feature.
        """
        if not isinstance(values, (tuple, list)):
            values = [values]
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))

    @staticmethod
    def bytes_tfexample(bytes_dict):
        """Convert bytes to tfexample.

        `bytes_dict` example:
            bytes_dict = {
                'img': img_bytes,
                'id': id_bytes,
                'attr': attr_bytes,
                'point': point_bytes
            }
        """
        feature_dict = {}
        for key, value in bytes_dict.items():
            feature_dict[key] = BytesTfrecordCreator.bytes_feature(value)
        return tf.train.Example(features=tf.train.Features(feature=feature_dict))

    def __init__(self, save_path, block_size=None, compression_type=0, overwrite_existing=False):
        self.save_path = save_path
        if os.path.exists(save_path):
            if not overwrite_existing:
                raise Exception('%s exists!' % save_path)
            else:
                shutil.rmtree(save_path)
                os.makedirs(save_path)
        else:
            os.makedirs(save_path)

        self.options = tf.python_io.TFRecordOptions(compression_type)
        self.info_f = open(os.path.join(save_path, 'info.txt'), 'w')

        self.feature_names = None
        self.info_names = []  # is the same as self.feature_names except for item order
        self.info_list = []

        self.data_num = 0
        self.block_num = 0
        self.block_size = [block_size, 2147483647][block_size is None]

        self.compression_type = compression_type

        self.closed = False

    def add(self, feature_bytes_dict):
        """Add example.

        `feature_bytes_dict` example:
            feature_bytes_dict = {
                'img': img_bytes,
                'id': id_bytes,
                'attr': attr_bytes,
                'point': point_bytes
            }
        """
        if self.data_num // self.block_size == self.block_num:
            self.block_num += 1

            if self.block_num > 1:
                self.writer.close()

            tf_record_path = os.path.join(self.save_path, 'data_%06d.tfrecord' % (self.block_num - 1))
            self.writer = tf.python_io.TFRecordWriter(tf_record_path, self.options)

        if self.feature_names is None:
            self.feature_names = feature_bytes_dict.keys()
        else:
            assert self.feature_names == feature_bytes_dict.keys(), \
                'Feature names are inconsistent!'

        tfexample = BytesTfrecordCreator.bytes_tfexample(feature_bytes_dict)
        self.writer.write(tfexample.SerializeToString())
        self.data_num += 1

    def add_info(self, name, dtype_or_format, shape):
        """Add feature informations.

        example:
            add_info('img', 'png', [64, 64, 3])
        """
        assert name not in self.info_names, 'info name duplicated!'

        dtype_or_format = dtype_or_format.lower()
        assert dtype_or_format in ALLOWED_TYPES, \
            "`dtype_or_format` should be in the list of %s!" \
            % str(ALLOWED_TYPES)

        self.info_names.append(name)
        self.info_list.append(dict(name=name,
                                   dtype_or_format=dtype_or_format,
                                   shape=shape))

    def close(self):
        assert sorted(self.feature_names) == sorted(self.info_names), \
            "Feature informations should be added by function 'add_info(...)!'"

        # save info
        self.info_list.append({'data_num': self.data_num,
                               'compression_type': self.compression_type})
        info_str = json.dumps(self.info_list)
        info_str = info_str.replace('}, {', '},\n {')
        self.info_f.write(info_str)

        # close files
        self.writer.close()
        self.info_f.close()

        self.closed = True

    def __del__(self):
        if not self.closed:
            self.close()


class DataLablePairTfrecordCreator(BytesTfrecordCreator):
    """DataLablePairTfrecordCreator.

    If `data_shape` is None, then the `data` to be added should be given as
    numpy array, and the shape and dtype of `data` will be inferred.
    If `data_shape` is not None, `data` should be given as byte string,
    and `data_dtype_or_format` should also be given.

    `compression_type`:
        0: NONE
        1: ZLIB
        2: GZIP
    """

    def __init__(self, save_path, data_shape=None, data_dtype_or_format=None,
                 data_name='data', block_size=None, compression_type=0, overwrite_existing=False):
        super(DataLablePairTfrecordCreator, self).__init__(save_path,
                                                           block_size,
                                                           compression_type,
                                                           overwrite_existing)
        if data_shape is not None:
            assert data_dtype_or_format is not None, \
                '`data_dtype_or_format` should be given when `data_shape` is given!'
            self.check_data = False
        else:
            self.check_data = True

        self.data_shape = data_shape
        self.data_dtype_or_format = data_dtype_or_format
        self.label_shape_dict = {}
        self.label_dtype_dict = {}
        self.data_name = data_name
        self.info_built = False

    def add(self, data, label_dict):
        """Add example.

        If `self.data_shape` is initialized as None, then the `data` to be
        added should be given as numpy array, and the shape and dtype of `data`
        will be inferred.
        If `self.data_shape` is not initialized as None, `data` should be given as
        byte string.

        `label_dict` each value should be a numpy array, shape and dtype will be inferred.
        """
        if self.check_data:
            assert isinstance(data, np.ndarray), '`data` should be numpy array!'
        else:
            assert isinstance(data, (str, bytes)), '`data` should be byte string!'
        for label in label_dict.values():
            assert isinstance(label, np.ndarray), '`label` should be numpy array!'

        if not self.info_built:
            if self.data_shape is None:
                self.data_shape = data.shape
                self.data_dtype_or_format = data.dtype.name
            for label_name, label in label_dict.items():
                self.label_shape_dict[label_name] = label.shape
                self.label_dtype_dict[label_name] = label.dtype.name
            self.info_built = True

        if self.check_data:
            assert data.shape == tuple(self.data_shape), \
                'shape of `data` should be %s!' % str(tuple(self.data_shape))
            assert data.dtype.name == self.data_dtype_or_format, \
                'dtype of `data` should be %s!' % self.data_dtype_or_format
            data = data.tobytes()

        feature_dict = {self.data_name: data}
        for label_name, label in label_dict.items():
            assert label.shape == tuple(self.label_shape_dict[label_name]), \
                'shape of `%s` should be %s!' % (label_name, str(tuple(self.label_shape_dict[label_name])))
            assert label.dtype.name == self.label_dtype_dict[label_name], \
                'dtype of `%s` should be %s!' % (label_name, self.label_dtype_dict[label_name])
            feature_dict[label_name] = label.tobytes()

        super(DataLablePairTfrecordCreator, self).add(feature_dict)

    def close(self):
        self.add_info(self.data_name, self.data_dtype_or_format, self.data_shape)
        for label_name in self.label_shape_dict.keys():
            self.add_info(label_name, self.label_dtype_dict[label_name], self.label_shape_dict[label_name])

        super(DataLablePairTfrecordCreator, self).close()


class ImageLablePairTfrecordCreator(DataLablePairTfrecordCreator):
    """ImageLablePairTfrecordCreator.

    `encode_type`: in [None, 'png', 'jpg', 'jpeg'].
    `quality`: for 'jpg' or 'jpeg'.

    `compression_type`:
        0: NONE
        1: ZLIB
        2: GZIP
    """

    def __init__(self, save_path, encode_type, quality=95, data_name='data',
                 block_size=None, compression_type=0, overwrite_existing=False):
        super(ImageLablePairTfrecordCreator, self).__init__(
            save_path, None, None, data_name, block_size, compression_type, overwrite_existing)

        if isinstance(encode_type, str):
            encode_type = encode_type.lower()
        assert encode_type in [None, 'png', 'jpg', 'jpeg'], \
            ("`encode_type` should be in the list of"
             " [None, 'png', 'jpg', 'jpeg']!")

        self.encode_type = encode_type
        self.quality = quality

    def add(self, data, label_dict):
        """Add example.

        `data`: H * W (* C) uint8 numpy array.
        `label_dict`: each value should be a numpy array.
        """
        assert data.dtype == np.uint8 and data.ndim in [2, 3], \
            '`data`: H * W (* C) uint8 numpy array!'

        if data.ndim == 2:
            data.shape = data.shape + (1,)

        if self.data_shape is None:
            self.data_shape = data.shape
            if self.encode_type is None:
                self.data_dtype_or_format = data.dtype.name
            else:
                self.data_dtype_or_format = self.encode_type
            self.check_data = False

        # tobytes
        if self.encode_type is not None:
            if data.ndim == 3:
                if data.shape[-1] == 1:
                    data.shape = data.shape[:2]
                elif data.shape[-1] != 3:
                    raise Exception('Only images with 1 or 3 '
                                    'channels are allowed to be encoded!')

            byte = io.BytesIO()
            data = Image.fromarray(data)
            if self.encode_type in ['jpg', 'jpeg']:
                data.save(byte, 'JPEG', quality=self.quality)
            elif self.encode_type == 'png':
                data.save(byte, 'PNG')
            data = byte.getvalue()
        else:
            data = data.tobytes()

        super(ImageLablePairTfrecordCreator, self).add(data, label_dict)


def tfrecord_batch(tfrecord_files, info_list, batch_size, preprocess_fns={},
                   shuffle=True, num_threads=16, min_after_dequeue=5000,
                   scope=None, compression_type=0):
    """Tfrecord batch ops.

    info_list:
        for example
        [{'name': 'img', 'decoder': tf.image.decode_png, 'decode_param': {}, 'shape': [112, 112, 1]},
         {'name': 'point', 'decoder': tf.decode_raw, 'decode_param': dict(out_type = tf.float32), 'shape':[136]}]

    preprocess_fns:
        for example
        {'img': img_preprocess_fn, 'point': point_preprocess_fn}
    """
    with tf.name_scope(scope, 'tfrecord_batch'):
        options = tf.python_io.TFRecordOptions(compression_type)

        features = {}
        fields = []
        for info in info_list:
            features[info['name']] = tf.FixedLenFeature([], tf.string)
            fields += [info['name']]

        # read the next record (there is only one tfrecord file in the file queue)
        _, serialized_example = tf.TFRecordReader(options=options).read(
            tf.train.string_input_producer(tfrecord_files,
                                           shuffle=shuffle,
                                           capacity=len(tfrecord_files)))

        # parse the record
        features = tf.parse_single_example(serialized_example, features=features)

        # decode, set shape and preprocess
        data_dict = {}
        for info in info_list:
            name = info['name']
            decoder = info['decoder']
            decode_param = info['decode_param']
            shape = info['shape']

            feature = decoder(features[name], **decode_param)
            feature = tf.reshape(feature, shape)
            if name in preprocess_fns:
                feature = preprocess_fns[name](feature)
            data_dict[name] = feature

        # batch datas
        if shuffle:
            capacity = min_after_dequeue + (num_threads + 1) * batch_size
            data_batch = tf.train.shuffle_batch(data_dict,
                                                batch_size=batch_size,
                                                capacity=capacity,
                                                min_after_dequeue=min_after_dequeue,
                                                num_threads=num_threads)
        else:
            data_batch = tf.train.batch(data_dict, batch_size=batch_size)

        return data_batch, fields


class TfrecordData(object):
    """TfrecordData.

    preprocess_fns:
        for example
        {'img': img_preprocess_fn, 'point': point_preprocess_fn}
    """

    def __init__(self, tfrecord_path, batch_size, preprocess_fns={},
                 shuffle=True, num_threads=16, min_after_dequeue=5000, scope=None):
        # info
        tfrecord_info_file = os.path.join(tfrecord_path, 'info.txt')
        tfrecord_files = sorted(os.listdir(tfrecord_path))
        tfrecord_files.remove('info.txt')
        tfrecord_files = [os.path.join(tfrecord_path, t) for t in tfrecord_files]

        with open(tfrecord_info_file) as f:
            try:  # for new version
                info_list = json.load(f)
                for info in info_list[0:-1]:
                    info['decoder'] = DECODERS[info['dtype_or_format']]['decoder']
                    info['decode_param'] = DECODERS[info['dtype_or_format']]['decode_param']
            except:
                f.seek(0)
                info_list = ''
                for line in f.readlines():
                    info_list += line.strip('\n')
                info_list = eval(info_list)

        self.data_num = info_list[-1]['data_num']
        compression_type = info_list[-1]['compression_type']
        info_list[-1:] = []

        # shapes
        self.shapes = {info['name']: tuple(info['shape']) for info in info_list}

        # graph
        self.graph = tf.Graph()  # declare ops in a separated graph
        with self.graph.as_default():
            # TODO
            # There are some strange errors if the gpu device is the
            # same with the main graph, but cpu device is ok. I don't know why...
            with tf.device('/cpu:0'):
                self.batch_ops, self._fields = tfrecord_batch(tfrecord_files, info_list, batch_size,
                                                              preprocess_fns, shuffle, num_threads,
                                                              min_after_dequeue, scope, compression_type)

        print(' [*] TfrecordData: create session!')

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.sess,
                                                    coord=self.coord)

        # iterator
        self.batch_num_per_epoch = (self.data_num + batch_size - 1) // batch_size
        self.current_batch_id = 0

    def __len__(self):
        return self.data_num

    def n_batch(self):
        return self.batch_num_per_epoch

    def batch(self, fields=None):
        batch_data = self.sess.run(self.batch_ops)
        if fields is None:
            # return a dict
            return batch_data
        elif isinstance(fields, (list, tuple)):
            return [batch_data[field] for field in fields]
        else:
            return batch_data[fields]

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_batch_id >= self.batch_num_per_epoch:
            self.current_batch_id = 0
            raise StopIteration
        self.current_batch_id += 1
        return self.batch()

    next = __next__

    def fields(self):
        return self._fields

    def shape(self, field=None):
        if field is None:
            return self.shapes
        else:
            return self.shapes[field]

    def __del__(self):
        print(' [*] TfrecordData: stop threads and close session!')
        self.coord.request_stop()
        self.coord.join(self.threads)
        self.sess.close()
