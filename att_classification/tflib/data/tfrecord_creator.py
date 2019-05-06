from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import json
import os
import shutil

import numpy as np
import tensorflow as tf

from PIL import Image
from tflib.data import tfrecord

__metaclass__ = type


_ALLOWED_TYPES = tfrecord._DECODERS.keys()


class BytesTfrecordCreator(object):
    """BytesTfrecordCreator.

    `infos` example:
        infos = [
            ['img', 'jpg', (64, 64, 3)],
            ['id', 'int64', ()],
            ['attr', 'int64', (40,)],
            ['point', 'float32', (5, 2)]
        ]

    `compression_type`:
        0 : NONE
        1 : ZLIB
        2 : GZIP
    """

    def __init__(self,
                 save_path,
                 infos,
                 size_each=None,
                 compression_type=0,
                 overwrite_existence=False):
        # overwrite existence
        if os.path.exists(save_path):
            if not overwrite_existence:
                raise Exception('%s exists!' % save_path)
            else:
                shutil.rmtree(save_path)
                os.makedirs(save_path)
        else:
            os.makedirs(save_path)

        self._save_path = save_path

        # add info
        self._infos = []
        self._info_names = []
        for info in infos:
            self._add_info(*info)

        self._data_num = 0
        self._tfrecord_num = 0
        self._size_each = [size_each, 2147483647][not size_each]
        self._writer = None

        self._compression_type = compression_type
        self._options = tf.python_io.TFRecordOptions(compression_type)

    def __del__(self):
        info = {'item': self._infos, 'info': {'data_num': self._data_num, 'compression_type': self._compression_type}}
        info_str = json.dumps(info, indent=4, separators=(',', ':'))

        with open(os.path.join(self._save_path, 'info.json'), 'w') as info_f:
            info_f.write(info_str)

        if self._writer:
            self._writer.close()

    def add(self, feature_bytes_dict):
        """Add example.

        `feature_bytes_dict` example:
            feature_bytes_dict = {
                'img'   : img_bytes,
                'id'    : id_bytes,
                'attr'  : attr_bytes,
                'point' : point_bytes
            }
        """
        assert sorted(self._info_names) == sorted(feature_bytes_dict.keys()), \
            'Feature names are inconsistent with the givens!'

        self._new_tfrecord_check()

        self._writer.write(self._bytes_tfexample(feature_bytes_dict).SerializeToString())
        self._data_num += 1

    def _new_tfrecord_check(self):
        if self._data_num // self._size_each == self._tfrecord_num:
            self._tfrecord_num += 1

            if self._writer:
                self._writer.close()

            tfrecord_path = os.path.join(self._save_path, 'data_%06d.tfrecord' % (self._tfrecord_num - 1))
            self._writer = tf.python_io.TFRecordWriter(tfrecord_path, self._options)

    def _add_info(self, name, dtype_or_format, shape):
        assert name not in self._info_names, 'Info name "%s" is duplicated!' % name
        assert dtype_or_format in _ALLOWED_TYPES, 'Allowed data types: %s!' % str(_ALLOWED_TYPES)
        self._infos.append(dict(name=name, dtype_or_format=dtype_or_format, shape=shape))
        self._info_names.append(name)

    @staticmethod
    def _bytes_feature(values):
        """Return a TF-Feature of bytes.

        Arguments:
            values : A byte string or list of byte strings.

        Returns:
            A TF-Feature.
        """
        if not isinstance(values, (tuple, list)):
            values = [values]
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))

    @staticmethod
    def _bytes_tfexample(bytes_dict):
        """Convert bytes to tfexample.

        `bytes_dict` example:
            bytes_dict = {
                'img'   : img_bytes,
                'id'    : id_bytes,
                'attr'  : attr_bytes,
                'point' : point_bytes
            }
        """
        feature_dict = {}
        for key, value in bytes_dict.items():
            feature_dict[key] = BytesTfrecordCreator._bytes_feature(value)
        return tf.train.Example(features=tf.train.Features(feature=feature_dict))


class DataLablePairTfrecordCreator(BytesTfrecordCreator):
    """DataLablePairTfrecordCreator.

    If `data_shape` is None, then the `data` to be added should be a
    numpy array, and the shape and dtype of `data` will be inferred.
    If `data_shape` is not None, `data` should be given as byte string,
    and `data_dtype_or_format` should also be given.

    `compression_type`:
        0 : NONE
        1 : ZLIB
        2 : GZIP
    """

    def __init__(self,
                 save_path,
                 data_shape=None,
                 data_dtype_or_format=None,
                 data_name='data',
                 size_each=None,
                 compression_type=0,
                 overwrite_existence=False):
        super(DataLablePairTfrecordCreator, self).__init__(save_path,
                                                           [],
                                                           size_each,
                                                           compression_type,
                                                           overwrite_existence)

        if data_shape:
            assert data_dtype_or_format, '`data_dtype_or_format` should be given when `data_shape` is given!'
            self._is_data_bytes = True
        else:
            self._is_data_bytes = False

        self._data_shape = data_shape
        self._data_dtype_or_format = data_dtype_or_format
        self._data_name = data_name
        self._label_shape_dict = {}
        self._label_dtype_dict = {}

        self._info_built = False

    def add(self, data, label_dict):
        """Add example.

        `label_dict` example:
            label_dict = {
                'id'    : id_ndarray,
                'attr'  : attr_ndarray,
                'point' : point_ndarray
            }
        """
        self._check_and_build(data, label_dict)

        if not self._is_data_bytes:
            data = data.tobytes()

        feature_dict = {self._data_name: data}
        for name, label in label_dict.items():
            feature_dict[name] = label.tobytes()

        super(DataLablePairTfrecordCreator, self).add(feature_dict)

    def _check_and_build(self, data, label_dict):
        # check type
        if self._is_data_bytes:
            assert isinstance(data, (str, bytes)), '`data` should be a byte string!'
        else:
            assert isinstance(data, np.ndarray), '`data` should be a numpy array!'
        for label in label_dict.values():
            assert isinstance(label, np.ndarray), 'labels should be numpy arrays!'

        # check shape and dtype or bulid info at first adding
        if self._info_built:
            if not self._is_data_bytes:
                assert data.shape == tuple(self._data_shape), 'Shapes of `data`s are inconsistent!'
                assert data.dtype.name == self._data_dtype_or_format, 'Dtypes of `data`s are inconsistent!'
            for name, label in label_dict.items():
                assert label.shape == self._label_shape_dict[name], 'Shapes of `%s`s are inconsistent!' % name
                assert label.dtype.name == self._label_dtype_dict[name], 'Dtypes of `%s`s are inconsistent!' % name
        else:
            if not self._is_data_bytes:
                self._data_shape = data.shape
                self._data_dtype_or_format = data.dtype.name
            self._add_info(self._data_name, self._data_dtype_or_format, self._data_shape)

            for name, label in label_dict.items():
                self._label_shape_dict[name] = label.shape
                self._label_dtype_dict[name] = label.dtype.name
                self._add_info(name, label.dtype.name, label.shape)

            self._info_built = True


class ImageLablePairTfrecordCreator(DataLablePairTfrecordCreator):
    """ImageLablePairTfrecordCreator.

    Arguments:
        encode_type      : One of [None, 'png', 'jpg'].
        quality          : For 'jpg'.
        compression_type :
            0 : NONE
            1 : ZLIB
            2 : GZIP
    """

    def __init__(self,
                 save_path,
                 encode_type='png',
                 quality=95,
                 data_name='img',
                 size_each=None,
                 compression_type=0,
                 overwrite_existence=False):
        super(ImageLablePairTfrecordCreator, self).__init__(save_path,
                                                            None,
                                                            None,
                                                            data_name,
                                                            size_each,
                                                            compression_type,
                                                            overwrite_existence)

        assert encode_type in [None, 'png', 'jpg'], "`encode_type` should be in the list of [None, 'png', 'jpg']!"

        self._encode_type = encode_type
        self._quality = quality

        self._data_shape = None
        self._data_dtype_or_format = None
        self._is_data_bytes = True

    def add(self, image, label_dict):
        """Add example.

        `image`: An H * W (* C) uint8 numpy array.

        `label_dict` example:
            label_dict = {
                'id'    : id_ndarray,
                'attr'  : attr_ndarray,
                'point' : point_ndarray
            }
        """
        self._check(image)
        image_bytes = self._encode(image)
        super(ImageLablePairTfrecordCreator, self).add(image_bytes, label_dict)

    def _check(self, image):
        if not self._data_shape:
            assert isinstance(image, np.ndarray) and image.dtype == np.uint8 and image.ndim in [2, 3], \
                '`image` should be an H * W (* C) uint8 numpy array!'
            if self._encode_type and image.ndim == 3 and image.shape[-1] != 3:
                raise Exception('Only images with 1 or 3 channels are allowed to be encoded!')

            if image.ndim == 2:
                self._data_shape = image.shape + (1,)
            else:
                self._data_shape = image.shape
            self._data_dtype_or_format = [self._encode_type, 'uint8'][not self._encode_type]
        else:
            sp = image.shape
            if image.ndim == 2:
                sp = sp + (1,)
            assert sp == self._data_shape, 'Shapes of `image`s are inconsistent!'
            assert image.dtype == np.uint8, 'Dtypes of `image`s are inconsistent!'

    def _encode(self, image):
        if self._encode_type:
            if image.shape[-1] == 1:
                image.shape = image.shape[:2]
            byte = io.BytesIO()
            image = Image.fromarray(image)
            if self._encode_type == 'jpg':
                image.save(byte, 'JPEG', quality=self._quality)
            elif self._encode_type == 'png':
                image.save(byte, 'PNG')
            image_bytes = byte.getvalue()
        else:
            image_bytes = image.tobytes()
        return image_bytes
