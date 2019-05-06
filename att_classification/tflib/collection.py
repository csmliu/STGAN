from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

import tensorflow as tf


def tensors_filter(tensors,
                   includes='',
                   includes_combine_type='or',
                   excludes=[],
                   excludes_combine_type='or'):
    assert isinstance(tensors, (list, tuple)), '`tensors` shoule be a list or tuple!'
    assert isinstance(includes, (str, list, tuple)), '`includes` should be a string or a list(tuple) of strings!'
    assert includes_combine_type in ['or', 'and'], "`includes_combine_type` should be 'or' or 'and'!"
    assert isinstance(excludes, (str, list, tuple)), '`excludes` should be a string or a list(tuple) of strings!'
    assert excludes_combine_type in ['or', 'and'], "`excludes_combine_type` should be 'or' or 'and'!"

    def _select(filters, combine_type):
        if isinstance(filters, str):
            filters = [filters]

        selected = []
        for t in tensors:
            if combine_type == 'or':
                for filt in filters:
                    if filt in t.name:
                        selected.append(t)
                        break
            elif combine_type == 'and':
                all_pass = True and filters  # for fiters == []
                for filt in filters:
                    if filt not in t.name:
                        all_pass = False
                        break
                if all_pass:
                    selected.append(t)

        return selected

    include_set = _select(includes, includes_combine_type)
    exclude_set = _select(excludes, excludes_combine_type)
    select_set = [t for t in include_set if t not in exclude_set]

    return select_set


def get_collection(key,
                   includes='',
                   includes_combine_type='or',
                   excludes=[],
                   excludes_combine_type='and'):
    tensors = tf.get_collection(key)
    return tensors_filter(tensors,
                          includes,
                          includes_combine_type,
                          excludes,
                          excludes_combine_type)

global_variables = partial(get_collection, key=tf.GraphKeys.GLOBAL_VARIABLES)
trainable_variables = partial(get_collection, key=tf.GraphKeys.TRAINABLE_VARIABLES)
update_ops = partial(get_collection, key=tf.GraphKeys.UPDATE_OPS)
