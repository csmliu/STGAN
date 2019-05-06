from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing

import tensorflow as tf

from tflib.data.memory_data import MemoryData


_N_CPU = multiprocessing.cpu_count()


class Mnist(MemoryData):

    def __init__(self,
                 batch_size,
                 split='train',
                 prefetch_batch=_N_CPU + 1,
                 drop_remainder=True,
                 filter=None,
                 map_func=None,
                 num_threads=_N_CPU,
                 shuffle=True,
                 buffer_size=None,
                 repeat=-1,
                 sess=None):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
        if split == 'train':
            imgs, lbls = x_train, y_train
        elif split == 'test':
            imgs, lbls = x_test, y_test
        else:
            raise ValueError("`split` must be 'test' or 'train'!")
        imgs = imgs / 127.5 - 1

        imgs.shape = imgs.shape + (1,)

        imgs_pl = tf.placeholder(tf.float32, imgs.shape)
        lbls_pl = tf.placeholder(tf.int64, lbls.shape)

        memory_data_dict = {'img': imgs_pl, 'lbl': lbls_pl}

        self.feed_dict = {imgs_pl: imgs, lbls_pl: lbls}
        super(Mnist, self).__init__(memory_data_dict,
                                    batch_size,
                                    prefetch_batch,
                                    drop_remainder,
                                    filter,
                                    map_func,
                                    num_threads,
                                    shuffle,
                                    buffer_size,
                                    repeat,
                                    sess)

    def reset(self):
        super(Mnist, self).reset(self.feed_dict)

if __name__ == '__main__':
    import imlib as im
    from tflib import session
    sess = session()
    mnist = Mnist(500, repeat=2, sess=sess, shuffle=True, split='train')
    print(len(mnist))
    for batch in mnist:
        print(batch['lbl'][-1])
        im.imshow(batch['img'][-1].squeeze())
        im.show()
    sess.close()
