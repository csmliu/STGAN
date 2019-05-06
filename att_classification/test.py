from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import models
import traceback
import numpy as np
import imlib as im
import tflib as tl
import tensorflow as tf
import argparse

from glob import glob

img_size = 128

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', dest='gpu', type=str, default='all')
parser.add_argument('--img_dir', dest='img_dir', type=str)
args = parser.parse_args()

gpu = args.gpu
img_dir = args.img_dir

if gpu != 'all':
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

ad = att_dict = {'5_o_Clock_Shadow': 0, 'Arched_Eyebrows': 1, 'Attractive': 2, 'Bags_Under_Eyes': 3, 'Bald': 4, 'Bangs': 5, 'Big_Lips': 6, 'Big_Nose': 7, 'Black_Hair': 8, 'Blond_Hair': 9, 'Blurry': 10, 'Brown_Hair': 11, 'Bushy_Eyebrows': 12, 'Chubby': 13, 'Double_Chin': 14, 'Eyeglasses': 15, 'Goatee': 16, 'Gray_Hair': 17, 'Heavy_Makeup': 18, 'High_Cheekbones': 19, 'Male': 20, 'Mouth_Slightly_Open': 21, 'Mustache': 22, 'Narrow_Eyes': 23, 'No_Beard': 24, 'Oval_Face': 25, 'Pale_Skin': 26, 'Pointy_Nose': 27, 'Receding_Hairline': 28, 'Rosy_Cheeks': 29, 'Sideburns': 30, 'Smiling': 31, 'Straight_Hair': 32, 'Wavy_Hair': 33, 'Wearing_Earrings': 34, 'Wearing_Hat': 35, 'Wearing_Lipstick': 36, 'Wearing_Necklace': 37, 'Wearing_Necktie': 38, 'Young': 39}


""" param """
gpu_id = 2
att_id = np.array([ad['Bald'],  # ok
                   ad['Bangs'],  # ok
                   ad['Black_Hair'],  # ok
                   ad['Blond_Hair'],  # ok
                   ad['Brown_Hair'],  # ok
                   ad['Bushy_Eyebrows'],  # ok
                   ad['Eyeglasses'],  # ok
                   ad['Male'],  # ok
                   ad['Mouth_Slightly_Open'],  # ok
                   ad['Mustache'],  # ok
                   ad['No_Beard'],  # ok
                   ad['Pale_Skin'],  # ok
                   ad['Young']])  # ok

''' data '''
ckpt_file = './checkpoints/128.ckpt'
test_tfrecord_path = './tfrecords/test'
test_data_pool = tl.TfrecordData(test_tfrecord_path, 1, shuffle=False)


""" graphs """
# with tf.device('/gpu:%d' % gpu_id):
''' models '''
classifier = models.classifier

''' graph '''
# inputs
x = tf.placeholder(tf.float32, shape=[None, 128, 128, 3])

# classify
logits = classifier(x, reuse=False, training=False)
pred = tf.cast(tf.round(tf.nn.sigmoid(logits)), tf.int64)


""" train """
''' init '''
# session
sess = tl.session()

''' initialization '''
tl.load_checkpoint(ckpt_file, sess)

''' train '''
try:
    img_paths = glob(os.path.join(img_dir, '*.png'))
    img_paths.sort()

    cnt = np.zeros([len(att_id)])
    err_cnt = np.zeros([len(att_id)])
    err_each_cnt = np.zeros([len(att_id), len(att_id)])
    for img_path in img_paths:
        imgs = im.imread(img_path)
        imgs = np.concatenate([imgs[:, :img_size, :], imgs[:, img_size+img_size//10:, :]], axis=1)
        imgs = np.expand_dims(imgs, axis=0)
        imgs = np.concatenate(np.split(imgs, 15, axis=2))
        preds_opt = sess.run(pred, feed_dict={x: imgs})
        preds_opt = preds_opt[:, att_id]

        att_gt = test_data_pool.batch('attr')[:, att_id]

        for i in range(2, len(preds_opt)):
            cnt[i - 2] += preds_opt[i, i - 2] == 1 - att_gt[0, i - 2]
            errs = preds_opt[i] != att_gt[0]
            errs[i - 2] = 0
            err_each_cnt[i - 2] += errs
            err_cnt[i - 2] += np.sum(errs)

        print(os.path.basename(img_path))

    print('Acc.')
    print(cnt / len(img_paths))
    print('Err.')
    print(err_cnt / len(img_paths) / (len(att_id) - 1))
    print('Err. Each')
    print(err_each_cnt / len(img_paths))

except Exception:
    traceback.print_exc()
finally:
    print(" [*] Close main session!")
    sess.close()
