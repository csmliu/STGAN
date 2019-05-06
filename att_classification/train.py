from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import models
import traceback
import numpy as np
import tflib as tl
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

""" param """
epoch_ = 25
batch_size = 64
lr = 0.0002
gpu_id = 0

''' data '''
train_tfrecord_path = 'D:\\Datasets\\CelebA\\tfrecord_128\\trainval'
test_tfrecord_path = 'D:\\Datasets\\CelebA\\tfrecord_128\\test'
train_data_pool = tl.TfrecordData(train_tfrecord_path, batch_size)
test_data_pool = tl.TfrecordData(test_tfrecord_path, 100)
att_dim = 40

def mean_accuracy_multi_binary_label_with_logits(att, logits):
#    return tf.count_nonzero(tf.equal(tf.greater(logits, 0.5), tf.greater(tf.to_float(att), 0.5)))
#    return tf.reduce_mean(tf.to_float(tf.equal(tf.to_int64(tf.round(logits)), att)))
    #return tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(att,1))))
    return tf.reduce_mean(tf.to_float(tf.equal(tf.to_int64(tf.greater(logits, 0.0)), att)))

""" graphs """
#with tf.device('/gpu:%d' % gpu_id):
''' models '''
classifier = models.classifier

''' graph '''
# inputs
x_255 = tf.placeholder(tf.float32, shape=[None, 128, 128, 3])
x = x_255 / 127.5 - 1
att = tf.placeholder(tf.int64, shape=[None, att_dim])

# classify
logits = classifier(x, reuse=False)

# loss
reg_loss = tf.losses.get_regularization_loss()
loss = tf.losses.sigmoid_cross_entropy(att, logits) + reg_loss
acc = mean_accuracy_multi_binary_label_with_logits(att, logits)

# summary
summary = tl.summary({loss: 'loss', acc: 'acc'})

lr_ = tf.placeholder(tf.float32, shape=[])

# optim
#with tf.variable_scope('Adam', reuse=tf.AUTO_REUSE):
step = tf.train.AdamOptimizer(lr_, beta1=0.9).minimize(loss)

# test
test_logits = classifier(x, training=False)
test_acc = mean_accuracy_multi_binary_label_with_logits(att, test_logits)
mean_acc = tf.placeholder(tf.float32, shape=())
test_summary = tl.summary({mean_acc: 'test_acc'})


""" train """
''' init '''
# session
sess = tf.Session()
# iteration counter
it_cnt, update_cnt = tl.counter()
# saver
saver = tf.train.Saver(max_to_keep=None)
# summary writer
summary_writer = tf.summary.FileWriter('./summaries', sess.graph)

''' initialization '''
ckpt_dir = './checkpoints'
if not os.path.exists(ckpt_dir):
    os.mkdir(ckpt_dir + '/')
if not tl.load_checkpoint(ckpt_dir, sess):
    sess.run(tf.global_variables_initializer())

''' train '''
try:
    batch_epoch = len(train_data_pool) // batch_size
    max_it = epoch_ * batch_epoch
    for it in range(sess.run(it_cnt), max_it):
        bth = it//batch_epoch - 8
        lr__ = lr*(1-max(bth, 0)/epoch_)**0.75
        if it % batch_epoch == 0:
            print('======learning rate:', lr__, '======')
        sess.run(update_cnt)

        # which epoch
        epoch = it // batch_epoch
        it_epoch = it % batch_epoch + 1

        x_255_ipt, att_ipt = train_data_pool.batch(['img', 'attr'])
        summary_opt, _ = sess.run([summary, step], feed_dict={x_255: x_255_ipt, att: att_ipt, lr_:lr__})
        summary_writer.add_summary(summary_opt, it)

        # display
        if it % 1 == 0:
            print("Epoch: (%3d) (%5d/%5d)" % (epoch, it_epoch, batch_epoch))

        # save
        if (it + 1) % 1000 == 0:
            save_path = saver.save(sess, '%s/Epoch_(%d)_(%dof%d).ckpt' % (ckpt_dir, epoch, it_epoch, batch_epoch))
            print('Model saved in file: % s' % save_path)

        # sample
        if it % 100 == 0:
            test_it = 100
            test_acc_opt_list = []
            for i in range(test_it):
                x_255_ipt, att_ipt = test_data_pool.batch(['img', 'attr'])
                
                test_acc_opt = sess.run(test_acc, feed_dict={x_255: x_255_ipt, att: att_ipt})
                print(test_acc_opt)
                test_acc_opt_list.append(test_acc_opt)
            test_summary_opt = sess.run(test_summary, feed_dict={mean_acc: np.mean(test_acc_opt_list)})
            summary_writer.add_summary(test_summary_opt, it)

except Exception:
    traceback.print_exc()
finally:
    print(" [*] Close main session!")
    sess.close()
