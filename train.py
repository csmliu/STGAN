from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime
from functools import partial
import json
import traceback


import imlib as im
import numpy as np
import pylib
import tensorflow as tf
import tflib as tl

import data
import models

import os

# ==============================================================================
# =                                    param                                   =
# ==============================================================================
def boolean(s):
    return s.lower() in ('true', 't', 'yes', 'y', '1')

parser = argparse.ArgumentParser()
# settings
dataroot_default = '/data/Datasets/CelebA/Img'
parser.add_argument('--dataroot', type=str, default=dataroot_default)
parser.add_argument('--gpu', type=str, default='all',
                    help='Specify which gpu to use by `CUDA_VISIBLE_DEVICES=num python train.py **kwargs`\
                          or `python train.py --gpu num` if you\'re running on a multi-gpu enviroment.\
                          You need to do nothing if your\'re running on a single-gpu environment or\
                          the gpu is assigned by a resource manager program.')
parser.add_argument('--threads', type=int, default=-1,
                    help='Control parallel computation threads,\
                          please leave it as is if no heavy cpu burden is observed.')
# model
att_default = ['Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows', 'Eyeglasses',
               'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young']
parser.add_argument('--atts', default=att_default, choices=data.Celeba.att_dict.keys(), nargs='+',
                    help='Attributes to modify by the model')
parser.add_argument('--img_size', type=int, default=128, help='input image size')
parser.add_argument('--shortcut_layers', type=int, default=4,
                    help='# of skip connections between the encoder and the decoder')
parser.add_argument('--inject_layers', type=int, default=4,
                    help='# of attribute vectors applied in the decoder')
parser.add_argument('--enc_dim', type=int, default=64)
parser.add_argument('--dec_dim', type=int, default=64)
parser.add_argument('--dis_dim', type=int, default=64)
parser.add_argument('--dis_fc_dim', type=int, default=1024,
                    help='# of discriminator fc channels')
parser.add_argument('--enc_layers', type=int, default=5)
parser.add_argument('--dec_layers', type=int, default=5)
parser.add_argument('--dis_layers', type=int, default=5)
# STGAN & STU
parser.add_argument('--label', type=str, default='diff', choices=['diff', 'target'])
parser.add_argument('--use_stu', type=boolean, default=True)
parser.add_argument('--stu_dim', type=int, default=64)
parser.add_argument('--stu_layers', type=int, default=4)
parser.add_argument('--stu_inject_layers', type=int, default=4)
parser.add_argument('--stu_kernel_size', type=int, default=3)
parser.add_argument('--stu_norm', type=str, default='none', choices=['none', 'bn', 'in'])
parser.add_argument('--stu_state', type=str, default='stu', choices=['stu', 'gru', 'direct'],
                    help='gru: gru arch.; stu: stu arch.; direct: directly pass the inner state to the outer layer')
parser.add_argument('--multi_inputs', type=int, default=1,
                    help='# of hierachical inputs (in the first several encoder layers')
parser.add_argument('--rec_loss_weight', type=float, default=100.0)
parser.add_argument('--one_more_conv', type=int, default=0, choices=[0, 1, 3],
                    help='0: no further conv after the decoder; 1: conv(k=1); 3: conv(k=3)')
# training
parser.add_argument('--mode', default='wgan', choices=['wgan', 'lsgan', 'dcgan'])
parser.add_argument('--epoch', type=int, default=200, help='# of epochs')
parser.add_argument('--init_epoch', type=int, default=100, help='# of epochs with init lr.')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--n_d', type=int, default=5, help='# of d updates per g update')
parser.add_argument('--thres_int', type=float, default=0.5)
parser.add_argument('--test_int', type=float, default=1.0)
parser.add_argument('--n_sample', type=int, default=64, help='# of sample images')
parser.add_argument('--save_freq', type=int, default=0,
                    help='save model evary save_freq iters, 0 means to save evary epoch.')
parser.add_argument('--sample_freq', type=int, default=0,
                    help='eval on validation set every sample_freq iters, 0 means to save evary epoch.')
# others
parser.add_argument('--use_cropped_img', action='store_true')
parser.add_argument('--experiment_name', default=datetime.datetime.now().strftime("%Y.%m.%d-%H%M%S"))
parser.add_argument('--num_ckpt', type=int, default=1)
parser.add_argument('--clear', default=False, action='store_true')

args = parser.parse_args()
# settings
dataroot = args.dataroot
if args.gpu != 'all':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
threads = args.threads
# model
atts = args.atts
n_att = len(atts)
img_size = args.img_size
shortcut_layers = args.shortcut_layers
inject_layers = args.inject_layers
enc_dim = args.enc_dim
dec_dim = args.dec_dim
dis_dim = args.dis_dim
dis_fc_dim = args.dis_fc_dim
enc_layers = args.enc_layers
dec_layers = args.dec_layers
dis_layers = args.dis_layers
# STU
label = args.label
use_stu = args.use_stu
stu_dim = args.stu_dim
stu_layers = args.stu_layers
stu_inject_layers = args.stu_inject_layers
stu_kernel_size = args.stu_kernel_size
stu_norm = args.stu_norm
stu_state = args.stu_state
multi_inputs = args.multi_inputs
rec_loss_weight = float(args.rec_loss_weight)
one_more_conv = args.one_more_conv
# training
mode = args.mode
epoch = args.epoch
init_epoch = args.init_epoch
batch_size = args.batch_size
lr_base = args.lr
n_d = args.n_d
thres_int = args.thres_int
test_int = args.test_int
n_sample = args.n_sample
save_freq = args.save_freq
sample_freq = args.sample_freq
# others
use_cropped_img = args.use_cropped_img
experiment_name = args.experiment_name
num_ckpt = args.num_ckpt
clear = args.clear

pylib.mkdir('./output/%s' % experiment_name)
with open('./output/%s/setting.txt' % experiment_name, 'w') as f:
    f.write(json.dumps(vars(args), indent=4, separators=(',', ':')))

# ==============================================================================
# =                                   graphs                                   =
# ==============================================================================

# data
if threads >= 0:
    cpu_config = tf.ConfigProto(intra_op_parallelism_threads = threads//2,
                                inter_op_parallelism_threads = threads//2,
                                device_count = {'CPU': threads})
    sess = tf.Session(config=cpu_config)
else:
    sess = tl.session()
crop_ = not use_cropped_img
tr_data = data.Celeba(dataroot, atts, img_size, batch_size, part='train', sess=sess, crop=crop_)
val_data = data.Celeba(dataroot, atts, img_size, n_sample, part='val', shuffle=False, sess=sess, crop=crop_)

# models
Genc = partial(models.Genc, dim=enc_dim, n_layers=enc_layers, multi_inputs=multi_inputs)
Gdec = partial(models.Gdec, dim=dec_dim, n_layers=dec_layers, shortcut_layers=shortcut_layers,
               inject_layers=inject_layers, one_more_conv=one_more_conv)
Gstu = partial(models.Gstu, dim=stu_dim, n_layers=stu_layers, inject_layers=stu_inject_layers,
               kernel_size=stu_kernel_size, norm=stu_norm, pass_state=stu_state)
D = partial(models.D, n_att=n_att, dim=dis_dim, fc_dim=dis_fc_dim, n_layers=dis_layers)

# inputs
lr = tf.placeholder(dtype=tf.float32, shape=[])

xa = tr_data.batch_op[0]
a = tr_data.batch_op[1]
b = tf.random_shuffle(a)
_a = (tf.to_float(a) * 2 - 1) * thres_int
_b = (tf.to_float(b) * 2 - 1) * thres_int

xa_sample = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 3])
_b_sample = tf.placeholder(tf.float32, shape=[None, n_att])
raw_b_sample = tf.placeholder(tf.float32, shape=[None, n_att])

# generate
z = Genc(xa)
zb = Gstu(z, _b-_a if label=='diff' else _b) if use_stu else z
xb_ = Gdec(zb, _b-_a if label=='diff' else _b)
with tf.control_dependencies([xb_]):
    za = Gstu(z, _a-_a if label=='diff' else _a) if use_stu else z
    xa_ = Gdec(za, _a-_a if label=='diff' else _a)

# discriminate
xa_logit_gan, xa_logit_att = D(xa)
xb__logit_gan, xb__logit_att = D(xb_)

# discriminator losses
if mode == 'wgan':  # wgan-gp
    wd = tf.reduce_mean(xa_logit_gan) - tf.reduce_mean(xb__logit_gan)
    d_loss_gan = -wd
    gp = models.gradient_penalty(D, xa, xb_)
elif mode == 'lsgan':  # lsgan-gp
    xa_gan_loss = tf.losses.mean_squared_error(tf.ones_like(xa_logit_gan), xa_logit_gan)
    xb__gan_loss = tf.losses.mean_squared_error(tf.zeros_like(xb__logit_gan), xb__logit_gan)
    d_loss_gan = xa_gan_loss + xb__gan_loss
    gp = models.gradient_penalty(D, xa)
elif mode == 'dcgan':  # dcgan-gp
    xa_gan_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(xa_logit_gan), xa_logit_gan)
    xb__gan_loss = tf.losses.sigmoid_cross_entropy(tf.zeros_like(xb__logit_gan), xb__logit_gan)
    d_loss_gan = xa_gan_loss + xb__gan_loss
    gp = models.gradient_penalty(D, xa)

xa_loss_att = tf.losses.sigmoid_cross_entropy(a, xa_logit_att)

d_loss = d_loss_gan + gp * 10.0 + xa_loss_att

# generator losses
if mode == 'wgan':
    xb__loss_gan = -tf.reduce_mean(xb__logit_gan)
elif mode == 'lsgan':
    xb__loss_gan = tf.losses.mean_squared_error(tf.ones_like(xb__logit_gan), xb__logit_gan)
elif mode == 'dcgan':
    xb__loss_gan = tf.losses.sigmoid_cross_entropy(tf.ones_like(xb__logit_gan), xb__logit_gan)

xb__loss_att = tf.losses.sigmoid_cross_entropy(b, xb__logit_att)
xa__loss_rec = tf.losses.absolute_difference(xa, xa_)

g_loss = xb__loss_gan + xb__loss_att * 10.0 + xa__loss_rec * rec_loss_weight

# optim
d_var = tl.trainable_variables('D')
d_step = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(d_loss, var_list=d_var)

g_var = tl.trainable_variables('G')
g_step = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(g_loss, var_list=g_var)

# summary
show_weights = False

d_summary = tl.summary({
    d_loss_gan: 'd_loss_gan',
    gp: 'gp',
    xa_loss_att: 'xa_loss_att',
}, scope='D')

lr_summary = tl.summary({lr: 'lr'}, scope='Learning_Rate')


g_summary = tl.summary({
    xb__loss_gan: 'xb__loss_gan',
    xb__loss_att: 'xb__loss_att',
    xa__loss_rec: 'xa__loss_rec',
}, scope='G')
if show_weights:
    d_histogram = tf.summary.merge([tf.summary.histogram(
        name=i.name,
        values=i
    ) for i in tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, 'D')])
    
    genc_histogram = tf.summary.merge([tf.summary.histogram(
        name=i.name,
        values=i
    ) for i in tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, 'Genc')])
        
    gdec_histogram = tf.summary.merge([tf.summary.histogram(
        name=i.name,
        values=i
    ) for i in tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, 'Gdec')])
    
    gstu_histogram = tf.summary.merge([tf.summary.histogram(
        name=i.name,
        values=i
    ) for i in tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, 'Gstu')])
    
    d_summary = tf.summary.merge([d_summary, lr_summary, d_histogram])
    g_summary = tf.summary.merge([g_summary, genc_histogram, gdec_histogram, gstu_histogram])
else:
    d_summary = tf.summary.merge([d_summary, lr_summary])

# sample
test_label = _b_sample - raw_b_sample if label=='diff' else _b_sample
if use_stu:
    x_sample = Gdec(Gstu(Genc(xa_sample, is_training=False),
                         test_label, is_training=False), test_label, is_training=False)
else:
    x_sample = Gdec(Genc(xa_sample, is_training=False), test_label, is_training=False)

# ==============================================================================
# =                                    train                                   =
# ==============================================================================

# iteration counter
it_cnt, update_cnt = tl.counter()

# saver
saver = tf.train.Saver(max_to_keep=num_ckpt)

# summary writer
summary_writer = tf.summary.FileWriter('./output/%s/summaries' % experiment_name, sess.graph)

# initialization
ckpt_dir = './output/%s/checkpoints' % experiment_name
pylib.mkdir(ckpt_dir)

try:
    assert clear == False
    tl.load_checkpoint(ckpt_dir, sess)
except:
    print('NOTE: Initializing all parameters...')
    sess.run(tf.global_variables_initializer())

# train
try:
    # data for sampling
    xa_sample_ipt, a_sample_ipt = val_data.get_next()
    b_sample_ipt_list = [a_sample_ipt]  # the first is for reconstruction
    for i in range(len(atts)):
        tmp = np.array(a_sample_ipt, copy=True)
        tmp[:, i] = 1 - tmp[:, i]   # inverse attribute
        tmp = data.Celeba.check_attribute_conflict(tmp, atts[i], atts)
        b_sample_ipt_list.append(tmp)

    it_per_epoch = len(tr_data) // (batch_size * (n_d + 1))
    max_it = epoch * it_per_epoch
    for it in range(sess.run(it_cnt), max_it):
        with pylib.Timer(is_output=False) as t:
            sess.run(update_cnt)

            # which epoch
            epoch = it // it_per_epoch
            it_in_epoch = it % it_per_epoch + 1

            # learning rate
            lr_ipt = lr_base / (10 ** (epoch // init_epoch))

            # train D
            for i in range(n_d):
                d_summary_opt, _ = sess.run([d_summary, d_step], feed_dict={lr: lr_ipt})
            summary_writer.add_summary(d_summary_opt, it)

            # train G
            g_summary_opt, _ = sess.run([g_summary, g_step], feed_dict={lr: lr_ipt})
            summary_writer.add_summary(g_summary_opt, it)

            # display
            if (it + 1) % 1 == 0:
                print("Epoch: (%3d) (%5d/%5d) Time: %s!" % (epoch, it_in_epoch, it_per_epoch, t))

            # save
            if (it + 1) % (save_freq if save_freq else it_per_epoch) == 0:
                save_path = saver.save(sess, '%s/Epoch_(%d)_(%dof%d).ckpt'%(ckpt_dir, epoch, it_in_epoch, it_per_epoch))
                print('Model is saved at %s!' % save_path)

            # sample
            if (it + 1) % (sample_freq if sample_freq else it_per_epoch) == 0:
                x_sample_opt_list = [xa_sample_ipt, np.full((n_sample, img_size, img_size // 10, 3), -1.0)]
                raw_b_sample_ipt = (b_sample_ipt_list[0].copy() * 2 - 1) * thres_int
                for i, b_sample_ipt in enumerate(b_sample_ipt_list):
                    _b_sample_ipt = (b_sample_ipt * 2 - 1) * thres_int
                    if i > 0:   # i == 0 is for reconstruction
                        _b_sample_ipt[..., i - 1] = _b_sample_ipt[..., i - 1] * test_int / thres_int
                    x_sample_opt_list.append(sess.run(x_sample, feed_dict={xa_sample: xa_sample_ipt,
                                                                           _b_sample: _b_sample_ipt,
                                                                           raw_b_sample: raw_b_sample_ipt}))
                    last_images = x_sample_opt_list[-1]
                    if i > 0:   # add a mark (+/-) in the upper-left corner to identify add/remove an attribute
                        for nnn in range(last_images.shape[0]):
                            last_images[nnn, 2:5, 0:7, :] = 1.
                            if _b_sample_ipt[nnn, i-1] > 0:
                                last_images[nnn, 0:7, 2:5, :] = 1.
                                last_images[nnn, 1:6, 3:4, :] = -1.
                            last_images[nnn, 3:4, 1:6, :] = -1.
                sample = np.concatenate(x_sample_opt_list, 2)

                save_dir = './output/%s/sample_training' % experiment_name
                pylib.mkdir(save_dir)
                im.imwrite(im.immerge(sample, n_sample, 1), '%s/Epoch_(%d)_(%dof%d).jpg' % \
                                                            (save_dir, epoch, it_in_epoch, it_per_epoch))
except:
    traceback.print_exc()
finally:
    save_path = saver.save(sess, '%s/Epoch_(%d)_(%dof%d).ckpt' % (ckpt_dir, epoch, it_in_epoch, it_per_epoch))
    print('Model is saved at %s!' % save_path)
    sess.close()
