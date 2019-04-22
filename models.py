from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tflib as tl


conv = partial(slim.conv2d, activation_fn=None)
dconv = partial(slim.conv2d_transpose, activation_fn=None)
fc = partial(tl.flatten_fully_connected, activation_fn=None)
relu = tf.nn.relu
lrelu = tf.nn.leaky_relu
sigmoid = tf.nn.sigmoid
tanh = tf.nn.tanh
batch_norm = partial(slim.batch_norm, scale=True, updates_collections=None)
instance_norm = slim.instance_norm

MAX_DIM = 64 * 16


def Genc(x, dim=64, n_layers=5, multi_inputs=1, is_training=True):
    bn = partial(batch_norm, is_training=is_training)
    conv_bn_lrelu = partial(conv, normalizer_fn=bn, activation_fn=lrelu)

    with tf.variable_scope('Genc', reuse=tf.AUTO_REUSE):
        h, w = x.shape[1:3]
        z = x
        zs = []
        for i in range(n_layers):
            d = min(dim * 2**i, MAX_DIM)
            if multi_inputs > i and i > 0:
                z = tf.concat([z, tf.image.resize_bicubic(x, (h//(2**i), w//(2**i)))], 3)
            z = conv_bn_lrelu(z, d, 4, 2)
            zs.append(z)
        return zs

def ConvGRUCell(in_data, state, out_channel, is_training=True, kernel_size=3, norm='none', pass_state='lstate'):
    if norm == 'bn':
        norm_fn = partial(batch_norm, is_training=is_training)
    elif norm == 'in':
        norm_fn = instance_norm
    else:
        norm_fn = None
    gate = partial(conv, normalizer_fn=norm_fn, activation_fn=sigmoid)
    info = partial(conv, normalizer_fn=norm_fn, activation_fn=tanh)
    with tf.name_scope('ConvGRUCell'):
        state_ = dconv(state, out_channel, 4, 2)  # upsample and make `channel` identical to `out_channel`
        reset_gate = gate(tf.concat([in_data, state_], axis=3), out_channel, kernel_size)
        update_gate = gate(tf.concat([in_data, state_], axis=3), out_channel, kernel_size)
        new_state = reset_gate * state_
        new_info = info(tf.concat([in_data, new_state], axis=3), out_channel, kernel_size)
        output = (1-update_gate)*state_ + update_gate*new_info
        if pass_state == 'gru':
            return output, output
        elif pass_state == 'direct':
            return output, state_
        else: # 'stu'
            return output, new_state

def Gstu(zs, _a, dim=64, n_layers=1, inject_layers=0, is_training=True, kernel_size=3, norm='none', pass_state='stu'):
    def _concat(z, z_, _a):
        feats = [z]
        if z_ is not None:
            feats.append(z_)
        if _a is not None:
            _a = tf.reshape(_a, [-1, 1, 1, tl.shape(_a)[-1]])
            _a = tf.tile(_a, [1, tl.shape(z)[1], tl.shape(z)[2], 1])
            feats.append(_a)
        return tf.concat(feats, axis=3)
    
    with tf.variable_scope('Gstu', reuse=tf.AUTO_REUSE):
        zs_ = [zs[-1]]
        state = _concat(zs[-1], None, _a)
        for i in range(n_layers): # n_layers <= 4
            d = min(dim * 2**(n_layers - 1 - i), MAX_DIM)
            output = ConvGRUCell(zs[n_layers - 1 - i], state, d, is_training=is_training,
                                 kernel_size=kernel_size, norm=norm, pass_state=pass_state)
            zs_.insert(0, output[0])
            if inject_layers > i:
                state = _concat(output[1], None, _a)
            else:
                state = output[1]
        return zs_

def Gdec(zs, _a, dim=64, n_layers=5, shortcut_layers=1, inject_layers=0, is_training=True, one_more_conv=0):
    bn = partial(batch_norm, is_training=is_training)
    dconv_bn_relu = partial(dconv, normalizer_fn=bn, activation_fn=relu)

    shortcut_layers = min(shortcut_layers, n_layers - 1)
    inject_layers = min(inject_layers, n_layers - 1)

    def _concat(z, z_, _a):
        feats = [z]
        if z_ is not None:
            feats.append(z_)
        if _a is not None:
            _a = tf.reshape(_a, [-1, 1, 1, tl.shape(_a)[-1]])
            _a = tf.tile(_a, [1, tl.shape(z)[1], tl.shape(z)[2], 1])
            feats.append(_a)
        return tf.concat(feats, axis=3)

    with tf.variable_scope('Gdec', reuse=tf.AUTO_REUSE):
        z = _concat(zs[-1], None, _a)
        for i in range(n_layers):
            if i < n_layers - 1:
                d = min(dim * 2**(n_layers - 1 - i), MAX_DIM)
                z = dconv_bn_relu(z, d, 4, 2)
                if shortcut_layers > i:
                    z = _concat(z, zs[n_layers - 2 - i], None)
                if inject_layers > i:
                    z = _concat(z, None, _a)
            else:
                if one_more_conv: # add one more conv after the decoder
                    z = dconv_bn_relu(z, dim//4, 4, 2)
                    x = tf.nn.tanh(dconv(z, 3, one_more_conv))
                else:
                    x = z = tf.nn.tanh(dconv(z, 3, 4, 2))
        return x


def D(x, n_att, dim=64, fc_dim=MAX_DIM, n_layers=5):
    conv_in_lrelu = partial(conv, normalizer_fn=instance_norm, activation_fn=lrelu)

    with tf.variable_scope('D', reuse=tf.AUTO_REUSE):
        y = x
        for i in range(n_layers):
            d = min(dim * 2**i, MAX_DIM)
            y = conv_in_lrelu(y, d, 4, 2)

        logit_gan = lrelu(fc(y, fc_dim))
        logit_gan = fc(logit_gan, 1)

        logit_att = lrelu(fc(y, fc_dim))
        logit_att = fc(logit_att, n_att)

        return logit_gan, logit_att


def gradient_penalty(f, real, fake=None):
    def _interpolate(a, b=None):
        with tf.name_scope('interpolate'):
            if b is None:   # interpolation in DRAGAN
                beta = tf.random_uniform(shape=tf.shape(a), minval=0., maxval=1.)
                _, variance = tf.nn.moments(a, range(a.shape.ndims))
                b = a + 0.5 * tf.sqrt(variance) * beta
            shape = [tf.shape(a)[0]] + [1] * (a.shape.ndims - 1)
            alpha = tf.random_uniform(shape=shape, minval=0., maxval=1.)
            inter = a + alpha * (b - a)
            inter.set_shape(a.get_shape().as_list())
            return inter

    with tf.name_scope('gradient_penalty'):
        x = _interpolate(real, fake)
        pred = f(x)
        if isinstance(pred, tuple):
            pred = pred[0]
        grad = tf.gradients(pred, x)[0]
        norm = tf.norm(slim.flatten(grad), axis=1)
        gp = tf.reduce_mean((norm - 1.)**2)
        return gp
