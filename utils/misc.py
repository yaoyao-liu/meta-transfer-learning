##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## NUS School of Computing
## Email: yaoyao.liu@mail.m2i.ac.cn
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

""" Utility functions. """
import numpy as np
import os
import cv2
import random
import tensorflow as tf
import scipy.misc as scm

from tensorflow.contrib.layers.python import layers as tf_layers
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

def get_smallest_k_index(input_, k):
    input_copy = np.copy(input_)
    k_list = []
    for idx in range(k):
        this_index = np.argmin(input_copy)
        k_list.append(this_index)
        input_copy[this_index]=np.max(input_copy)
    return k_list

def one_hot(inp):
    n_class = inp.max() + 1
    n_sample = inp.shape[0]
    out = np.zeros((n_sample, n_class))
    for idx in range(n_sample):
        out[idx, inp[idx]] = 1
    return out

def one_hot_class(inp, n_class):
    #n_class = inp.max() + 1
    n_sample = inp.shape[0]
    out = np.zeros((n_sample, n_class))
    for idx in range(n_sample):
        out[idx, inp[idx]] = 1
    return out

def process_batch(input_filename_list, input_label_list, dim_input, batch_sample_num):
    new_path_list = []
    new_label_list = []
    for k in range(batch_sample_num):
        class_idxs = range(0, FLAGS.way_num)
        random.shuffle(class_idxs)
        for class_idx in class_idxs:
            true_idx = class_idx*batch_sample_num + k
            new_path_list.append(input_filename_list[true_idx])
            new_label_list.append(input_label_list[true_idx])

    img_list = []
    for filepath in new_path_list:
        this_img = scm.imread(filepath)
        this_img = np.reshape(this_img, [-1, dim_input])
        this_img = this_img / 255.0
        img_list.append(this_img)

    img_array = np.array(img_list).reshape([FLAGS.way_num*batch_sample_num, dim_input])
    label_array = one_hot(np.array(new_label_list)).reshape([FLAGS.way_num*batch_sample_num, -1])
    return img_array, label_array

def process_batch_augmentation(input_filename_list, input_label_list, dim_input, batch_sample_num):
    new_path_list = []
    new_label_list = []
    for k in range(batch_sample_num):
        class_idxs = range(0, FLAGS.way_num)
        random.shuffle(class_idxs)
        for class_idx in class_idxs:
            true_idx = class_idx*batch_sample_num + k
            new_path_list.append(input_filename_list[true_idx])
            new_label_list.append(input_label_list[true_idx])

    img_list = []
    img_list_h = []
    for filepath in new_path_list:
        this_img = scm.imread(filepath)
        this_img_h = cv2.flip(this_img, 1)
        this_img = np.reshape(this_img, [-1, dim_input])
        this_img = this_img / 255.0
        img_list.append(this_img)
        this_img_h = np.reshape(this_img_h, [-1, dim_input])
        this_img_h = this_img_h / 255.0
        img_list_h.append(this_img_h)

    img_list_all = img_list + img_list_h
    label_list_all = new_label_list + new_label_list

    img_array = np.array(img_list_all).reshape([FLAGS.way_num*batch_sample_num*2, dim_input])
    label_array = one_hot(np.array(label_list_all)).reshape([FLAGS.way_num*batch_sample_num*2, -1])
    return img_array, label_array


def get_images(paths, labels, nb_samples=None, shuffle=True):
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    images = [(i, os.path.join(path, image)) \
        for i, path in zip(labels, paths) \
        for image in sampler(os.listdir(path))]
    if shuffle:
        random.shuffle(images)
    return images

def get_pretrain_images(path, label, is_val=False):
    images = []
    if is_val==False:
        for image in os.listdir(path):
            images.append((label, os.path.join(path, image)))
    else:
        for image in os.listdir(path)[550:]:
            images.append((label, os.path.join(path, image)))    
    return images

def get_images_tc(paths, labels, nb_samples=None, shuffle=True, is_val=False):
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    if is_val==False:
        images = [(i, os.path.join(path, image)) \
            for i, path in zip(labels, paths) \
            for image in sampler(os.listdir(path)[0:500])]
    else:
        images = [(i, os.path.join(path, image)) \
            for i, path in zip(labels, paths) \
            for image in sampler(os.listdir(path)[500:])]
    if shuffle:
        random.shuffle(images)
    return images


## Network helpers

def leaky_relu(x, leak=0.1):
    return tf.maximum(x, leak*x)

def resnet_conv_block(inp, cweight, bweight, reuse, scope, activation=leaky_relu):
    """ Perform, conv, batch norm, nonlinearity, and max pool """
    stride, no_stride = [1,2,2,1], [1,1,1,1]

    if FLAGS.activation == 'leaky_relu':
        activation = leaky_relu
    elif FLAGS.activation == 'relu':
        activation = tf.nn.relu
    else:
        activation = None

    conv_output = tf.nn.conv2d(inp, cweight, no_stride, 'SAME') + bweight
    normed = normalize(conv_output, activation, reuse, scope)

    return normed

def resnet_nob_conv_block(inp, cweight, reuse, scope):
    """ Perform, conv, batch norm, nonlinearity, and max pool """
    stride, no_stride = [1,2,2,1], [1,1,1,1]
    conv_output = tf.nn.conv2d(inp, cweight, no_stride, 'SAME')
    return conv_output

def normalize(inp, activation, reuse, scope):
    if FLAGS.norm == 'batch_norm':
        return tf_layers.batch_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
    elif FLAGS.norm == 'layer_norm':
        return tf_layers.layer_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
    elif FLAGS.norm == 'None':
        if activation is not None:
            return activation(inp)
        else:
            return inp

## Loss functions

def mse(pred, label):
    pred = tf.reshape(pred, [-1])
    label = tf.reshape(label, [-1])
    return tf.reduce_mean(tf.square(pred-label))

def softmaxloss(pred, label):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label))

def xent(pred, label):
    # Note - with tf version <=0.12, this loss has incorrect 2nd derivatives
    return tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label) / FLAGS.shot_num
