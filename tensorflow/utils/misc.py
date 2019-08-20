##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## Modified from: https://github.com/cbfinn/maml
## Tianjin University
## liuyaoyao@tju.edu.cn
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

""" Additional utility functions. """
import numpy as np
import os
import cv2
import random
import tensorflow as tf

from matplotlib.pyplot import imread
from tensorflow.contrib.layers.python import layers as tf_layers
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

def get_smallest_k_index(input_, k):
    """The function to get the smallest k items' indices.
    Args:
      input_: the list to be processed.
      k: the number of indices to return.
    Return:
      The index list with k dimensions.
    """
    input_copy = np.copy(input_)
    k_list = []
    for idx in range(k):
        this_index = np.argmin(input_copy)
        k_list.append(this_index)
        input_copy[this_index]=np.max(input_copy)
    return k_list

def one_hot(inp):
    """The function to make the input to one-hot vectors.
    Arg:
      inp: the input numpy array.
    Return:
      The reorganized one-shot array.
    """
    n_class = inp.max() + 1
    n_sample = inp.shape[0]
    out = np.zeros((n_sample, n_class))
    for idx in range(n_sample):
        out[idx, inp[idx]] = 1
    return out

def one_hot_class(inp, n_class):
    """The function to make the input to n-class one-hot vectors.
    Args:
      inp: the input numpy array.
      n_class: the number of classes.
    Return:
      The reorganized n-class one-shot array.
    """
    n_sample = inp.shape[0]
    out = np.zeros((n_sample, n_class))
    for idx in range(n_sample):
        out[idx, inp[idx]] = 1
    return out

def process_batch(input_filename_list, input_label_list, dim_input, batch_sample_num):
    """The function to process a part of an episode.
    Args:
      input_filename_list: the image files' directory list.
      input_label_list: the image files' corressponding label list.
      dim_input: the dimension number of the images.
      batch_sample_num: the sample number of the inputed images.
    Returns:
      img_array: the numpy array of processed images.
      label_array: the numpy array of processed labels.
    """
    new_path_list = []
    new_label_list = []
    for k in range(batch_sample_num):
        class_idxs = list(range(0, FLAGS.way_num))
        random.shuffle(class_idxs)
        for class_idx in class_idxs:
            true_idx = class_idx*batch_sample_num + k
            new_path_list.append(input_filename_list[true_idx])
            new_label_list.append(input_label_list[true_idx])

    img_list = []
    for filepath in new_path_list:
        this_img = imread(filepath)
        this_img = np.reshape(this_img, [-1, dim_input])
        this_img = this_img / 255.0
        img_list.append(this_img)

    img_array = np.array(img_list).reshape([FLAGS.way_num*batch_sample_num, dim_input])
    label_array = one_hot(np.array(new_label_list)).reshape([FLAGS.way_num*batch_sample_num, -1])
    return img_array, label_array

def process_batch_augmentation(input_filename_list, input_label_list, dim_input, batch_sample_num):
    """The function to process a part of an episode. All the images will be augmented by flipping.
    Args:
      input_filename_list: the image files' directory list.
      input_label_list: the image files' corressponding label list.
      dim_input: the dimension number of the images.
      batch_sample_num: the sample number of the inputed images.
    Returns:
      img_array: the numpy array of processed images.
      label_array: the numpy array of processed labels.
    """
    new_path_list = []
    new_label_list = []
    for k in range(batch_sample_num):
        class_idxs = list(range(0, FLAGS.way_num))
        random.shuffle(class_idxs)
        for class_idx in class_idxs:
            true_idx = class_idx*batch_sample_num + k
            new_path_list.append(input_filename_list[true_idx])
            new_label_list.append(input_label_list[true_idx])

    img_list = []
    img_list_h = []
    for filepath in new_path_list:
        this_img = imread(filepath)
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
    """The function to get the image files' directories with given class labels.
    Args:
      paths: the base path for the images.
      labels: the class name labels.
      nb_samples: the number of samples.
      shuffle: whether shuffle the generated image list.
    Return:
      The list for the image files' directories.
    """
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

def get_pretrain_images(path, label):
    """The function to get the image files' directories for pre-train phase.
    Args:
      paths: the base path for the images.
      labels: the class name labels.
      is_val: whether the images are for the validation phase during pre-training.
    Return:
      The list for the image files' directories.
    """
    images = []
    for image in os.listdir(path):
        images.append((label, os.path.join(path, image)))
    return images

def get_images_tc(paths, labels, nb_samples=None, shuffle=True, is_val=False):
    """The function to get the image files' directories with given class labels for pre-train phase.
    Args:
      paths: the base path for the images.
      labels: the class name labels.
      nb_samples: the number of samples.
      shuffle: whether shuffle the generated image list.
      is_val: whether the images are for the validation phase during pre-training.
    Return:
      The list for the image files' directories.
    """
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    if is_val is False:
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
    """The leaky relu function.
    Args:
      x: the input feature maps.
      leak: the parameter for leaky relu.
    Return:
      The feature maps processed by non-liner layer.
    """
    return tf.maximum(x, leak*x)

def resnet_conv_block(inp, cweight, bweight, reuse, scope, activation=leaky_relu):
    """The function to forward a conv layer.
    Args:
      inp: the input feature maps.
      cweight: the filters' weights for this conv layer.
      bweight: the biases' weights for this conv layer.
      reuse: whether reuse the variables for the batch norm.
      scope: the label for this conv layer.
      activation: the activation function for this conv layer.
    Return:
      The processed feature maps.
    """
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
    """The function to forward a conv layer without biases, normalization and non-liner layer.
    Args:
      inp: the input feature maps.
      cweight: the filters' weights for this conv layer.
      reuse: whether reuse the variables for the batch norm.
      scope: the label for this conv layer.
    Return:
      The processed feature maps.
    """
    stride, no_stride = [1,2,2,1], [1,1,1,1]
    conv_output = tf.nn.conv2d(inp, cweight, no_stride, 'SAME')
    return conv_output

def normalize(inp, activation, reuse, scope):
    """The function to forward the normalization.
    Args:
      inp: the input feature maps.
      reuse: whether reuse the variables for the batch norm.
      scope: the label for this conv layer.
      activation: the activation function for this conv layer.
    Return:
      The processed feature maps.
    """
    if FLAGS.norm == 'batch_norm':
        return tf_layers.batch_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
    elif FLAGS.norm == 'layer_norm':
        return tf_layers.layer_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
    elif FLAGS.norm == 'None':
        if activation is not None:
            return activation(inp)        
        return inp
    else:
        raise ValueError('Please set correct normalization.')

## Loss functions

def mse(pred, label):
    """The MSE loss function.
    Args:
      pred: the predictions.
      label: the ground truth labels.
    Return:
      The Loss.
    """
    pred = tf.reshape(pred, [-1])
    label = tf.reshape(label, [-1])
    return tf.reduce_mean(tf.square(pred-label))

def softmaxloss(pred, label):
    """The softmax cross entropy loss function.
    Args:
      pred: the predictions.
      label: the ground truth labels.
    Return:
      The Loss.
    """
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label))

def xent(pred, label):
    """The softmax cross entropy loss function. The losses will be normalized by the shot number.
    Args:
      pred: the predictions.
      label: the ground truth labels.
    Return:
      The Loss.
    Note: with tf version <=0.12, this loss has incorrect 2nd derivatives
    """
    return tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label) / FLAGS.shot_num
