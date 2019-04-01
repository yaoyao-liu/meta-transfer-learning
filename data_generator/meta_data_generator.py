##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## NUS School of Computing
## Email: yaoyao.liu@u.nus.edu
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np
import os
import random
import tensorflow as tf

from tqdm import trange
from tensorflow.python.platform import flags
from utils.misc import get_images

FLAGS = flags.FLAGS

class MetaDataGenerator(object):
    def __init__(self, num_samples_per_class):
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = FLAGS.way_num
        metatrain_folder = FLAGS.metatrain_dir
        metatest_folder = FLAGS.metatest_dir
        metaval_folder = FLAGS.metaval_dir

        filename_dir = FLAGS.logdir_base + 'filenames_and_labels/'
        if not os.path.exists(filename_dir):
            os.mkdir(filename_dir)

        self.this_setting_filename_dir = filename_dir + str(FLAGS.shot_num) + 'shot_' + str(FLAGS.way_num) + 'way/'
        if not os.path.exists(self.this_setting_filename_dir):
            os.mkdir(self.this_setting_filename_dir)

        metatrain_folders = [os.path.join(metatrain_folder, label) \
           for label in os.listdir(metatrain_folder) \
           if os.path.isdir(os.path.join(metatrain_folder, label)) \
           ]
        metatest_folders = [os.path.join(metatest_folder, label) \
            for label in os.listdir(metatest_folder) \
            if os.path.isdir(os.path.join(metatest_folder, label)) \
            ]
        metaval_folders = [os.path.join(metaval_folder, label) \
            for label in os.listdir(metaval_folder) \
            if os.path.isdir(os.path.join(metaval_folder, label)) \
            ]

        self.metatrain_character_folders = metatrain_folders
        self.metatest_character_folders = metatest_folders
        self.metaval_character_folders = metaval_folders
    

    def make_data_list(self, data_type='train'):
        if data_type=='train':
            folders = self.metatrain_character_folders
            num_total_batches = 80000
        elif data_type=='test':
            folders = self.metatest_character_folders
            num_total_batches = 600
        elif data_type=='val':
            folders = self.metaval_character_folders
            num_total_batches = 600
        else:
            print('Please check data list type')

        if not os.path.exists(self.this_setting_filename_dir+'/' + data_type + '_filenames.npy'):
            print('Generating ' + data_type + ' filenames')
            all_filenames = []
            for _ in trange(num_total_batches):
                sampled_character_folders = random.sample(folders, self.num_classes)
                random.shuffle(sampled_character_folders)
                labels_and_images = get_images(sampled_character_folders, range(self.num_classes), nb_samples=self.num_samples_per_class, shuffle=False)
                # make sure the above isn't randomized order
                labels = [li[0] for li in labels_and_images]
                filenames = [li[1] for li in labels_and_images]
                all_filenames.extend(filenames)
            np.save(self.this_setting_filename_dir+'/' + data_type + '_labels.npy', labels)
            np.save(self.this_setting_filename_dir+'/' + data_type + '_filenames.npy', all_filenames)
            print('The ' + data_type + ' filename and label lists are saved')
        else:
            print('The ' + data_type + ' filename and label lists have already been created')


