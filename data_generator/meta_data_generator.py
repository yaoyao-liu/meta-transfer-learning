import numpy as np
import os
import random
import tensorflow as tf
from tqdm import trange

from tensorflow.python.platform import flags
from utils.misc import get_images

FLAGS = flags.FLAGS

class MetaDataGenerator(object):
    def __init__(self, num_samples_per_class, batch_size, config={}):
        self.batch_size = batch_size
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = config.get('num_classes', FLAGS.num_classes)
        self.img_size = config.get('img_size', (84, 84))
        self.dim_input = np.prod(self.img_size)*3
        self.dim_output = self.num_classes
        metatrain_folder = config.get('metatrain_folder', './data/meta-train/train')
        metaval_folder = config.get('metaval_folder', './data/meta-train/test')

        all_npy_dir = FLAGS.logdir_base + 'filenames_and_labels/'
        if not os.path.exists(all_npy_dir):
            os.mkdir(all_npy_dir)

        self.npy_base_dir = all_npy_dir + str(FLAGS.update_batch_size) + 'shot_' + str(FLAGS.num_classes) + 'way/'
        if not os.path.exists(self.npy_base_dir):
            os.mkdir(self.npy_base_dir)

        metatrain_folders = [os.path.join(metatrain_folder, label) \
           for label in os.listdir(metatrain_folder) \
           if os.path.isdir(os.path.join(metatrain_folder, label)) \
           ]
        metaval_folders = [os.path.join(metaval_folder, label) \
            for label in os.listdir(metaval_folder) \
            if os.path.isdir(os.path.join(metaval_folder, label)) \
            ]

        self.metatrain_character_folders = metatrain_folders
        self.metaval_character_folders = metaval_folders
    


    def make_data_list(self, train=True):
        if train:
            folders = self.metatrain_character_folders
            # number of tasks, not number of meta-iterations. (divide by metabatch size to measure)
            num_total_batches = 100000
        else:
            folders = self.metaval_character_folders
            num_total_batches = 600

        # make list of files
        if train:
            if not os.path.exists(self.npy_base_dir+'/train_filenames.npy'):
                print('Generating train filenames')
                all_filenames = []
                for _ in trange(num_total_batches):
                    sampled_character_folders = random.sample(folders, self.num_classes)
                    random.shuffle(sampled_character_folders)
                    labels_and_images = get_images(sampled_character_folders, range(self.num_classes), nb_samples=self.num_samples_per_class, shuffle=False)
                    # make sure the above isn't randomized order
                    labels = [li[0] for li in labels_and_images]
                    filenames = [li[1] for li in labels_and_images]
                    all_filenames.extend(filenames)
                np.save(self.npy_base_dir+'/train_labels.npy', labels)
                np.save(self.npy_base_dir+'/train_filenames.npy', all_filenames)
                print('Train filename list and labels saved')
            else:
                print('Train filename list and labels already exist')
    
        else:
            if FLAGS.metatrain:
                if not os.path.exists(self.npy_base_dir+'/val_filenames.npy'):
                    print('Generating val filenames')
                    all_filenames = []
                    for _ in trange(num_total_batches):
                        sampled_character_folders = random.sample(folders, self.num_classes)
                        random.shuffle(sampled_character_folders)
                        labels_and_images = get_images(sampled_character_folders, range(self.num_classes), nb_samples=self.num_samples_per_class, shuffle=False)
                        # make sure the above isn't randomized order
                        labels = [li[0] for li in labels_and_images]
                        filenames = [li[1] for li in labels_and_images]
                        all_filenames.extend(filenames)
                    np.save(self.npy_base_dir+'/val_labels.npy', labels)
                    np.save(self.npy_base_dir+'/val_filenames.npy', all_filenames)
                    print('Val filename list and labels saved')
                else:
                    print('Val filename list and labels already exist')
            else:
                if not os.path.exists(self.npy_base_dir+'/test_filenames.npy'):
                    print('Generating test filenames')
                    all_filenames = []
                    for _ in trange(num_total_batches):
                        sampled_character_folders = random.sample(folders, self.num_classes)
                        random.shuffle(sampled_character_folders)
                        labels_and_images = get_images(sampled_character_folders, range(self.num_classes), nb_samples=self.num_samples_per_class, shuffle=False)
                        # make sure the above isn't randomized order
                        labels = [li[0] for li in labels_and_images]
                        filenames = [li[1] for li in labels_and_images]
                        all_filenames.extend(filenames)
                    np.save(self.npy_base_dir+'/test_labels.npy', labels)
                    np.save(self.npy_base_dir+'/test_filenames.npy', all_filenames)
                    print('Test filename list and labels saved')
                else:
                    print('Test filename list and labels already exist')

