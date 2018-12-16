import numpy as np
import os
import random
import tensorflow as tf
from tqdm import trange

from tensorflow.python.platform import flags
from utils import get_images, get_pretrain_images

FLAGS = flags.FLAGS

class DataGenerator(object):
    def __init__(self, num_samples_per_class, batch_size, config={}):
        self.batch_size = batch_size
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = config.get('num_classes', FLAGS.num_classes)
        self.img_size = config.get('img_size', (84, 84))
        self.dim_input = np.prod(self.img_size)*3
        self.dim_output = self.num_classes
        metatrain_folder = config.get('metatrain_folder', './data/meta-train/train')
        metaval_folder = config.get('metaval_folder', './data/meta-train/test')
        pretrain_folder = config.get('pretrain_folder', FLAGS.pretrain_folders)
        pretrainval_folder = config.get('pretrainval_folder', './data/pre-train/pre-train_val')

        all_npy_dir = './filenames_and_labels/'
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
        pretrain_folders = [os.path.join(pretrain_folder, label) \
           for label in os.listdir(pretrain_folder) \
           if os.path.isdir(os.path.join(pretrain_folder, label)) \
           ]
        pretrainval_folders = [os.path.join(pretrainval_folder, label) \
           for label in os.listdir(pretrain_folder) \
           if os.path.isdir(os.path.join(pretrainval_folder, label)) \
           ]
        self.pretrain_character_folders = pretrain_folders
        self.pretrainval_character_folders = pretrainval_folders
        self.metatrain_character_folders = metatrain_folders
        self.metaval_character_folders = metaval_folders
    
    def make_data_tensor_for_pretrain_classifier(self, is_val=False):
        self.pretrain_batch_size = 64
        all_filenames_and_labels = []
        if is_val==False:
            folders = self.pretrain_character_folders
            for idx in range(len(folders)):
                path = folders[idx]     
                all_filenames_and_labels += get_pretrain_images(path, idx)
        else:
            folders = self.pretrainval_character_folders
            for idx in range(len(folders)):
                path = folders[idx]     
                all_filenames_and_labels += get_pretrain_images(path, idx)
        random.shuffle(all_filenames_and_labels)
        all_labels = [li[0] for li in all_filenames_and_labels]
        all_filenames = [li[1] for li in all_filenames_and_labels]
        filename_queue = tf.train.string_input_producer(tf.convert_to_tensor(all_filenames), shuffle=False)
        label_queue = tf.train.slice_input_producer([tf.convert_to_tensor(all_labels)], shuffle=False)
        image_reader = tf.WholeFileReader()
        _, image_file = image_reader.read(filename_queue)

        image = tf.image.decode_jpeg(image_file, channels=3)
        image.set_shape((self.img_size[0],self.img_size[1],3))
        image = tf.reshape(image, [self.dim_input])
        image = tf.cast(image, tf.float32) / 255.0

        num_preprocess_threads = 1 # TODO - enable this to be set to >1
        min_queue_examples = 256
        batch_image_size = self.pretrain_batch_size
        print('Batching images')
        image_batch, label_batch = tf.train.batch(
                [image, label_queue],
                batch_size = batch_image_size,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * batch_image_size,
                )

        label_batch = tf.one_hot(tf.reshape(label_batch, [-1]), FLAGS.pretrain_class_num)
        return image_batch, label_batch

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
            if FLAGS.train:
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

