import numpy as np
import os
import random
import tensorflow as tf
from tqdm import trange

from tensorflow.python.platform import flags
from utils.misc import get_pretrain_images

FLAGS = flags.FLAGS

class PreDataGenerator(object):
    def __init__(self, num_samples_per_class, batch_size, config={}):
        self.batch_size = batch_size
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = config.get('num_classes', FLAGS.num_classes)
        self.img_size = config.get('img_size', (84, 84))
        self.dim_input = np.prod(self.img_size)*3
        self.dim_output = self.num_classes
        pretrain_folder = config.get('pretrain_folder', FLAGS.pretrain_folders)
        pretrainval_folder = config.get('pretrainval_folder', './data/pre-train/pre-train_val')

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
    
    def make_data_tensor_for_pretrain_classifier(self, is_val=False):
        print('Generating pre-training data')
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


