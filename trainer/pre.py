import os
import csv
import pickle
import random
import numpy as np
import tensorflow as tf
import cv2
import pdb

from tqdm import trange
from data_generator.pre_data_generator import PreDataGenerator
from models.pre_model import PreModel
from tensorflow.python.platform import flags
from utils.misc import process_batch

FLAGS = flags.FLAGS

class PreTrainer:
    def __init__(self):
        print('Generating pre-training class')
        pre_train_data_generator = PreDataGenerator(FLAGS.update_batch_size, FLAGS.meta_batch_size)
        pretrain_input, pretrain_label = pre_train_data_generator.make_data_tensor_for_pretrain_classifier()
        pre_train_input_tensors = {'pretrain_input': pretrain_input, 'pretrain_label': pretrain_label}

        dim_output = pre_train_data_generator.dim_output
        dim_input = pre_train_data_generator.dim_input
        self.model = PreModel(dim_input, dim_output)

        self.model.construct_pretrain_model(input_tensors=pre_train_input_tensors)
        self.model.pretrain_summ_op = tf.summary.merge_all()

        if FLAGS.full_gpu_memory_mode:
            gpu_config = tf.ConfigProto()
            gpu_config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_rate
            self.sess = tf.InteractiveSession(config=gpu_config)
        else:
            self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        tf.train.start_queue_runners()


    def pre_train(self):
        SUMMARY_INTERVAL = 10
        SAVE_INTERVAL = 1000
        PRINT_INTERVAL = 1000
        LR_DROP_STEP = FLAGS.pre_lr_dropstep
        pretrain_iterations = FLAGS.pretrain_resnet_iterations
        weights_save_dir_base = FLAGS.pretrain_dir

        weights_save_dir = os.path.join(weights_save_dir_base, "pretrain_dropout_{}_activation_{}_prelr_{}_dropstep_{}".format(FLAGS.pretrain_dropout, FLAGS.activation, FLAGS.pre_lr, FLAGS.pre_lr_dropstep) + FLAGS.pretrain_label)
        if FLAGS.pre_lr_stop:
            weights_save_dir = weights_save_dir+'_prelrstop'
        if not os.path.exists(weights_save_dir):
            os.mkdir(weights_save_dir)

        pretrain_writer = tf.summary.FileWriter(weights_save_dir, self.sess.graph)
        pre_lr = FLAGS.pre_lr
        
        for itr in trange(pretrain_iterations):
            feed_dict = {self.model.pretrain_lr: pre_lr}
            input_tensors = [self.model.pretrain_op, self.model.pretrain_summ_op]
            input_tensors.extend([self.model.pretrain_task_loss, self.model.pretrain_task_accuracy])
            result = self.sess.run(input_tensors, feed_dict)
            if (itr!=0) and itr % PRINT_INTERVAL == 0:
                print_str = '[*] Loss: ' + str(result[-2]) + ', Acc: ' + str(result[-1])
                print(print_str)

            if itr % SUMMARY_INTERVAL == 0:
                pretrain_writer.add_summary(result[1], itr)

            if (itr!=0) and itr % LR_DROP_STEP == 0:
                pre_lr = pre_lr * 0.5
                if FLAGS.pre_lr_stop:
                    if pre_lr<0.0001:
                        pre_lr = 0.0001
                    
            if (itr!=0) and itr % SAVE_INTERVAL == 0:
                print('Saving pretrain weights to npy.')
                weights = self.sess.run(self.model.weights)
                np.save(os.path.join(weights_save_dir, "weights_{}.npy".format(itr)), weights)

