##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## NUS School of Computing
## Email: yaoyaoliu@outlook.com
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import csv
import pickle
import random
import numpy as np
import tensorflow as tf
import cv2

from tqdm import trange
from data_generator.meta_data_generator import MetaDataGenerator
from models.meta_model import MetaModel
from tensorflow.python.platform import flags
from utils.misc import process_batch, process_batch_augmentation

FLAGS = flags.FLAGS

class MetaTrainer:
    def __init__(self):
        data_generator = MetaDataGenerator()
        if FLAGS.metatrain:
            print('Building train model')
            self.model = MetaModel()
            self.model.construct_model()
            print('Finish building train model')
            
            self.start_session()

            random.seed(5) # The same random seed with MAML
            data_generator.generate_data(data_type='train')
            random.seed(6) # The same random seed with MAML
            data_generator.generate_data(data_type='test')
            random.seed(7) # The same random seed with MAML
            data_generator.generate_data(data_type='val')

        else:
            print('Building test mdoel')
            self.model = MetaModel()
            self.model.construct_test_model()
            self.model.summ_op = tf.summary.merge_all()
            print('Finish building test model')

            self.start_session()

            random.seed(6) # The same random seed with MAML
            data_generator.generate_data(data_type='test')

        exp_string = FLAGS.exp_string

        tf.global_variables_initializer().run()
        tf.train.start_queue_runners()

        if FLAGS.metatrain:
            init_dir = FLAGS.logdir_base + 'init_weights/'
            if not os.path.exists(init_dir):
                os.mkdir(init_dir)
            pre_save_str = FLAGS.pre_string
            this_init_dir = init_dir + pre_save_str + '.pre_iter(' + str(FLAGS.pretrain_iterations) + ')/'
            if not os.path.exists(this_init_dir):
                os.mkdir(this_init_dir)
                print('Loading pretrain weights')
                weights_save_dir_base = FLAGS.pretrain_dir
                weights_save_dir = os.path.join(weights_save_dir_base, pre_save_str)
                weights = np.load(os.path.join(weights_save_dir, "weights_{}.npy".format(FLAGS.pretrain_iterations))).tolist()
                bais_list = [bais_item for bais_item in weights.keys() if '_bias' in bais_item]
                for bais_key in bais_list:
                    self.sess.run(tf.assign(self.model.ss_weights[bais_key], weights[bais_key]))
                for key in weights.keys():
                    self.sess.run(tf.assign(self.model.weights[key], weights[key]))
                print('Pretrain weights loaded, saving init weights')
                new_weights = self.sess.run(self.model.weights)
                ss_weights = self.sess.run(self.model.ss_weights)
                fc_weights = self.sess.run(self.model.fc_weights)
                np.save(this_init_dir + 'weights_init.npy', new_weights)
                np.save(this_init_dir + 'ss_weights_init.npy', ss_weights)
                np.save(this_init_dir + 'fc_weights_init.npy', fc_weights)
            else:
                print('Loading previous saved weights')
                weights = np.load(this_init_dir + 'weights_init.npy').tolist()
                ss_weights = np.load(this_init_dir + 'ss_weights_init.npy').tolist()
                fc_weights = np.load(this_init_dir + 'fc_weights_init.npy').tolist()
                for key in weights.keys():
                    self.sess.run(tf.assign(self.model.weights[key], weights[key]))
                for key in ss_weights.keys():
                    self.sess.run(tf.assign(self.model.ss_weights[key], ss_weights[key]))
                for key in fc_weights.keys():
                    self.sess.run(tf.assign(self.model.fc_weights[key], fc_weights[key]))
                print('Weights loaded')
        else:
            weights = np.load(FLAGS.logdir + '/' + exp_string +  '/weights_' + str(FLAGS.test_iter) + '.npy').tolist()
            ss_weights = np.load(FLAGS.logdir + '/' + exp_string +  '/ss_weights_' + str(FLAGS.test_iter) + '.npy').tolist()
            fc_weights = np.load(FLAGS.logdir + '/' + exp_string +  '/fc_weights_' + str(FLAGS.test_iter) + '.npy').tolist()
            print((lr_weights))
            for key in weights.keys():
                self.sess.run(tf.assign(self.model.weights[key], weights[key]))
            for key in ss_weights.keys():
                self.sess.run(tf.assign(self.model.ss_weights[key], ss_weights[key]))
            for key in fc_weights.keys():
                self.sess.run(tf.assign(self.model.fc_weights[key], fc_weights[key]))
            print('Weights loaded')
            print('Test iter: ' + str(FLAGS.test_iter))

        if FLAGS.metatrain:
            self.train(data_generator)
        else:
            self.test(data_generator)

    def start_session(self):
        if FLAGS.full_gpu_memory_mode:
            gpu_config = tf.ConfigProto()
            gpu_config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_rate
            self.sess = tf.InteractiveSession(config=gpu_config)
        else:
            self.sess = tf.InteractiveSession()

    def train(self, data_generator):
        exp_string = FLAGS.exp_string
        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + exp_string, self.sess.graph)
        print('Done initializing, starting training')
        loss_list, acc_list = [], []
        train_lr = FLAGS.meta_lr

        data_generator.load_data(data_type='train')
        data_generator.load_data(data_type='val')

        test_idx = 0

        for train_idx in trange(FLAGS.metatrain_iterations):

            inputa = []
            labela = []
            inputb = []
            labelb = []
            for meta_batch_idx in range(FLAGS.meta_batch_size):
                this_episode = data_generator.load_episode(index=train_idx*FLAGS.meta_batch_size+meta_batch_idx, data_type='train')
                inputa.append(this_episode[0])
                labela.append(this_episode[1])
                inputb.append(this_episode[2])
                labelb.append(this_episode[3])

            inputa = np.array(inputa)
            labela = np.array(labela)
            inputb = np.array(inputb)
            labelb = np.array(labelb)

            feed_dict = {self.model.inputa: inputa, self.model.inputb: inputb, \
                self.model.labela: labela, self.model.labelb: labelb, self.model.meta_lr: train_lr}

            input_tensors = [self.model.metatrain_op]
            input_tensors.extend([self.model.total_loss])
            input_tensors.extend([self.model.total_accuracy])
            input_tensors.extend([self.model.training_summ_op])

            result = self.sess.run(input_tensors, feed_dict)

            loss_list.append(result[1])
            acc_list.append(result[2])
            train_writer.add_summary(result[3], train_idx)

            if (train_idx!=0) and train_idx % FLAGS.meta_print_step == 0:
                print_str = 'Iteration:' + str(train_idx)
                print_str += ' Loss:' + str(np.mean(loss_list)) + ' Acc:' + str(np.mean(acc_list))
                print(print_str)
                loss_list, acc_list = [], []

            if train_idx % FLAGS.meta_save_step == 0:
                weights = self.sess.run(self.model.weights)
                ss_weights = self.sess.run(self.model.ss_weights)
                fc_weights = self.sess.run(self.model.fc_weights)
                np.save(FLAGS.logdir + '/' + exp_string +  '/weights_' + str(train_idx) + '.npy', weights)
                np.save(FLAGS.logdir + '/' + exp_string +  '/ss_weights_' + str(train_idx) + '.npy', ss_weights)
                np.save(FLAGS.logdir + '/' + exp_string +  '/fc_weights_' + str(train_idx) + '.npy', fc_weights)

            if train_idx % FLAGS.meta_val_print_step == 0:
                test_loss = []
                test_accs = []
                for test_itr in range(FLAGS.meta_intrain_val_sample):          
                    this_episode = data_generator.load_episode(index=test_itr, data_type='val')
                    test_inputa = this_episode[0][np.newaxis, :]
                    test_labela = this_episode[1][np.newaxis, :]
                    test_inputb = this_episode[2][np.newaxis, :]
                    test_labelb = this_episode[3][np.newaxis, :]

                    test_feed_dict = {self.model.inputa: test_inputa, self.model.inputb: test_inputb, \
                        self.model.labela: test_labela, self.model.labelb: test_labelb, \
                        self.model.meta_lr: 0.0}
                    test_input_tensors = [self.model.total_loss, self.model.total_accuracy]
                    test_result = self.sess.run(test_input_tensors, test_feed_dict)
                    test_loss.append(test_result[0])
                    test_accs.append(test_result[1])

                valsum_feed_dict = {self.model.input_val_loss: \
                    np.mean(test_loss)*np.float(FLAGS.meta_batch_size)/np.float(FLAGS.shot_num), \
                    self.model.input_val_acc: np.mean(test_accs)*np.float(FLAGS.meta_batch_size)}
                valsum = self.sess.run(self.model.val_summ_op, valsum_feed_dict)
                train_writer.add_summary(valsum, train_idx)
                print_str = '[***] Val Loss:' + str(np.mean(test_loss)*FLAGS.meta_batch_size) + \
                    ' Val Acc:' + str(np.mean(test_accs)*FLAGS.meta_batch_size)
                print(print_str)
                        
            if (train_idx!=0) and train_idx % FLAGS.lr_drop_step == 0:
                train_lr = train_lr * FLAGS.lr_drop_rate
                if train_lr < 0.1 * FLAGS.meta_lr:
                    train_lr = 0.1 * FLAGS.meta_lr
                print('Train LR: {}'.format(train_lr))

        weights = self.sess.run(self.model.weights)
        ss_weights = self.sess.run(self.model.ss_weights)
        fc_weights = self.sess.run(self.model.fc_weights)
        np.save(FLAGS.logdir + '/' + exp_string +  '/weights_' + str(train_idx+1) + '.npy', weights)
        np.save(FLAGS.logdir + '/' + exp_string +  '/ss_weights_' + str(train_idx+1) + '.npy', ss_weights)
        np.save(FLAGS.logdir + '/' + exp_string +  '/fc_weights_' + str(train_idx+1) + '.npy', fc_weights)


    def test(self, data_generator):
        NUM_TEST_POINTS = 600
        exp_string = FLAGS.exp_string 
        np.random.seed(1)
        metaval_accuracies = []
        data_generator.load_data(data_type='test')

        for test_idx in trange(NUM_TEST_POINTS):
            this_episode = data_generator.load_episode(index=test_idx, data_type='test')
            inputa = this_episode[0][np.newaxis, :]
            labela = this_episode[1][np.newaxis, :]
            inputb = this_episode[2][np.newaxis, :]
            labelb = this_episode[3][np.newaxis, :]
            feed_dict = {self.model.inputa: inputa, self.model.inputb: inputb, \
                self.model.labela: labela, self.model.labelb: labelb, self.model.meta_lr: 0.0}
            result = self.sess.run(self.model.metaval_total_accuracies, feed_dict)
            metaval_accuracies.append(result)

        metaval_accuracies = np.array(metaval_accuracies)
        means = np.mean(metaval_accuracies, 0)
        max_idx = np.argmax(means)
        max_acc = np.max(means)
        stds = np.std(metaval_accuracies, 0)
        ci95 = 1.96*stds/np.sqrt(NUM_TEST_POINTS)
        max_ci95 = ci95[max_idx]

        print('Mean validation accuracy and confidence intervals')
        print((means, ci95))

        print('***** Best Acc: '+ str(max_acc) + ' CI95: ' + str(max_ci95))

        if FLAGS.base_augmentation:
            out_filename = FLAGS.logdir +'/'+ exp_string + '/' + 'result_aug_' + str(FLAGS.shot_num) + 'shot_' + str(FLAGS.test_iter) + '.csv'
            out_pkl = FLAGS.logdir +'/'+ exp_string + '/' + 'result_aug_' + str(FLAGS.shot_num) + 'shot_' + str(FLAGS.test_iter) + '.pkl'
        else:
            out_filename = FLAGS.logdir +'/'+ exp_string + '/' + 'result_noaug_' + str(FLAGS.shot_num) + 'shot_' + str(FLAGS.test_iter) + '.csv'
            out_pkl = FLAGS.logdir +'/'+ exp_string + '/' + 'result_noaug_' + str(FLAGS.shot_num) + 'shot_' + str(FLAGS.test_iter) + '.pkl'
        with open(out_pkl, 'wb') as f:
            pickle.dump({'mses': metaval_accuracies}, f)
        with open(out_filename, 'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(['update'+str(i) for i in range(len(means))])
            writer.writerow(means)
            writer.writerow(stds)
            writer.writerow(ci95)

