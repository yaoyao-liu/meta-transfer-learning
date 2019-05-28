##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## Tianjin University
## Email: liuyaoyao@tju.edu.cn
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

""" Trainer for meta-learning. """
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
    """The class that contains the code for the meta-train and meta-test.
    """
    def __init__(self):
        # Remove the saved datalist for a new experiment
        os.system('rm -r ./logs/processed_data/*')
        data_generator = MetaDataGenerator()
        if FLAGS.metatrain:
            # Build model for meta-train phase
            print('Building train model')
            self.model = MetaModel()
            self.model.construct_model()
            print('Building train model done')  
            # Start tensorflow session          
            self.start_session()
            # Generate data for meta-train phase
            if FLAGS.load_saved_weights:
                random.seed(5) 
            data_generator.generate_data(data_type='train')
            if FLAGS.load_saved_weights:
                random.seed(7) 
            data_generator.generate_data(data_type='test')
            if FLAGS.load_saved_weights:
                random.seed(9) 
            data_generator.generate_data(data_type='val')
        else:
            # Build model for meta-test phase
            print('Building test mdoel')
            self.model = MetaModel()
            self.model.construct_test_model()
            self.model.summ_op = tf.summary.merge_all()
            print('Building test model done')
            # Start tensorflow session 
            self.start_session()
            # Generate data for meta-test phase
            if FLAGS.load_saved_weights:
                random.seed(7) 
            data_generator.generate_data(data_type='test')
        # Load the experiment setting string from FLAGS
        exp_string = FLAGS.exp_string

        # Global initialization and starting queue
        tf.global_variables_initializer().run()
        tf.train.start_queue_runners()

        if FLAGS.metatrain:              
            # Process initialization weights for meta-train
            init_dir = FLAGS.logdir_base + 'init_weights/'
            if not os.path.exists(init_dir):
                os.mkdir(init_dir)
            pre_save_str = FLAGS.pre_string
            this_init_dir = init_dir + pre_save_str + '.pre_iter(' + str(FLAGS.pretrain_iterations) + ')/'
            if not os.path.exists(this_init_dir):
                # If there is no saved initialization weights for meta-train, load pre-train model and save initialization weights 
                os.mkdir(this_init_dir)
                if FLAGS.load_saved_weights:
                    print('Loading downloaded pretrain weights')
                    weights = np.load('logs/download_weights/weights.npy', allow_pickle=True, encoding="latin1").tolist()
                else:
                    print('Loading pretrain weights')
                    weights_save_dir_base = FLAGS.pretrain_dir
                    weights_save_dir = os.path.join(weights_save_dir_base, pre_save_str)
                    weights = np.load(os.path.join(weights_save_dir, "weights_{}.npy".format(FLAGS.pretrain_iterations)), allow_pickle=True, encoding="latin1").tolist()
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
                # If the initialization weights are already generated, load the previous saved ones
                print('Loading previous saved weights')
                weights = np.load(this_init_dir + 'weights_init.npy', allow_pickle=True, encoding="latin1").tolist()
                ss_weights = np.load(this_init_dir + 'ss_weights_init.npy', allow_pickle=True, encoding="latin1").tolist()
                fc_weights = np.load(this_init_dir + 'fc_weights_init.npy', allow_pickle=True, encoding="latin1").tolist()
                for key in weights.keys():
                    self.sess.run(tf.assign(self.model.weights[key], weights[key]))
                for key in ss_weights.keys():
                    self.sess.run(tf.assign(self.model.ss_weights[key], ss_weights[key]))
                for key in fc_weights.keys():
                    self.sess.run(tf.assign(self.model.fc_weights[key], fc_weights[key]))
                print('Weights loaded')
        else:
            # Load the saved meta model for meta-test phase
            if FLAGS.load_saved_weights:
                weights = np.load('./logs/download_weights/weights.npy', allow_pickle=True, encoding="latin1").tolist()
                ss_weights = np.load('./logs/download_weights/ss_weights.npy', allow_pickle=True, encoding="latin1").tolist()
                fc_weights = np.load('./logs/download_weights/fc_weights.npy', allow_pickle=True, encoding="latin1").tolist()
            else:
                weights = np.load(FLAGS.logdir + '/' + exp_string +  '/weights_' + str(FLAGS.test_iter) + '.npy', allow_pickle=True, encoding="latin1").tolist()
                ss_weights = np.load(FLAGS.logdir + '/' + exp_string +  '/ss_weights_' + str(FLAGS.test_iter) + '.npy', allow_pickle=True, encoding="latin1").tolist()
                fc_weights = np.load(FLAGS.logdir + '/' + exp_string +  '/fc_weights_' + str(FLAGS.test_iter) + '.npy', allow_pickle=True, encoding="latin1").tolist()
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
        """The function to start tensorflow session.
        """
        if FLAGS.full_gpu_memory_mode:
            gpu_config = tf.ConfigProto()
            gpu_config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_rate
            self.sess = tf.InteractiveSession(config=gpu_config)
        else:
            self.sess = tf.InteractiveSession()

    def train(self, data_generator):
        """The function for the meta-train phase
        Arg:
          data_generator: the data generator class for this phase
        """
        # Load the experiment setting string from FLAGS 
        exp_string = FLAGS.exp_string
        # Generate tensorboard file writer
        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + exp_string, self.sess.graph)
        print('Done initializing, start meta-training')
        # Generate empty list to record losses and accuracies
        loss_list, acc_list = [], []
        # Load the meta learning rate from FLAGS 
        train_lr = FLAGS.meta_lr
        # Load data for meta-train and meta validation
        data_generator.load_data(data_type='train')
        data_generator.load_data(data_type='val')
        # set the validation index to 0
        test_idx = 0

        for train_idx in trange(FLAGS.metatrain_iterations):
            # Load the episodes for this meta batch
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

            # Generate feed dict for the tensorflow graph
            feed_dict = {self.model.inputa: inputa, self.model.inputb: inputb, \
                self.model.labela: labela, self.model.labelb: labelb, self.model.meta_lr: train_lr}

            # Set the variables to load from the tensorflow graph
            input_tensors = [self.model.metatrain_op] # The meta train optimizer
            input_tensors.extend([self.model.total_loss]) # The meta train loss
            input_tensors.extend([self.model.total_accuracy]) # The meta train accuracy
            input_tensors.extend([self.model.training_summ_op]) # The tensorboard summary operation

            # run this meta-train iteration
            result = self.sess.run(input_tensors, feed_dict)

            # record losses, accuracies and tensorboard 
            loss_list.append(result[1])
            acc_list.append(result[2])
            train_writer.add_summary(result[3], train_idx)

            # print meta-train information on the screen after several iterations
            if (train_idx!=0) and train_idx % FLAGS.meta_print_step == 0:
                print_str = 'Iteration:' + str(train_idx)
                print_str += ' Loss:' + str(np.mean(loss_list)) + ' Acc:' + str(np.mean(acc_list))
                print(print_str)
                loss_list, acc_list = [], []

            # Save the model during meta-teain
            if train_idx % FLAGS.meta_save_step == 0:
                weights = self.sess.run(self.model.weights)
                ss_weights = self.sess.run(self.model.ss_weights)
                fc_weights = self.sess.run(self.model.fc_weights)
                np.save(FLAGS.logdir + '/' + exp_string +  '/weights_' + str(train_idx) + '.npy', weights)
                np.save(FLAGS.logdir + '/' + exp_string +  '/ss_weights_' + str(train_idx) + '.npy', ss_weights)
                np.save(FLAGS.logdir + '/' + exp_string +  '/fc_weights_' + str(train_idx) + '.npy', fc_weights)

            # Run the meta-validation during meta-train
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
                  
            # Reduce the meta learning rate to half after several iterations      
            if (train_idx!=0) and train_idx % FLAGS.lr_drop_step == 0:
                train_lr = train_lr * FLAGS.lr_drop_rate
                if train_lr < 0.1 * FLAGS.meta_lr:
                    train_lr = 0.1 * FLAGS.meta_lr
                print('Train LR: {}'.format(train_lr))

        # Save the final model
        weights = self.sess.run(self.model.weights)
        ss_weights = self.sess.run(self.model.ss_weights)
        fc_weights = self.sess.run(self.model.fc_weights)
        np.save(FLAGS.logdir + '/' + exp_string +  '/weights_' + str(train_idx+1) + '.npy', weights)
        np.save(FLAGS.logdir + '/' + exp_string +  '/ss_weights_' + str(train_idx+1) + '.npy', ss_weights)
        np.save(FLAGS.logdir + '/' + exp_string +  '/fc_weights_' + str(train_idx+1) + '.npy', fc_weights)

    def test(self, data_generator):
        """The function for the meta-test phase
        Arg:
          data_generator: the data generator class for this phase
        """
        # Set meta-test episode number
        NUM_TEST_POINTS = 600
        # Load the experiment setting string from FLAGS 
        exp_string = FLAGS.exp_string 
        np.random.seed(1)
        # Generate empty list to record accuracies
        metaval_accuracies = []
        # Load data for meta-test
        data_generator.load_data(data_type='test')
        for test_idx in trange(NUM_TEST_POINTS):
            # Load one episode for meta-test
            this_episode = data_generator.load_episode(index=test_idx, data_type='test')
            inputa = this_episode[0][np.newaxis, :]
            labela = this_episode[1][np.newaxis, :]
            inputb = this_episode[2][np.newaxis, :]
            labelb = this_episode[3][np.newaxis, :]
            feed_dict = {self.model.inputa: inputa, self.model.inputb: inputb, \
                self.model.labela: labela, self.model.labelb: labelb, self.model.meta_lr: 0.0}
            result = self.sess.run(self.model.metaval_total_accuracies, feed_dict)
            metaval_accuracies.append(result)
        # Calculate the mean accuarcies and the confidence intervals
        metaval_accuracies = np.array(metaval_accuracies)
        means = np.mean(metaval_accuracies, 0)
        stds = np.std(metaval_accuracies, 0)
        ci95 = 1.96*stds/np.sqrt(NUM_TEST_POINTS)

        # Print the meta-test results
        print('Test accuracies and confidence intervals')
        print((means, ci95))

        # Save the meta-test results in the csv files
        if not FLAGS.load_saved_weights:
            out_filename = FLAGS.logdir +'/'+ exp_string + '/' + 'result_' + str(FLAGS.shot_num) + 'shot_' + str(FLAGS.test_iter) + '.csv'
            out_pkl = FLAGS.logdir +'/'+ exp_string + '/' + 'result_' + str(FLAGS.shot_num) + 'shot_' + str(FLAGS.test_iter) + '.pkl'
            with open(out_pkl, 'wb') as f:
                pickle.dump({'mses': metaval_accuracies}, f)
            with open(out_filename, 'w') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow(['update'+str(i) for i in range(len(means))])
                writer.writerow(means)
                writer.writerow(stds)
                writer.writerow(ci95)
