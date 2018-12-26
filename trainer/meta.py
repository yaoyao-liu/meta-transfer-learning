import os
import csv
import pickle
import random
import numpy as np
import tensorflow as tf
import cv2
import pdb

from tqdm import trange
from data_generator.meta_data_generator import MetaDataGenerator
from models.meta_model import MetaModel
from tensorflow.python.platform import flags
from utils.misc import process_batch

FLAGS = flags.FLAGS

class MetaTrainer:
    def __init__(self):
        if FLAGS.metatrain == False:
            orig_meta_batch_size = FLAGS.meta_batch_size
            # always use meta batch size of 1 when testing.
            FLAGS.meta_batch_size = 1
            test_data_generator = MetaDataGenerator(FLAGS.update_batch_size*2, FLAGS.meta_batch_size)
            dim_output = test_data_generator.dim_output
            dim_input = test_data_generator.dim_input
            print('Building test mdoel')
            self.model = MetaModel(dim_input, dim_output)
            print('Finish building train model')
            self.model.construct_test_model(prefix='metaval_')
            self.model.summ_op = tf.summary.merge_all()
            random.seed(6)
            test_data_generator.make_data_list(train=False)
            FLAGS.meta_batch_size = orig_meta_batch_size

        else:
            data_generator = MetaDataGenerator(FLAGS.update_batch_size+15, FLAGS.meta_batch_size)
            test_data_generator = MetaDataGenerator(FLAGS.update_batch_size*2, FLAGS.meta_batch_size)
            dim_output = data_generator.dim_output
            dim_input = data_generator.dim_input
            print('Building train model')
            self.model = MetaModel(dim_input, dim_output)
            print('Finish building train model')
            self.model.construct_model(prefix='metatrain_')
            self.model.summ_op = tf.summary.merge_all()
            random.seed(5)
            data_generator.make_data_list(train=True)
            random.seed(6)
            test_data_generator.make_data_list(train=False)


        if FLAGS.full_gpu_memory_mode:
            gpu_config = tf.ConfigProto()
            gpu_config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_rate
            self.sess = tf.InteractiveSession(config=gpu_config)
        else:
            self.sess = tf.InteractiveSession()

        exp_string = FLAGS.exp_string

        tf.global_variables_initializer().run()
        tf.train.start_queue_runners()

        if FLAGS.metatrain or FLAGS.test_iter==0:
            print('Loading pretrain weights pretrain_dropout_{}_activation_{}_prelr_{}_dropstep_{}_label_{}_itr_{}'.format(FLAGS.pretrain_dropout, FLAGS.activation, FLAGS.pre_lr, FLAGS.pre_lr_dropstep, FLAGS.pretrain_label, FLAGS.pretrain_resnet_iterations))
            weights_save_dir_base = FLAGS.pretrain_dir
            weights_save_dir = os.path.join(weights_save_dir_base, "pretrain_dropout_{}_activation_{}_prelr_{}_dropstep_{}".format(FLAGS.pretrain_dropout, FLAGS.activation, FLAGS.pre_lr, FLAGS.pre_lr_dropstep) + FLAGS.pretrain_label)
            if FLAGS.pre_lr_stop:
                weights_save_dir = weights_save_dir+'_prelrstop'
            weights = np.load(os.path.join(weights_save_dir, "weights_{}.npy".format(FLAGS.pretrain_resnet_iterations))).tolist()
            bais_list = [bais_item for bais_item in weights.keys() if '_bias' in bais_item]
            for bais_key in bais_list:
                self.sess.run(tf.assign(self.model.ss_weights[bais_key], weights[bais_key]))
            for key in weights.keys():
                self.sess.run(tf.assign(self.model.weights[key], weights[key]))
            print('Weights loaded')
        else:
            weights = np.load(FLAGS.logdir + '/' + exp_string +  '/weights_' + str(FLAGS.test_iter) + '.npy').tolist()
            ss_weights = np.load(FLAGS.logdir + '/' + exp_string +  '/ss_weights_' + str(FLAGS.test_iter) + '.npy').tolist()
            fc_weights = np.load(FLAGS.logdir + '/' + exp_string +  '/fc_weights_' + str(FLAGS.test_iter) + '.npy').tolist()
            for key in weights.keys():
                self.sess.run(tf.assign(self.model.weights[key], weights[key]))
            for key in ss_weights.keys():
                self.sess.run(tf.assign(self.model.ss_weights[key], ss_weights[key]))
            for key in fc_weights.keys():
                self.sess.run(tf.assign(self.model.fc_weights[key], fc_weights[key]))
            print('Weights loaded')
            print('Info: ' + exp_string)
            print('Test iter: ' + str(FLAGS.test_iter))
           

    def train(self):

        exp_string = FLAGS.exp_string

        SUMMARY_INTERVAL = 100
        SAVE_INTERVAL = FLAGS.save_step
        PRINT_INTERVAL = 100
        TEST_PRINT_INTERVAL = PRINT_INTERVAL*5
        LR_DROP_STEP = FLAGS.lr_drop_step

        if FLAGS.log:
            train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + exp_string, self.sess.graph)
        print('Done initializing, starting training')
        maml_loss, maml_acc = [], []

        num_classes = FLAGS.num_classes
        multitask_weights, reg_weights = [], []

        train_lr = 0.001

        num_samples_per_class = FLAGS.update_batch_size+15
        task_num = FLAGS.num_classes * num_samples_per_class

        num_samples_per_class_test = FLAGS.update_batch_size*2
        test_task_num = FLAGS.num_classes * num_samples_per_class_test

        epitr_sample_num = FLAGS.update_batch_size
        epite_sample_num = 15
        test_task_sample_num = FLAGS.update_batch_size
        dim_input = 21168

        all_npy_dir = FLAGS.logdir_base + 'filenames_and_labels/'
        npy_base_dir = all_npy_dir + str(FLAGS.update_batch_size) + 'shot_' + str(FLAGS.num_classes) + 'way/'
        all_filenames = np.load(npy_base_dir + 'train_filenames.npy').tolist()
        labels = np.load(npy_base_dir + 'train_labels.npy').tolist()

        all_test_filenames = np.load(npy_base_dir + 'val_filenames.npy').tolist()
        test_labels = np.load(npy_base_dir + 'val_labels.npy').tolist()

        test_idx = 0

        for train_idx in trange(FLAGS.metatrain_iterations):

            inputa = []
            labela = []
            inputb = []
            labelb = []
            for meta_batch_idx in range(FLAGS.meta_batch_size):

                this_task_filenames = all_filenames[(train_idx*FLAGS.meta_batch_size+meta_batch_idx)*task_num:(train_idx*FLAGS.meta_batch_size+meta_batch_idx+1)*task_num]

                this_task_tr_filenames = []
                this_task_tr_labels = []
                this_task_te_filenames = []
                this_task_te_labels = []
                for class_k in range(FLAGS.num_classes):
                    this_class_filenames = this_task_filenames[class_k*num_samples_per_class:(class_k+1)*num_samples_per_class]
                    this_class_label = labels[class_k*num_samples_per_class:(class_k+1)*num_samples_per_class]
                    this_task_tr_filenames += this_class_filenames[0:epitr_sample_num]
                    this_task_tr_labels += this_class_label[0:epitr_sample_num]
                    this_task_te_filenames += this_class_filenames[epitr_sample_num:]
                    this_task_te_labels += this_class_label[epitr_sample_num:]

                this_inputa, this_labela = process_batch(this_task_tr_filenames, this_task_tr_labels, dim_input, epitr_sample_num, reshape_with_one=False)
                this_inputb, this_labelb = process_batch(this_task_te_filenames, this_task_te_labels, dim_input, epite_sample_num, reshape_with_one=False)
                inputa.append(this_inputa)
                labela.append(this_labela)
                inputb.append(this_inputb)
                labelb.append(this_labelb)

            inputa = np.array(inputa)
            labela = np.array(labela)
            inputb = np.array(inputb)
            labelb = np.array(labelb)

            feed_dict = {self.model.inputa: inputa, self.model.inputb: inputb,  self.model.labela: labela, self.model.labelb: labelb, self.model.meta_lr: train_lr}

            input_tensors = [self.model.metatrain_op]

            if (train_idx % SUMMARY_INTERVAL == 0 or train_idx % PRINT_INTERVAL == 0):
                input_tensors.extend([self.model.summ_op, self.model.total_loss])
                input_tensors.extend([self.model.total_accuracy])

            result = self.sess.run(input_tensors, feed_dict)

            if train_idx % SUMMARY_INTERVAL == 0:
                maml_loss.append(result[2])
                if FLAGS.log:
                    train_writer.add_summary(result[1], train_idx)
                maml_acc.append(result[3])

            if (train_idx!=0) and train_idx % PRINT_INTERVAL == 0:
                print_str = 'Iteration:' + str(train_idx)
                print_str += ' Loss:' + str(np.mean(maml_loss)) + ' Acc:' + str(np.mean(maml_acc))
                print(print_str)
                maml_loss, maml_acc = [], []

            if (train_idx!=0) and train_idx % SAVE_INTERVAL == 0:
                weights = self.sess.run(self.model.weights)
                ss_weights = self.sess.run(self.model.ss_weights)
                fc_weights = self.sess.run(self.model.fc_weights)
                np.save(FLAGS.logdir + '/' + exp_string +  '/weights_' + str(train_idx) + '.npy', weights)
                np.save(FLAGS.logdir + '/' + exp_string +  '/ss_weights_' + str(train_idx) + '.npy', ss_weights)
                np.save(FLAGS.logdir + '/' + exp_string +  '/fc_weights_' + str(train_idx) + '.npy', fc_weights)

            if (train_idx!=0) and train_idx % TEST_PRINT_INTERVAL == 0:
                test_loss = []
                test_accs = []

                for test_itr in range(10):            

                    this_test_task_filenames = all_test_filenames[test_idx*test_task_num:(test_idx+1)*test_task_num]
                    this_test_task_tr_filenames = []
                    this_test_task_tr_labels = []
                    this_test_task_te_filenames = []
                    this_test_task_te_labels = []
                    for class_k in range(FLAGS.num_classes):
                        this_test_class_filenames = this_test_task_filenames[class_k*num_samples_per_class_test:(class_k+1)*num_samples_per_class_test]
                        this_test_class_label = test_labels[class_k*num_samples_per_class_test:(class_k+1)*num_samples_per_class_test]
                        this_test_task_tr_filenames += this_test_class_filenames[0:test_task_sample_num]
                        this_test_task_tr_labels += this_test_class_label[0:test_task_sample_num]
                        this_test_task_te_filenames += this_test_class_filenames[test_task_sample_num:]
                        this_test_task_te_labels += this_test_class_label[test_task_sample_num:]

                    test_inputa, test_labela = process_batch(this_test_task_tr_filenames, this_test_task_tr_labels, dim_input, test_task_sample_num)
                    test_inputb, test_labelb = process_batch(this_test_task_te_filenames, this_test_task_te_labels, dim_input, test_task_sample_num)

                    test_feed_dict = {self.model.inputa: test_inputa, self.model.inputb: test_inputb, self.model.labela: test_labela, self.model.labelb: test_labelb, self.model.meta_lr: 0.0}
                    test_input_tensors = [self.model.total_loss, self.model.total_accuracy]

                    test_result = self.sess.run(test_input_tensors, test_feed_dict)

                    test_loss.append(test_result[0])
                    test_accs.append(test_result[1])
                
                    test_idx += 1

                print_str = '[***] Val Loss:' + str(np.mean(test_loss)*FLAGS.meta_batch_size) + ' Val Acc:' + str(np.mean(test_accs)*FLAGS.meta_batch_size)
                print(print_str)
                        

            if (train_idx!=0) and train_idx % LR_DROP_STEP == 0:
                train_lr = train_lr * 0.5
                if train_lr < 0.0001:
                    train_lr = 0.0001
                print('Train LR: {}'.format(train_lr))

            if train_idx > 20000:
                SAVE_INTERVAL = 5000

        weights = self.sess.run(self.model.weights)
        ss_weights = self.sess.run(self.model.ss_weights)
        fc_weights = self.sess.run(self.model.fc_weights)
        np.save(FLAGS.logdir + '/' + exp_string +  '/weights_' + str(train_idx+1) + '.npy', weights)
        np.save(FLAGS.logdir + '/' + exp_string +  '/ss_weights_' + str(train_idx+1) + '.npy', ss_weights)
        np.save(FLAGS.logdir + '/' + exp_string +  '/fc_weights_' + str(train_idx+1) + '.npy', fc_weights)


    def test(self):

        exp_string = FLAGS.exp_string
        NUM_TEST_POINTS = 600   
        num_classes = FLAGS.num_classes

        np.random.seed(1)

        metaval_accuracies = []

        num_samples_per_class = FLAGS.update_batch_size*2
        task_num = FLAGS.num_classes * num_samples_per_class
        half_num_samples = num_samples_per_class/2
        dim_input = 21168

        all_npy_dir = './filenames_and_labels/'
        npy_base_dir = all_npy_dir + str(FLAGS.update_batch_size) + 'shot_' + str(FLAGS.num_classes) + 'way/'
        all_filenames = np.load(npy_base_dir + 'test_filenames.npy').tolist()
        labels = np.load(npy_base_dir + 'test_labels.npy').tolist()

        for test_idx in trange(NUM_TEST_POINTS):

            this_task_filenames = all_filenames[test_idx*task_num:(test_idx+1)*task_num]

            this_task_tr_filenames = []
            this_task_tr_labels = []
            this_task_te_filenames = []
            this_task_te_labels = []
            for class_k in range(FLAGS.num_classes):
                this_class_filenames = this_task_filenames[class_k*num_samples_per_class:(class_k+1)*num_samples_per_class]
                this_class_label = labels[class_k*num_samples_per_class:(class_k+1)*num_samples_per_class]
                this_task_tr_filenames += this_class_filenames[0:half_num_samples]
                this_task_tr_labels += this_class_label[0:half_num_samples]
                this_task_te_filenames += this_class_filenames[half_num_samples:]
                this_task_te_labels += this_class_label[half_num_samples:]

            inputa, labela = process_batch(this_task_tr_filenames, this_task_tr_labels, dim_input, half_num_samples)
            inputb, labelb = process_batch(this_task_te_filenames, this_task_te_labels, dim_input, half_num_samples)

            feed_dict = {self.model.inputa: inputa, self.model.inputb: inputb,  self.model.labela: labela, self.model.labelb: labelb, self.model.meta_lr: 0.0}

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

        out_filename = FLAGS.logdir +'/'+ exp_string + '/' + 'new_test_ubs' + str(FLAGS.update_batch_size) + '_stepsize' + str(FLAGS.update_lr) + '_testiter' + str(FLAGS.test_iter) + '.csv'
        out_pkl = FLAGS.logdir +'/'+ exp_string + '/' + 'new_test_ubs' + str(FLAGS.update_batch_size) + '_stepsize' + str(FLAGS.update_lr) + '_testiter' + str(FLAGS.test_iter) + '.pkl'
        with open(out_pkl, 'wb') as f:
            pickle.dump({'mses': metaval_accuracies}, f)
        with open(out_filename, 'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(['update'+str(i) for i in range(len(means))])
            writer.writerow(means)
            writer.writerow(stds)
            writer.writerow(ci95)

