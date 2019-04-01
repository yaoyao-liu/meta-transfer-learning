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
import sys
import tensorflow as tf
from models import Models
from tensorflow.python.platform import flags
from utils.misc import mse, softmaxloss, xent, resnet_conv_block, resnet_nob_conv_block, normalize

FLAGS = flags.FLAGS

class MetaModel(Models):

    def construct_model(self, prefix='metatrain_'):
        self.inputa = tf.placeholder(tf.float32)
        self.inputb = tf.placeholder(tf.float32)
        self.labela = tf.placeholder(tf.float32)
        self.labelb = tf.placeholder(tf.float32)

        with tf.variable_scope('meta-model', reuse=None) as training_scope:

            self.ss_weights = ss_weights = self.construct_resnet_ss_weights()
            self.weights = weights = self.construct_resnet_weights()
            self.fc_weights = fc_weights = self.construct_fc_weights()

            num_updates = FLAGS.train_base_epoch_num

            def task_metalearn(inp, reuse=True):

                inputa, inputb, labela, labelb = inp
                task_outputbs, task_lossesb = [], []

                emb_outputa = self.forward_resnet(inputa, weights, ss_weights, reuse=reuse)
                emb_outputb = self.forward_resnet(inputb, weights, ss_weights, reuse=True)

                outputa = self.forward_fc(emb_outputa, fc_weights)
                maml_lossa = self.loss_func(outputa, labela)     
                grads = tf.gradients(maml_lossa, list(fc_weights.values()))
                gradients = dict(zip(fc_weights.keys(), grads))
                fast_fc_weights = dict(zip(fc_weights.keys(), [fc_weights[key] - self.update_lr*gradients[key] for key in fc_weights.keys()]))
          
                for j in range(num_updates - 1):
                    maml_lossa = self.loss_func(self.forward_fc(emb_outputa, fast_fc_weights), labela)
                    grads = tf.gradients(maml_lossa, list(fast_fc_weights.values()))
                    gradients = dict(zip(fast_fc_weights.keys(), grads))
                    fast_fc_weights = dict(zip(fast_fc_weights.keys(), [fast_fc_weights[key] - self.update_lr*gradients[key] for key in fast_fc_weights.keys()]))

                outputb = self.forward_fc(emb_outputb, fast_fc_weights)
                maml_lossb = self.loss_func(outputb, labelb)
                accb = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb), 1), tf.argmax(labelb, 1))

                task_output = [maml_lossb, accb]

                return task_output

            if FLAGS.norm is not 'None':
                unused = task_metalearn((self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0]), False)

            out_dtype = [tf.float32, tf.float32]

            result = tf.map_fn(task_metalearn, elems=(self.inputa, self.inputb, self.labela, self.labelb), dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)
            maml_lossesb, accsb = result

        self.total_loss = total_loss = tf.reduce_sum(maml_lossesb) / tf.to_float(FLAGS.meta_batch_size)
        self.total_accuracy = total_accuracy = tf.reduce_sum(accsb) / tf.to_float(FLAGS.meta_batch_size)

        optimizer = tf.train.AdamOptimizer(self.meta_lr)
        self.metatrain_op = optimizer.minimize(total_loss, var_list=ss_weights.values() + fc_weights.values())

        tf.summary.scalar(prefix+'Loss', total_loss)
        tf.summary.scalar(prefix+'Accuracy', total_accuracy)


    def construct_test_model(self, prefix='metaval_'):

        self.inputa = tf.placeholder(tf.float32)
        self.inputb = tf.placeholder(tf.float32)
        self.labela = tf.placeholder(tf.float32)
        self.labelb = tf.placeholder(tf.float32)

        with tf.variable_scope('meta-test-model', reuse=None) as training_scope:             
            self.ss_weights = ss_weights = self.construct_resnet_ss_weights()
            self.weights = weights = self.construct_resnet_weights()
            self.fc_weights = fc_weights = self.construct_fc_weights()

            num_updates = FLAGS.test_base_epoch_num

            def task_metalearn(inp, reuse=True):
                """ Perform gradient descent for one task in the meta-batch. """
                inputa, inputb, labela, labelb = inp
                accb_list = []

                emb_outputa = self.forward_resnet(inputa, weights, ss_weights, reuse=reuse)
                emb_outputb = self.forward_resnet(inputb, weights, ss_weights, reuse=True)

                outputa = self.forward_fc(emb_outputa, fc_weights)
                maml_lossa = self.loss_func(outputa, labela)     
                grads = tf.gradients(maml_lossa, list(fc_weights.values()))
                gradients = dict(zip(fc_weights.keys(), grads))
                fast_fc_weights = dict(zip(fc_weights.keys(), [fc_weights[key] - self.update_lr*gradients[key] for key in fc_weights.keys()]))
                outputb = self.forward_fc(emb_outputb, fast_fc_weights)
                accb = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb), 1), tf.argmax(labelb, 1))
                accb_list.append(accb)
          
                for j in range(num_updates - 1):
                    maml_lossa = self.loss_func(self.forward_fc(emb_outputa, fast_fc_weights), labela)
                    grads = tf.gradients(maml_lossa, list(fast_fc_weights.values()))
                    gradients = dict(zip(fast_fc_weights.keys(), grads))
                    fast_fc_weights = dict(zip(fast_fc_weights.keys(), [fast_fc_weights[key] - self.update_lr*gradients[key] for key in fast_fc_weights.keys()]))
                    outputb = self.forward_fc(emb_outputb, fast_fc_weights)
                    accb = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb), 1), tf.argmax(labelb, 1))
                    accb_list.append(accb)

                maml_lossb = self.loss_func(outputb, labelb)

                task_output = [maml_lossb, accb, accb_list]

                return task_output

            if FLAGS.norm is not 'None':
                unused = task_metalearn((self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0]), False)

            out_dtype = [tf.float32, tf.float32, [tf.float32]*num_updates]

            result = tf.map_fn(task_metalearn, elems=(self.inputa, self.inputb, self.labela, self.labelb), dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)
            maml_lossesb, accsb, accsb_list = result

        self.metaval_total_loss = total_loss = tf.reduce_sum(maml_lossesb)
        self.metaval_total_accuracy = total_accuracy = tf.reduce_sum(accsb)
        self.metaval_total_accuracies = total_accuracies =[tf.reduce_sum(accsb_list[j]) for j in range(num_updates)]

        tf.summary.scalar(prefix+'Loss', total_loss)
        tf.summary.scalar(prefix+'Accuracy', total_accuracy)



