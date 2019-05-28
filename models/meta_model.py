##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## Tianjin University
## Email: liuyaoyao@tju.edu.cn
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

""" Models for meta-learning. """
import numpy as np
import sys
import tensorflow as tf
from tensorflow.python.platform import flags
from utils.misc import mse, softmaxloss, xent, resnet_conv_block, resnet_nob_conv_block, normalize

try:#python2
    from models import Models
except ImportError:#python3
    from models.models import Models

FLAGS = flags.FLAGS

class MetaModel(Models):
    """The class for the meta models. This class is inheritance from Models, so some variables are in the Models class.
    """
    def construct_model(self):
        """The function to construct meta-train model.
        """
        # Set the placeholder for the input episode
        self.inputa = tf.placeholder(tf.float32) # episode train images
        self.inputb = tf.placeholder(tf.float32) # episode test images
        self.labela = tf.placeholder(tf.float32) # episode train labels
        self.labelb = tf.placeholder(tf.float32) # episode test labels

        with tf.variable_scope('meta-model', reuse=None) as training_scope:
            # construct the model weights
            self.ss_weights = ss_weights = self.construct_resnet_ss_weights()
            self.weights = weights = self.construct_resnet_weights()
            self.fc_weights = fc_weights = self.construct_fc_weights()

            # Load base epoch number from FLAGS
            num_updates = FLAGS.train_base_epoch_num

            def task_metalearn(inp, reuse=True):
                """The function to process one episode in a meta-batch.
                Arg:
                  inp: the input episode.
                Returns:
                  A serious outputs like losses and accuracies.
                """
                # Seperate inp to different variables
                inputa, inputb, labela, labelb = inp
                # Generate empty list to record losses
                lossa_list = [] # Base train loss list
                lossb_list = [] # Base test loss list

                # Embed the input images to embeddings with ss weights
                emb_outputa = self.forward_resnet(inputa, weights, ss_weights, reuse=reuse) # Embed episode train 
                emb_outputb = self.forward_resnet(inputb, weights, ss_weights, reuse=True) # Embed episode test 

                # Run the first epoch of the base learning
                # Forward fc layer for episode train 
                outputa = self.forward_fc(emb_outputa, fc_weights)
                # Calculate base train loss
                lossa = self.loss_func(outputa, labela)
                # Record base train loss
                lossa_list.append(lossa)
                # Forward fc layer for episode test
                outputb = self.forward_fc(emb_outputb, fc_weights)
                # Calculate base test loss
                lossb = self.loss_func(outputb, labelb)
                # Record base test loss
                lossb_list.append(lossb) 
                # Calculate the gradients for the fc layer 
                grads = tf.gradients(lossa, list(fc_weights.values()))
                gradients = dict(zip(fc_weights.keys(), grads))
                # Use graient descent to update the fc layer
                fast_fc_weights = dict(zip(fc_weights.keys(), [fc_weights[key] - \
                    self.update_lr*gradients[key] for key in fc_weights.keys()]))
          
                for j in range(num_updates - 1):
                    # Run the following base epochs, these are similar to the first base epoch
                    lossa = self.loss_func(self.forward_fc(emb_outputa, fast_fc_weights), labela)
                    lossa_list.append(lossa)
                    lossb = self.loss_func(self.forward_fc(emb_outputb, fast_fc_weights), labelb)
                    lossb_list.append(lossb) 
                    grads = tf.gradients(lossa, list(fast_fc_weights.values()))
                    gradients = dict(zip(fast_fc_weights.keys(), grads))
                    fast_fc_weights = dict(zip(fast_fc_weights.keys(), [fast_fc_weights[key] - \
                        self.update_lr*gradients[key] for key in fast_fc_weights.keys()]))

                # Calculate final episode test predictions
                outputb = self.forward_fc(emb_outputb, fast_fc_weights)
                # Calculate the final episode test loss, it is the loss for the episode on meta-train 
                final_lossb = self.loss_func(outputb, labelb)
                # Calculate the final episode test accuarcy
                accb = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb), 1), tf.argmax(labelb, 1))

                # Reorganize all the outputs to a list
                task_output = [final_lossb, lossb_list, lossa_list, accb]

                return task_output

            # Initial the batch normalization weights
            if FLAGS.norm is not 'None':
                unused = task_metalearn((self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0]), False)

            # Set the dtype of the outputs
            out_dtype = [tf.float32, [tf.float32]*num_updates, [tf.float32]*num_updates, tf.float32]

            # Run two episodes for a meta batch using parallel setting
            result = tf.map_fn(task_metalearn, elems=(self.inputa, self.inputb, self.labela, self.labelb), \
                dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)
            # Seperate the outputs to different variables
            lossb, lossesb, lossesa, accsb = result

        # Set the variables to output from the tensorflow graph
        self.total_loss = total_loss = tf.reduce_sum(lossb) / tf.to_float(FLAGS.meta_batch_size)
        self.total_accuracy = total_accuracy = tf.reduce_sum(accsb) / tf.to_float(FLAGS.meta_batch_size)
        self.total_lossa = total_lossa = [tf.reduce_sum(lossesa[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
        self.total_lossb = total_lossb = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]

        # Set the meta-train optimizer
        optimizer = tf.train.AdamOptimizer(self.meta_lr)
        self.metatrain_op = optimizer.minimize(total_loss, var_list=list(ss_weights.values()) + list(fc_weights.values()))

        # Set the tensorboard 
        self.training_summaries = []
        self.training_summaries.append(tf.summary.scalar('Meta Train Loss', (total_loss / tf.to_float(FLAGS.metatrain_epite_sample_num))))
        self.training_summaries.append(tf.summary.scalar('Meta Train Accuracy', total_accuracy))
        for j in range(num_updates):
            self.training_summaries.append(tf.summary.scalar('Base Train Loss Step' + str(j+1), total_lossa[j]))
        for j in range(num_updates):
            self.training_summaries.append(tf.summary.scalar('Base Val Loss Step' + str(j+1), total_lossb[j]))

        self.training_summ_op = tf.summary.merge(self.training_summaries)

        self.input_val_loss = tf.placeholder(tf.float32)
        self.input_val_acc = tf.placeholder(tf.float32)
        self.val_summaries = []
        self.val_summaries.append(tf.summary.scalar('Meta Val Loss', self.input_val_loss))
        self.val_summaries.append(tf.summary.scalar('Meta Val Accuracy', self.input_val_acc))
        self.val_summ_op = tf.summary.merge(self.val_summaries)


    def construct_test_model(self):
        """The function to construct meta-test model.
        """
        # Set the placeholder for the input episode
        self.inputa = tf.placeholder(tf.float32)
        self.inputb = tf.placeholder(tf.float32)
        self.labela = tf.placeholder(tf.float32)
        self.labelb = tf.placeholder(tf.float32)

        with tf.variable_scope('meta-test-model', reuse=None) as training_scope: 
            # construct the model weights            
            self.ss_weights = ss_weights = self.construct_resnet_ss_weights()
            self.weights = weights = self.construct_resnet_weights()
            self.fc_weights = fc_weights = self.construct_fc_weights()

            # Load test base epoch number from FLAGS
            num_updates = FLAGS.test_base_epoch_num

            def task_metalearn(inp, reuse=True):
                """The function to process one episode in a meta-batch.
                Arg:
                  inp: the input episode.
                Returns:
                  A serious outputs like losses and accuracies.
                """
                # Seperate inp to different variables
                inputa, inputb, labela, labelb = inp
                # Generate empty list to record accuracies
                accb_list = []

                # Embed the input images to embeddings with ss weights
                emb_outputa = self.forward_resnet(inputa, weights, ss_weights, reuse=reuse)
                emb_outputb = self.forward_resnet(inputb, weights, ss_weights, reuse=True)

                # This part is similar to the meta-train function, you may refer to the comments above
                outputa = self.forward_fc(emb_outputa, fc_weights)
                lossa = self.loss_func(outputa, labela)     
                grads = tf.gradients(lossa, list(fc_weights.values()))
                gradients = dict(zip(fc_weights.keys(), grads))
                fast_fc_weights = dict(zip(fc_weights.keys(), [fc_weights[key] - \
                    self.update_lr*gradients[key] for key in fc_weights.keys()]))
                outputb = self.forward_fc(emb_outputb, fast_fc_weights)
                accb = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb), 1), tf.argmax(labelb, 1))
                accb_list.append(accb)
          
                for j in range(num_updates - 1):
                    lossa = self.loss_func(self.forward_fc(emb_outputa, fast_fc_weights), labela)
                    grads = tf.gradients(lossa, list(fast_fc_weights.values()))
                    gradients = dict(zip(fast_fc_weights.keys(), grads))
                    fast_fc_weights = dict(zip(fast_fc_weights.keys(), [fast_fc_weights[key] - \
                        self.update_lr*gradients[key] for key in fast_fc_weights.keys()]))
                    outputb = self.forward_fc(emb_outputb, fast_fc_weights)
                    accb = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb), 1), tf.argmax(labelb, 1))
                    accb_list.append(accb)

                lossb = self.loss_func(outputb, labelb)

                task_output = [lossb, accb, accb_list]

                return task_output

            if FLAGS.norm is not 'None':
                unused = task_metalearn((self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0]), False)

            out_dtype = [tf.float32, tf.float32, [tf.float32]*num_updates]

            result = tf.map_fn(task_metalearn, elems=(self.inputa, self.inputb, self.labela, self.labelb), \
                dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)
            lossesb, accsb, accsb_list = result

        self.metaval_total_loss = total_loss = tf.reduce_sum(lossesb)
        self.metaval_total_accuracy = total_accuracy = tf.reduce_sum(accsb)
        self.metaval_total_accuracies = total_accuracies =[tf.reduce_sum(accsb_list[j]) for j in range(num_updates)]


