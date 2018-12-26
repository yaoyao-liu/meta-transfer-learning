import numpy as np
import sys
import tensorflow as tf
from models import Models
from tensorflow.python.platform import flags
from utils.misc import mse, softmaxloss, xent, resnet_conv_block, resnet_nob_conv_block, normalize

FLAGS = flags.FLAGS

class PreModel(Models):

    def construct_pretrain_model(self, input_tensors=None, is_val=False):
        if input_tensors is None:
            self.input = tf.placeholder(tf.float32)
            self.label = tf.placeholder(tf.float32)
        else:
            self.input = input_tensors['pretrain_input']
            self.label = input_tensors['pretrain_label']
        with tf.variable_scope('model', reuse=None) as training_scope:

            self.weights = weights = self.construct_resnet_weights()
            self.fc_weights = fc_weights = self.construct_fc_weights()

            if is_val==False:

                self.pretrain_task_output = self.forward_fc(self.forward_pretrain_resnet(self.input, weights, reuse=False), fc_weights)
                self.pretrain_task_loss = self.pretrain_loss_func(self.pretrain_task_output, self.label)
                optimizer = tf.train.AdamOptimizer(self.pretrain_lr)
                self.pretrain_op = optimizer.minimize(self.pretrain_task_loss, var_list=weights.values()+fc_weights.values())
                self.pretrain_task_accuracy = tf.reduce_mean(tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(self.pretrain_task_output), 1), tf.argmax(self.label, 1)))
            else:
                self.pretrain_task_output_val = self.forward_fc(self.forward_pretrain_resnet(self.input, weights, reuse=True), fc_weights)
                self.pretrain_task_accuracy_val = tf.reduce_mean(tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(self.pretrain_task_output_val), 1), tf.argmax(self.label, 1)))

            if is_val==False:
                tf.summary.scalar('pretrain train loss', self.pretrain_task_loss)
                tf.summary.scalar('pretrain train accuracy', self.pretrain_task_accuracy)
            else:
                tf.summary.scalar('pretrain val accuracy', self.pretrain_task_accuracy_val)



