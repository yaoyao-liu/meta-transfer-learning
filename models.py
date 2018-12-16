from __future__ import print_function
import numpy as np
import sys
import tensorflow as tf
from tensorflow.python.platform import flags
from utils import mse, softmaxloss, xent, resnet_conv_block, resnet_nob_conv_block, normalize

FLAGS = flags.FLAGS

class MODELS:
    def __init__(self, dim_input=1, dim_output=1):
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.update_lr = FLAGS.update_lr
        self.pretrain_class_num = FLAGS.pretrain_class_num
        self.meta_lr = tf.placeholder_with_default(FLAGS.meta_lr, ())
        self.test_num_updates = FLAGS.test_num_updates
        if FLAGS.train:
            self.test_num_updates = 1
        self.pretrain_lr = tf.placeholder_with_default(FLAGS.pre_lr, ())

        self.loss_func = xent
        self.pretrain_loss_func = softmaxloss

        self.channels = 3
        self.img_size = int(np.sqrt(self.dim_input/self.channels))

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


    def construct_model(self, prefix='metatrain_'):

        self.inputa = tf.placeholder(tf.float32)
        self.inputb = tf.placeholder(tf.float32)
        self.labela = tf.placeholder(tf.float32)
        self.labelb = tf.placeholder(tf.float32)

        with tf.variable_scope('model', reuse=None) as training_scope:

            self.ss_weights = ss_weights = self.construct_resnet_ss_weights()
            self.weights = weights = self.construct_resnet_weights()
            self.fc_weights = fc_weights = self.construct_fc_weights()

            if FLAGS.train:
                num_updates = FLAGS.num_updates
            else:
                num_updates = self.test_num_updates


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

        with tf.variable_scope('model', reuse=None) as training_scope:
            self.ss_weights = ss_weights = self.construct_resnet_ss_weights()
            self.weights = weights = self.construct_resnet_weights()
            self.fc_weights = fc_weights = self.construct_fc_weights()

            if FLAGS.train:
                num_updates = FLAGS.num_updates
            else:
                num_updates = self.test_num_updates

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

        self.metaval_total_loss = total_loss = tf.reduce_sum(maml_lossesb) / tf.to_float(FLAGS.meta_batch_size)
        self.metaval_total_accuracy = total_accuracy = tf.reduce_sum(accsb) / tf.to_float(FLAGS.meta_batch_size)
        self.metaval_total_accuracies = total_accuracies =[tf.reduce_sum(accsb_list[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]

        tf.summary.scalar(prefix+'Loss', total_loss)
        tf.summary.scalar(prefix+'Accuracy', total_accuracy)


    def process_ss_weights(self, weights, ss_weights, label):     
        [dim0, dim1] = weights[label].get_shape().as_list()[0:2]
        this_ss_weights = tf.tile(ss_weights[label], multiples=[dim0, dim1, 1, 1])
        return tf.multiply(weights[label], this_ss_weights)

    def forward_pretrain_resnet(self, inp, weights, reuse=False, scope=''):
        inp = tf.reshape(inp, [-1, self.img_size, self.img_size, self.channels])
        net = self.pretrain_block_forward(inp, weights, 'block1', reuse, scope)
        net = self.pretrain_block_forward(net, weights, 'block2', reuse, scope)
        net = self.pretrain_block_forward(net, weights, 'block3', reuse, scope)
        net = self.pretrain_block_forward(net, weights, 'block4', reuse, scope)
        net = tf.nn.avg_pool(net, [1,5,5,1], [1,5,5,1], 'VALID')
        net = tf.reshape(net, [-1, np.prod([int(dim) for dim in net.get_shape()[1:]])])
        return net

    def forward_resnet(self, inp, weights, ss_weights, reuse=False, scope=''):
        inp = tf.reshape(inp, [-1, self.img_size, self.img_size, self.channels])
        net = self.block_forward(inp, weights, ss_weights, 'block1', reuse, scope)
        net = self.block_forward(net, weights, ss_weights, 'block2', reuse, scope)
        net = self.block_forward(net, weights, ss_weights, 'block3', reuse, scope)
        net = self.block_forward(net, weights, ss_weights, 'block4', reuse, scope)
        net = tf.nn.avg_pool(net, [1,5,5,1], [1,5,5,1], 'VALID')
        net = tf.reshape(net, [-1, np.prod([int(dim) for dim in net.get_shape()[1:]])])
        return net

    def forward_fc(self, inp, fc_weights):
        net = tf.matmul(inp, fc_weights['w5']) + fc_weights['b5']
        return net

    def pretrain_block_forward(self, inp, weights, block, reuse, scope):
        net = resnet_conv_block(inp, weights[block + '_conv1'], weights[block + '_bias1'], reuse, scope+block+'0')
        net = resnet_conv_block(net, weights[block + '_conv2'], weights[block + '_bias2'], reuse, scope+block+'1')
        net = resnet_conv_block(net, weights[block + '_conv3'], weights[block + '_bias3'], reuse, scope+block+'2')
        res = resnet_nob_conv_block(inp, weights[block + '_conv_res'], reuse, scope+block+'res')
        net = net + res
        net = tf.nn.max_pool(net, [1,2,2,1], [1,2,2,1], 'VALID')
        net = tf.nn.dropout(net, keep_prob=FLAGS.pretrain_dropout)
        return net

    def block_forward(self, inp, weights, ss_weights, block, reuse, scope):
        net = resnet_conv_block(inp, self.process_ss_weights(weights, ss_weights, block + '_conv1'), ss_weights[block + '_bias1'], reuse, scope+block+'0')
        net = resnet_conv_block(net, self.process_ss_weights(weights, ss_weights, block + '_conv2'), ss_weights[block + '_bias2'], reuse, scope+block+'1')
        net = resnet_conv_block(net, self.process_ss_weights(weights, ss_weights, block + '_conv3'), ss_weights[block + '_bias3'], reuse, scope+block+'2')
        res = resnet_nob_conv_block(inp, weights[block + '_conv_res'], reuse, scope+block+'res')
        net = net + res
        net = tf.nn.max_pool(net, [1,2,2,1], [1,2,2,1], 'VALID')
        net = tf.nn.dropout(net, keep_prob=1)
        return net

    def construct_fc_weights(self):
        dtype = tf.float32        
        fc_weights = {}
        fc_initializer =  tf.contrib.layers.xavier_initializer(dtype=dtype)
        if FLAGS.pretrain:
            fc_weights['w5'] = tf.get_variable('fc_w5', [512, FLAGS.pretrain_class_num], initializer=fc_initializer)
            fc_weights['b5'] = tf.Variable(tf.zeros([FLAGS.pretrain_class_num]), name='fc_b5')
        else:
            fc_weights['w5'] = tf.get_variable('fc_w5', [512, self.dim_output], initializer=fc_initializer)
            fc_weights['b5'] = tf.Variable(tf.zeros([self.dim_output]), name='fc_b5')
        return fc_weights

    def construct_resnet_weights(self):
        weights = {}
        dtype = tf.float32
        conv_initializer =  tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
        fc_initializer =  tf.contrib.layers.xavier_initializer(dtype=dtype)
        weights = self.construct_residual_block_weights(weights, 3, 3, 64, conv_initializer, dtype, 'block1')
        weights = self.construct_residual_block_weights(weights, 3, 64, 128, conv_initializer, dtype, 'block2')
        weights = self.construct_residual_block_weights(weights, 3, 128, 256, conv_initializer, dtype, 'block3')
        weights = self.construct_residual_block_weights(weights, 3, 256, 512, conv_initializer, dtype, 'block4')
        weights['w5'] = tf.get_variable('w5', [512, FLAGS.pretrain_class_num], initializer=fc_initializer)
        weights['b5'] = tf.Variable(tf.zeros([FLAGS.pretrain_class_num]), name='b5')
        return weights

    def construct_residual_block_weights(self, weights, k, last_dim_hidden, dim_hidden, conv_initializer, dtype, scope='block0'):
        weights[scope + '_conv1'] = tf.get_variable(scope + '_conv1', [k, k, last_dim_hidden, dim_hidden], initializer=conv_initializer, dtype=dtype)
        weights[scope + '_bias1'] = tf.Variable(tf.zeros([dim_hidden]), name=scope + '_bias1')
        weights[scope + '_conv2'] = tf.get_variable(scope + '_conv2', [k, k, dim_hidden, dim_hidden], initializer=conv_initializer, dtype=dtype)
        weights[scope + '_bias2'] = tf.Variable(tf.zeros([dim_hidden]), name=scope + '_bias2')
        weights[scope + '_conv3'] = tf.get_variable(scope + '_conv3', [k, k, dim_hidden, dim_hidden], initializer=conv_initializer, dtype=dtype)
        weights[scope + '_bias3'] = tf.Variable(tf.zeros([dim_hidden]), name=scope + '_bias3')
        weights[scope + '_conv_res'] = tf.get_variable(scope + '_conv_res', [1, 1, last_dim_hidden, dim_hidden], initializer=conv_initializer, dtype=dtype)
        return weights

    def construct_resnet_ss_weights(self):
        ss_weights = {}
        dtype = tf.float32
        conv_initializer =  tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
        fc_initializer =  tf.contrib.layers.xavier_initializer(dtype=dtype)
        ss_weights = self.construct_residual_block_ss_weights(ss_weights, 3, 64, dtype, 'block1')
        ss_weights = self.construct_residual_block_ss_weights(ss_weights, 64, 128, dtype, 'block2')
        ss_weights = self.construct_residual_block_ss_weights(ss_weights, 128, 256, dtype, 'block3')
        ss_weights = self.construct_residual_block_ss_weights(ss_weights, 256, 512, dtype, 'block4')
        return ss_weights

    def construct_residual_block_ss_weights(self, ss_weights, last_dim_hidden, dim_hidden, dtype, scope='block0'):
        ss_weights[scope + '_conv1'] = tf.Variable(tf.ones([1, 1, last_dim_hidden, dim_hidden]), name=scope + '_conv1')
        ss_weights[scope + '_bias1'] = tf.Variable(tf.zeros([dim_hidden]), name=scope + '_bias1')
        ss_weights[scope + '_conv2'] = tf.Variable(tf.ones([1, 1, dim_hidden, dim_hidden]), name=scope + '_conv2')
        ss_weights[scope + '_bias2'] = tf.Variable(tf.zeros([dim_hidden]), name=scope + '_bias2')
        ss_weights[scope + '_conv3'] = tf.Variable(tf.ones([1, 1, dim_hidden, dim_hidden]), name=scope + '_conv3')
        ss_weights[scope + '_bias3'] = tf.Variable(tf.zeros([dim_hidden]), name=scope + '_bias3')
        return ss_weights




