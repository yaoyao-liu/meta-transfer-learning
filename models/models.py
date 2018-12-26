import numpy as np
import sys
import tensorflow as tf
from tensorflow.python.platform import flags
from utils.misc import mse, softmaxloss, xent, resnet_conv_block, resnet_nob_conv_block, normalize

FLAGS = flags.FLAGS

class Models:
    def __init__(self, dim_input=1, dim_output=1):
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.update_lr = FLAGS.update_lr
        self.pretrain_class_num = FLAGS.pretrain_class_num
        self.meta_lr = tf.placeholder_with_default(FLAGS.meta_lr, ())
        self.test_num_updates = FLAGS.test_num_updates
        if FLAGS.metatrain:
            self.test_num_updates = 1
        self.pretrain_lr = tf.placeholder_with_default(FLAGS.pre_lr, ())

        self.loss_func = xent
        self.pretrain_loss_func = softmaxloss

        self.channels = 3
        self.img_size = int(np.sqrt(self.dim_input/self.channels))


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
        if FLAGS.phase=='pre':
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




