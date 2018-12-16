import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v1
slim = tf.contrib.slim
import pdb


inputs = tf.Variable(tf.zeros([10,224,224,3]))

#with slim.arg_scope(resnet_v1.resnet_arg_scope()):
with tf.variable_scope("ResNet",reuse=False) as vs: 
    net, end_points = resnet_v1.resnet_v1_101(inputs, 1000, is_training=False)
    variables = tf.contrib.framework.get_variables(vs)
    pdb.set_trace()
    test_label = 1 
