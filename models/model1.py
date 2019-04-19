#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 20:50:46 2018

@author: ttt
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 20:38:20 2018

@author: ttt
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 12:20:37 2018

@author: a204
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 09:55:51 2018

@author: ttt
"""

#coding=utf-8
import tensorflow as tf
import re
import numpy as np
import globals3d as g_


FLAGS = tf.app.flags.FLAGS
# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', g_.BATCH_SIZE,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_float('learning_rate', g_.INIT_LEARNING_RATE,
                            """Initial learning rate.""")


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
WEIGHT_DECAY_FACTOR = 0.004 / 5. # 3500 -> 2.8

TOWER_NAME = 'tower'
DEFAULT_PADDING = 'SAME'


def _activation_summary(x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.
    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.
    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


def _variable_with_weight_decay(name, shape, wd):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
      name: name of the variable
      shape: list of ints
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
    Returns:
      Variable Tensor
    """
    var = _variable_on_cpu(name, shape,
                           initializer=tf.contrib.layers.xavier_initializer())
    if wd:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def _conv(name, in_ ,ksize, strides=[1,1,1,1], padding=DEFAULT_PADDING, group=1, reuse=False):
    
    n_kern = ksize[3]
    convolve = lambda i, k: tf.nn.conv2d(i, k, strides, padding=padding)

    with tf.variable_scope(name, reuse=reuse) as scope:
        if group == 1:
            kernel = _variable_with_weight_decay('weights', shape=ksize, wd=0.0)
            conv = convolve(in_, kernel)
        else:
            ksize[2] /= group
            kernel = _variable_with_weight_decay('weights', shape=ksize, wd=0.0)
            input_groups = tf.split(in_, group, 3)
            kernel_groups = tf.split(kernel, group, 3)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
#            oncatenate the groups
            conv = tf.concat(output_groups, 3)

        biases = _variable_on_cpu('biases', [n_kern], tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)
        conv = tf.nn.relu(conv, name=scope.name)
        _activation_summary(conv)

    print (name, conv.get_shape().as_list())
    return conv

def _conv3d(name, in_ ,ksize, strides=[1,1,1,1,1], padding=DEFAULT_PADDING, group=1, reuse=False):
    
    n_kern = ksize[4]
    convolve = lambda i, k: tf.nn.conv3d(i, k, strides, padding=padding)

    with tf.variable_scope(name, reuse=reuse) as scope:
        if group == 1:
            kernel = _variable_with_weight_decay('weights', shape=ksize, wd=0.0)
            conv = convolve(in_, kernel)
        else:
            ksize[2] /= group
            kernel = _variable_with_weight_decay('weights', shape=ksize, wd=0.0)
            input_groups = tf.split(in_, group, 3)
            kernel_groups = tf.split(kernel, group, 3)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
#            oncatenate the groups
            conv = tf.concat(output_groups, 3)

        biases = _variable_on_cpu('biases', [n_kern], tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)
        conv = tf.nn.relu(conv, name=scope.name)
        _activation_summary(conv)

    print (name, conv.get_shape().as_list())
    return conv
def _conv3d_s(name, in_ ,ksize, strides=[1,1,1,1,1], padding=DEFAULT_PADDING, group=1, reuse=False):
    
    n_kern = ksize[4]
    convolve = lambda i, k: tf.nn.conv3d(i, k, strides, padding=padding)

    with tf.variable_scope(name, reuse=reuse) as scope:
        if group == 1:
            kernel = _variable_with_weight_decay('weights', shape=ksize, wd=0.0)
            conv = convolve(in_, kernel)
        else:
            ksize[2] /= group
            kernel = _variable_with_weight_decay('weights', shape=ksize, wd=0.0)
            input_groups = tf.split(in_, group, 3)
            kernel_groups = tf.split(kernel, group, 3)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
#            oncatenate the groups
            conv = tf.concat(output_groups, 3)

        biases = _variable_on_cpu('biases', [n_kern], tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)
        conv = tf.nn.sigmoid(conv, name=scope.name)
        _activation_summary(conv)

    print (name, conv.get_shape().as_list())
    return conv

def _maxpool(name, in_, ksize, strides, padding=DEFAULT_PADDING):
    pool = tf.nn.max_pool(in_, ksize=ksize, strides=strides,
                          padding=padding, name=name)

    print (name, pool.get_shape().as_list())
    return pool

def _deconv(name, layer_input, output_shape, ksize, strides=[1,1,1,1,1], padding=DEFAULT_PADDING, reuse=False):
    n_kern = ksize[-2]
    with tf.variable_scope(name, reuse=reuse) as scope:
        kernel = _variable_with_weight_decay('weights', shape=ksize, wd=0.0)
        biases = _variable_on_cpu('biases', [n_kern], tf.constant_initializer(0.0))
        conv = tf.nn.conv3d_transpose(layer_input, kernel, output_shape, strides, padding)
        conv = tf.nn.bias_add(conv, biases)
        conv = tf.nn.relu(conv, name=scope.name)
    return conv

def _fc(name, in_, outsize, dropout=1.0, reuse=False):
    with tf.variable_scope(name, reuse=reuse) as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        
        insize = in_.get_shape().as_list()[-1]
        weights = _variable_with_weight_decay('weights', shape=[insize, outsize], wd=0.004)
        biases = _variable_on_cpu('biases', [outsize], tf.constant_initializer(0.0))
        fc = tf.nn.relu(tf.matmul(in_, weights) + biases, name=scope.name)
        fc = tf.nn.dropout(fc, dropout)

        _activation_summary(fc)

    

    print (name, fc.get_shape().as_list())
    return fc
    


def inference_multiview(views, n_classes, keep_prob):
        """
        views: N x X x Y x Z x V tensor
        """
    
        reuse = False   
        '''conv3d input  [batch, in_depth, in_height, in_width, in_channels]'''
        '''conv3d fillter [filter_depth, filter_height, filter_width, in_channels, out_channels] '''
        size = views.get_shape().as_list()
        rr = size[0]
        Dconv3d_1 = _conv3d('Dconv3d_1', views, [5, 5, 5, 1,32], [1, 1, 1, 1, 1], 'SAME', reuse=reuse)
        Dconv3d_2 = _conv3d('Dconv3d_2', Dconv3d_1, [3, 3, 3, 32,64], [1, 2, 2, 2, 1], 'SAME', reuse=reuse)
        
        Dconv3d_3 = _conv3d('Dconv3d_3', Dconv3d_2, [5, 5, 5, 64,64], [1, 1, 1, 1, 1], 'SAME', reuse=reuse)
        Dconv3d_4 = _conv3d('Dconv3d_4', Dconv3d_3, [3, 3, 3, 64,64], [1, 2, 2, 2, 1], 'SAME', reuse=reuse)
        
        Dconv3d_5 = _conv3d('Dconv3d_5', Dconv3d_4, [5, 5, 5, 64,64], [1, 1, 1, 1, 1], 'SAME', reuse=reuse)
        Dconv3d_6 = _conv3d('Dconv3d_6', Dconv3d_5, [3, 3, 3, 64,64], [1, 2, 2, 2, 1], 'SAME', reuse=reuse)
        
        Dconv3d_7 = _conv3d('Dconv3d_7', Dconv3d_6, [5, 5, 5, 64,64], [1, 1, 1, 1, 1], 'SAME', reuse=reuse)
        Uconv_1 = _deconv('Uconv_1', Dconv3d_7, tf.shape(Dconv3d_5), [3, 3, 3, 64,64], strides=[1,2,2,2,1], padding=DEFAULT_PADDING, reuse=reuse) 
        can1 = tf.concat((Uconv_1,Dconv3d_5),axis = -1)
        Uconv_2 = _deconv('Uconv_2', can1, tf.shape(Dconv3d_3), [3, 3, 3, 64,128], strides=[1,2,2,2,1], padding=DEFAULT_PADDING, reuse=reuse) 
        can2 = tf.concat((Uconv_2,Dconv3d_3),axis = -1)
        Uconv_3 = _deconv('Uconv_3', can2, tf.shape(Dconv3d_1), [3, 3, 3, 32,128], strides=[1,2,2,2,1], padding=DEFAULT_PADDING, reuse=reuse) 
        fc1 = _conv3d('conv3d1x1', Uconv_3, [1, 1, 1, 32,16], [1, 1, 1, 1, 1], 'SAME', reuse=reuse)
        fc2 = _conv3d('conv3d1x11',fc1, [1, 1, 1, 16,2], [1, 1, 1, 1, 1], 'SAME', reuse=reuse)

        return fc2 
    

def load_alexnet_to_mvcnn(sess, caffetf_modelpath):
    """ caffemodel: np.array, """

    caffemodel = np.load(caffetf_modelpath)
    data_dict = caffemodel.item()
    for l in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7']:
        name = l
        _load_param(sess, name, data_dict[l])
    

def _load_param(sess, name, layer_data):
    w, b = layer_data

    with tf.variable_scope(name, reuse=True):
        for subkey, data in zip(('weights', 'biases'), (w, b)):
            print ('loading ', name, subkey)

            try:
                var = tf.get_variable(subkey)
                sess.run(var.assign(data))
            except ValueError as e: 
                print ('varirable loading failed:', subkey, '(%s)' % str(e))


def _view_pool(view_features, name):
    vp = tf.expand_dims(view_features[0], 0) # eg. [100] -> [1, 100]
    for v in view_features[1:]:
        v = tf.expand_dims(v, 0)
        vp = tf.concat([vp, v], 0)
    print ('vp before reducing:', vp.get_shape().as_list())
    vp = tf.reduce_max(vp, [0], name=name)
    return vp 


def loss(fc8, labels):
    l = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=fc8)
    l = tf.reduce_mean(l)
    
    tf.add_to_collection('losses', l)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def classify(fc8):
    softmax = tf.nn.softmax(fc8)
#    y = tf.argmax(softmax, 1)
    return softmax


def _add_loss_summaries(total_loss):
    """Add summaries for losses in CIFAR-10 model.
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
    Args:
    total_loss: Total loss from loss().
    Returns:
    loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    print ('losses:', losses)
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name +' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))
    return loss_averages_op
    

def train(total_loss, global_step, data_size):
    num_batches_per_epoch = data_size / g_.BATCH_SIZE
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    lr = tf.train.exponential_decay(g_.INIT_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)
    
    loss_averages_op = _add_loss_summaries(total_loss)

    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.AdamOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    
    # apply gradients
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    for grad,var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step)

    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variable_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op
