""" 2D and 3D convolutional layers. """

import numpy as np
import tensorflow as tf

from central import layers_base
from utils.layer_utils import cross_replica_batch_norm
from utils.layer_utils import tpu_normalization

class Convolution(layers_base.NNLayer):
    """ Class wrapping TF convolution layers with certain activations and batchnorm. """

    def __init__(self,
                 input_var,
                 layer_name,
                 out_channels,
                 activation=tf.keras.activations.relu,
                 kernel_size=1,
                 stride=1,
                 padding='SAME',
                 spectral_norm_flag=False,
                 update_collection=False,
                 is_training=True,
                 bias=False,
                 use_batch_norm=False,
                 use_cross_replica_batch_norm=False,
                 num_cores=8,
                 initializer=tf.keras.initializers.glorot_normal,
                 weights_init_val=None,
                 weights_mask_units=None):

        self._input_var = input_var
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._pad = padding
        self._spectral_norm = spectral_norm_flag
        self._update_collection = update_collection
        self._is_training = is_training
        self._cross_replica_batch_norm = use_cross_replica_batch_norm
        self._num_cores = num_cores

        self._preset = weights_init_val
        self._mask_weights = weights_mask_units

        super(Convolution, self).__init__(
            name=layer_name,
            input_shape=input_var.get_shape().as_list(),
            activation=activation,
            use_bias=bias,
            batch_norm=use_batch_norm,
            kernel_initializer=initializer
        )

        self._in_channels = self._inputs_shape[-1]

    def forward(self):

        if len(self._inputs_shape) == 4:
            # input_shape: [batch, height, width, in_channels]
            filter_shape = [self._kernel_size, self._kernel_size, self._in_channels, self._out_channels]
            strides = [1, self._stride, self._stride, 1]
            conv_name = 'conv_2d'

        elif len(self._inputs_shape) == 5:
            # input_shape: [batch, depth, height, width, in_channels]
            filter_shape = [self._kernel_size, self._kernel_size, self._kernel_size, self._in_channels, self._out_channels]
            strides = [1, self._stride, self._stride, self._stride, 1]
            conv_name = 'conv_3d'

        with tf.variable_scope(self._name):
            with tf.variable_scope(conv_name):

                if self._preset:
                    self._initializer = tf.constant_initializer(self._preset)

                self._weights = tf.get_variable(
                    name='w',
                    shape=filter_shape,
                    dtype=tf.float32,
                    initializer=self._initializer
                )

                if self._mask_weights:
                    self._mask_initializer = tf.constant_initializer(self._mask_weights)
                    self._mask = tf.get_variable(
                        name='m',
                        shape=filter_shape,
                        initializer=self._mask_initializer,
                        trainable=False
                    )
                    self._weights = tf.multiply(self._weights, self._mask)

                if len(self._inputs_shape) == 4:
                    conv = tf.nn.conv2d(
                        input=self._input_var,
                        filter=self._weights,
                        strides=strides,
                        padding=self._pad,
                        name='2dconv'
                    )
                elif len(self._inputs_shape) == 5:
                    conv = tf.nn.conv3d(
                        input=self._input_var,
                        filter=self._weights,
                        strides=strides,
                        padding=self._pad,
                        name='3dconv'
                    )

            if self._batch_norm:
                if self._cross_replica_batch_norm:
                    conv_bn = tpu_normalization.cross_replica_batch_normalization(
                        inputs=conv,
                        axis=-1,
                        training=self._is_training,
                        name='batch_norm',
                        scale=False,
                        center=True,
                        num_distributed_groups=1
                    )
                else:
                    conv_bn = tf.layers.batch_normalization(
                        inputs=conv,
                        scale=True,
                        center=True,
                        training=self._is_training,
                        name='batch_normalization'
                    )
                output = tf.identity(conv_bn)
            else:
                if self._use_bias:
                    bias = tf.get_variable(
                        name='b',
                        shape=[out_channels],
                        initializer=tf.zeros_initializer()
                    )
                    output = tf.nn.bias_add(
                        conv_bn,
                        bias
                    )
                else:
                    output = tf.identity(conv)

            if self._activation:
                return self._activation(output)

            return output
