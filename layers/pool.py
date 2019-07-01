""" Pooling layers. """

import numpy as np
import tensorflow as tf

from central import layers_base

class Pooling(layers_base.NNLayer):
    """ Maxpool and Averagepool layers. """

    def __init__(self,
                 input_var,
                 layer_name,
                 kernel_size,
                 strides,
                 padding='SAME',
                 pool='MAX'):           # Either 'MAX' or 'AVG'

        self._input_var = input_var
        self._kernel_size = kernel_size
        self._strides = strides
        self._pad = padding
        self._layer_type = pool

        super(Pooling, self).__init__(
            name=layer_name,
            input_shape=input_var.get_shape().as_list(),
            use_bias=False
        )

    def forward(self):

        with tf.variable_scope(self._name):

            if len(self._inputs_shape) == 4:
                if self._layer_type == 'MAX':
                    pool = tf.nn.max_pool(
                        value=self._input_var,
                        ksize=self._kernel_size,
                        strides=self._strides,
                        padding=self._pad,
                    )
                elif self._layer_type == 'AVG':
                    pool = tf.nn.avg_pool(
                        value=self._input_var,
                        ksize=self._kernel_size,
                        strides=self._strides,
                        padding=self._pad,
                    )

            elif len(self._inputs_shape) == 5:
                if self._layer_type == 'MAX':
                    pool = tf.nn.max_pool3d(
                        input=self._input_var,
                        ksize=self._kernel_size,
                        strides=self._strides,
                        padding=self._pad,
                    )
                elif self._layer_type == 'AVG':
                    pool = tf.nn.avg_pool3d(
                        input=self._input_var,
                        ksize=self._kernel_size,
                        strides=self._strides,
                        padding=self._pad,
                    )

        return pool
