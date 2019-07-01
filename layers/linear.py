""" A fully-connected layer. """

import numpy as np
import tensorflow as tf

from central import layers_base

class Linear(layers_base.NNLayer):
    """ Class for initialization of a fully-connected layer. """

    def __init__(self,
                 input_var,
                 layer_name,
                 output_units,
                 activation=tf.keras.activations.relu,
                 initializer=tf.keras.initializers.glorot_normal,
                 batch_norm=False,
                 update_collection=False,
                 is_training=True,
                 bias=False,
                 weight_init_val=None,
                 weight_mask_units=None):

        self._input_var = input_var
        self._output_units = output_units
        self._update_collection = update_collection
        self._is_training = is_training

        self._preset = weight_init_val
        self._mask_weights = weight_mask_units

        super(Linear, self).__init__(
            name=layer_name,
            input_shape=input_var.get_shape().as_list,
            activation=activation,
            use_bias=bias,
            batch_norm=batch_norm,
            kernel_initializer=initializer
        )

    def forward(self):
        with tf.variable_scope(self._name):
            # Check if the weight for this layer needs to be set with predefined value
            if self._preset:
                self._initializer = tf.constant_initializer(self._preset)

            self._weights = tf.get_variable(
                name='w',
                shape=[self._inputs_shape, self._output_units],
                dtype=tf.float32,
                initializer=self._initializer
            )

            # Check which weight units are to be removed / kept
            if self._mask_weights:
                self._mask_initializer = tf.constant_initializer(self._mask_weights)
                self._mask = tf.get_variable(
                    name='m',
                    shape=[self._inputs_shape, self._output_units],
                    initializer=self._mask_initializer,
                    trainable=False
                )
                self._weights = tf.multiply(self._weights, self._mask)

            output = tf.matmul(self._input_var, self._weights)

            if self._use_bias:
                try:
                    assert self._batch_norm is False
                    self._bias = tf.get_variable(
                        name='b',
                        shape=[self._output_units],
                        initializer=tf.zeros_initializer()
                    )
                    output += bias
                except AssertionError as e:
                    print("{} : should not use batchnorm with bias in Linear layer".format(e))

            elif self._batch_norm:
                raise NotImplementedError

            if self._activation:
                return self._activation(output)

            return output
