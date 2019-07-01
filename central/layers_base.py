""" Base class for any NN layer. """

import numpy as np
import tensorflow as tf

class NNLayer(object):

    def __init__(self,
                 name,
                 input_shape,
                 activation=None,
                 use_bias=True,
                 batch_norm=False,
                 kernel_initializer=None,
                 **kwargs):
        """ Base class that can be extended for any layer, either dense or conv or any other type. """

        self._name = name
        self._inputs_shape = input_shape
        self._activation = activation
        self._use_bias = use_bias
        self._initializer = kernel_initializer
        self._batch_norm = batch_norm
        for k, v in kwargs.items():
            setattr(self, k, v)
