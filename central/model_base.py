""" Base class for all models. """

import numpy as np
import tensorflow as tf

class ModelBase(object):
    """ A model class containing the presets and masks required for performing lottery ticket experiments. """

    def __init__(self,
                 presets=None,
                 masks=None,
                 name=None):

        self._masks = masks if masks else {}
        self._presets = presets if presets else {}

        self._train_summaries = None
        self._test_summaries = None
        self._val_summaries = None

        @property
        def train_summaries(self):
            return self._train_summaries

        @property
        def test_summaries(self):
            return self._test_summaries

        @property
        def val_summaries(self):
            return self._val_summaries

        @property
        def masks(self):
            return self._masks

        @property
        def presets(self):
            return self._presets


        def get_current_checkpoint(self, sess):
            return

        def build_model(self, inputs, is_training):
            raise NotImplementedError


        def loss_function(self, label_placeholder, output_logits, **kwargs):
            raise NotImplementedError

        def performance_metric(self, label_placeholder, output_logits, **kwargs):
            raise NotImplementedError
