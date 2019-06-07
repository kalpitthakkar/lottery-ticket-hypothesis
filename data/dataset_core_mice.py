""" The dataset for mice behavior recorded at CORE. """

from central import data_base

import tensorflow as tf
import numpy as np

class DataMiceCORE(data_base.DataBase):
    """ Mice behavior (CORE) dataset. """

    def __init__(self,
                 data_type,
                 train_dir,
                 batch_size,
                 test_dir,
                 val_dir=None,
                 random_seed=None,
                 feature_desc=None,
                 preprocessing_fn=None):

        super(DataMiceCORE, self).__init__(
            data_type, train_dir, batch_size, test_dir,
            val_dir, feature_desc, random_seed,
            preprocessing_fn
        )

    def __repr__(self):
        return_str = ("DataMiceCORE(\n"
            "data_type=%s,\n"
            "train_dir=%s,\n"
            "batch_size=%s,\n"
            "test_dir=%s,\n"
            "val_dir=%s,\n"
            "random_seed=%s,\n"
            "feature_desc=%s,\n"
            "preprocessing_fn=%s".format(
                self._data_type,
                self._train_dir,
                self._train_batch_size,
                self._test_dir,
                str(self._val_dir),
                str(self._seed),
                str(self._feature_desc),
                str(self._preprocess)
            )
        )
        return return_str

