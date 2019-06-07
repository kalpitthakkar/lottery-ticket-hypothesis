""" Parent class for every dataset to be used. """

import glob, os

import tensorflow as tf
import numpy as np

"""
DATASET_TYPES = [
    'TFRecord',
    'Numpy',
    'Sequence'
]
"""

class DataSplit(object):

    def __init__(self,
                 dataset,
                 batch_size=None,
                 shuffle=False,
                 shuffle_size=256,
                 seed=None,
                 preprocessing_fn=None):

        self._dataset = dataset

        if not isinstance(preprocessing_fn, list):
            preprocessing_fn = [preprocessing_fn]

        try:
            for pre_fn in preprocessing_fn:
                if not pre_fn:
                    self._dataset.map(pre_fn)
        except e:
            print("Preprocessing function not compatible with dataset API throwing \
                   the error ", e)

        # Shuffle if asked for
        if shuffle:
            self._dataset = self._dataset.shuffle(shuffle_size, seed=seed)

        # Batch the given dataset with specified batch size
        if not batch_size:
            batch_size = 64
        self._dataset = self._dataset.batch(batch_size)

        # Create an iterator for the dataset
        self._iterator = self._dataset.make_initializable_iterator()
        self._initializer = self._iterator.initializer

    @property
    def initializer(self):
        return self._initializer

    @property
    def dataset(self):
        return self._dataset

    def get_handle(self, sess):
        return sess.run(self._iterator.string_handle())


class DataReader(object):
    """ The dataset reader class that reads different datasets
    according to the type of format. """

    def __init__(self,
                 data_type,
                 train_dir,
                 batch_size,
                 test_dir,
                 val_dir,
                 feature_description,
                 random_seed,
                 preprocess_fn):
        if data_type == 'TFRecord':
            def _parse_function(example_proto):
                return tf.parse_single_example(example_proto, feature_description)

            # Create the train dataset
            train_filenames = glob.glob(os.path.join(train_dir, '*'))
            train_paths = tf.data.TFRecordDataset(train_filenames)
            parsed_dataset = train_paths.map(_parse_function)
            self._train = DataSplit(
                parsed_dataset, batch_size=batch_size, shuffle=True,
                seed=random_seed, preprocessing_fn=preprocess_fn
            )

            # Create the test dataset
            test_filenames = glob.glob(os.path.join(test_dir, '*'))
            test_paths = tf.data.TFRecordDataset(test_filenames)
            parsed_dataset = test_paths.map(_parse_function)
            self._test = DataSplit(parsed_dataset)

            # Create the validation dataset if asked for
            if val_dir:
                val_filenames = glob.glob(os.path.join(val_dir, '*'))
                val_paths = tf.data.TFRecordDataset(val_filenames)
                parsed_dataset = val_paths.map(_parse_function)
                self._val = DataSplit(parsed_dataset)
            else:
                self._val = None
        else:
            raise NotImplementedError

    @property
    def train_data(self):
        return self._train.dataset

    @property
    def test_data(self):
        return self._test.dataset

    @property
    def val_data(self):
        if self._val:
            return self._val.dataset
        else:
            return None


class DataBase(object):
    """ The base class for datasets to use. """

    def __init__(self,
                 data_type,
                 train_dir,
                 train_batch_size,
                 test_dir,
                 val_dir=None,
                 feature_desc=None,
                 random_seed_shuffle=None,
                 preprocessing_fn=None):

        self._dtype = data_type
        self._train_dir = train_dir
        self._train_batch_size = train_batch_size
        self._test_dir = test_dir
        self._val_dir = val_dir
        self._feature_desc = feature_desc
        self._seed = random_seed_shuffle
        self._preprocess = preprocessing_fn
        self.create_dataset_instance()

        # Define the iterator handle placeholder -> Can be used
        # for different splits. All have same types and shapes as
        # training data split.
        self._handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(
            self._handle, self._data.train_data.output_types,
            self._data.train_data.output_shapes
        )
        self._placeholders = iterator.get_next()

    def create_dataset_instance(self):
        self._data = DataReader(
            self._dtype, self._train_dir, self._train_batch_size,
            self._test_dir, self._val_dir, self._feature_desc,
            random_seed=self._seed, preprocessing_fn=self._preprocess
        )

    @property
    def train_initializer(self):
        return self._data.train_data.initializer

    @property
    def test_initializer(self):
        return self._data.test_data.initializer

    @property
    def val_initializer(self):
        if self._val_dir:
            return self._data.val_data.initializer
        else:
            return None

    @property
    def handle(self):
        return self._handle

    @property
    def placeholders(self):
        return self._placeholders

    def get_train_handle(self, sess):
        return self._data.train_data.get_handle(sess)

    def get_test_handle(self, sess):
        return self._data.test_data.get_handle(sess)

    def get_val_handle(self, sess):
        if self._val_dir:
            return self._val.get_handle(sess)
        else:
            return None
