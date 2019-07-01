""" Inception I3D model. """

import numpy as np
import tensorflow as tf

from central import model_base
from layers import Convolution, Linear, Pooling

class InceptionI3D(model_base.ModelBase):
    """ Class for the inflated Inception-Resnet-v2 used for action recognition. """

    def __init__(self,
                 endpoints_dict,
                 presets=None,
                 masks=None,
                 name='Inception-Inflated-3D',
                 final_endpoint='Logits',
                 batch_norm=False,
                 cross_replica_batch_norm=False,
                 num_classes=101,
                 spatial_squeeze=True,
                 num_cores=8,
                 dropout_keep_prob=1.0):

        self._endpoints = endpoints_dict
        self._final_endpoint = final_endpoint
        self._batch_norm = batch_norm
        self._cross_replica_batch_norm = cross_replica_batch_norm
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._num_cores = num_cores
        self._keep = dropout_keep_prob

        super(InceptionI3D, self).__init__(
            presets=presets,
            masks=masks,
            name=name
        )

    def build_model(self, inputs, is_training):

        if self._final_endpoint not in self._endpoints:
            raise ValueError("Unknown final endpoint {}".format(self._final_endpoint))

	net = inputs
        end_points = {}
        layers = {}
        idx = 0

        def inception_block(net, is_training, end_point, layers_config, layers, idx):
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    # 1x1x1 Conv, stride 1
                    layers[idx] = Convolution(net, 'Conv3d_0a_1x1', layers_config['Branch_0'][0],
                        kernel_size=1, stride=1, is_training=is_training, num_cores=self._num_cores,
                        use_batch_norm=self._batch_norm, use_cross_replica_batch_norm=self._cross_replica_batch_norm)
                    branch_0 = layers[idx].forward(); idx += 1

                with tf.variable_scope('Branch_1'):
                    # 1x1x1 Conv, stride 1
                    layers[idx] = Convolution(net, 'Conv3d_0a_1x1', layers_config['Branch_1'][0],
                        kernel_size=1, stride=1, is_training=is_training, num_cores=self._num_cores,
                        use_batch_norm=self._batch_norm, use_cross_replica_batch_norm=self._cross_replica_batch_norm)
                    branch_1 = layers[idx].forward(); idx += 1
                    # 3x3x3 Conv, stride 1
                    layers[idx] = Convolution(branch_1, 'Conv3d_0b_3x3', layers_config['Branch_1'][1],
                        kernel_size=3, stride=1, is_training=is_training, num_cores=self._num_cores,
                        use_batch_norm=self._batch_norm, use_cross_replica_batch_norm=self._cross_replica_batch_norm)
                    branch_1 = layers[idx].forward(); idx += 1

                with tf.variable_scope('Branch_2'):
                    # 1x1x1 Conv, stride 1
                    layers[idx] = Convolution(net, 'Conv3d_0a_1x1', layers_config['Branch_2'][0],
                        kernel_size=1, stride=1, is_training=is_training, num_cores=self._num_cores,
                        use_batch_norm=self._batch_norm, use_cross_replica_batch_norm=self._cross_replica_batch_norm)
                    branch_2 = layers[idx].forward(); idx += 1
                    # 3x3x3 Conv, stride 1
                    layers[idx] = Convolution(branch_2, 'Conv3d_0b_3x3', layers_config['Branch_2'][1],
                        kernel_size=3, stride=1, is_training=is_training, num_cores=self._num_cores,
                        use_batch_norm=self._batch_norm, use_cross_replica_batch_norm=self._cross_replica_batch_norm)
                    branch_2 = layers[idx].forward(); idx += 1

                with tf.variable_scope('Branch_3'):
                    # 3x3x3 Max-pool, stride 1, 1, 1
                    layers[idx] = Pooling(net, 'MaxPool3d_0a_3x3',
                        kernel_size=layers_config['Branch_3'][0], strides=layers_config['Branch_3'][1],
                        padding='SAME', pool='MAX')
                    branch_3 = layers[idx].forward(); idx += 1
                    # 1x1x1 Conv, stride 1
                    layers[idx] = Convolution(branch_3, 'Conv3d_0b_1x1', layers_config['Branch_3'][2],
                        kernel_size=1, stride=1, is_training=is_training, num_cores=self._num_cores,
                        use_batch_norm=self._batch_norm, use_cross_replica_batch_norm=self._cross_replica_batch_norm)
                    branch_3 = layers[idx].forward(); idx += 1

                # Concat branch_[0-3]
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)

            return net, idx

        print('Inputs: {}'.format(net.get_shape().as_list()))

        with tf.variable_scope('RGB'):
            with tf.variable_scope('inception_i3d'):
                # 7x7x7 Conv, stride 2
                end_point = 'Conv3d_1a_7x7'
                layers[idx] = Convolution(net, end_point, 64,
                    kernel_size=7, stride=2, is_training=is_training, num_cores=self._num_cores,
                    use_batch_norm=self._batch_norm, use_cross_replica_batch_norm=self._cross_replica_batch_norm)
                net = layers[idx].forward(); idx += 1
                get_shape = net.get_shape().as_list()
                print('{} : {}'.format(end_point, get_shape))

                end_points[end_point] = net
                if self._final_endpoint == end_point: return net, end_points, layers

                # 1x3x3 Max-pool, stride 1, 2, 2
                end_point = 'MaxPool3d_2a_3x3'
                layers[idx] = Pooling(net, end_point, kernel_size=[1, 1, 3, 3, 1],
                    strides=[1, 1, 2, 2, 1], padding='SAME', pool='MAX')
                net = layers[idx].forward(); idx += 1
                get_shape = net.get_shape().as_list()
                print('{} : {}'.format(end_point, get_shape))

                end_points[end_point] = net
                if self._final_endpoint == end_point: return net, end_points, layers

                # 1x1x1 Conv, stride 1
                end_point = 'Conv3d_2b_1x1'
                layers[idx] = Convolution(net, end_point, 64,
                    kernel_size=1, stride=1, is_training=is_training, num_cores=self._num_cores,
                    use_batch_norm=self._batch_norm, use_cross_replica_batch_norm=self._cross_replica_batch_norm)
                net = layers[idx].forward(); idx += 1
                get_shape = net.get_shape().as_list()
                print('{} : {}'.format(end_point, get_shape))

                end_points[end_point] = net
                if self._final_endpoint == end_point: return net, end_points, layers

                # 3x3x3 Conv, stride 1
                end_point = 'Conv3d_2c_3x3'
                layers[idx] = Convolution(net, end_point, 192,
                    kernel_size=3, stride=1, is_training=is_training, num_cores=self._num_cores,
                    use_batch_norm=self._batch_norm, use_cross_replica_batch_norm=self._cross_replica_batch_norm)
                net = layers[idx].forward(); idx += 1
                get_shape = net.get_shape().as_list()
                print('{} : {}'.format(end_point, get_shape))

                end_points[end_point] = net
                if self._final_endpoint == end_point: return net, end_points, layers

                # 1x3x3 Max-pool, stride 1, 2, 2
                end_point = 'MaxPool3d_3a_3x3'
                layers[idx] = Pooling(net, end_point, kernel_size=[1, 1, 3, 3, 1],
                    strides=[1, 1, 2, 2, 1], padding='SAME', pool='MAX')
                net = layers[idx].forward(); idx += 1
                get_shape = net.get_shape().as_list()
                print('{} : {}'.format(end_point, get_shape))

                end_points[end_point] = net
                if self._final_endpoint == end_point: return net, end_points, layers

                # Mixed 3b : Inception block
                end_point = 'Mixed_3b'
                layers_config = {
                    'Branch_0': [64],
                    'Branch_1': [96, 128],
                    'Branch_2': [16, 32],
                    'Branch_3': [[1, 3, 3, 3, 1], [1, 1, 1, 1, 1], 32]
                }
                net, idx = inception_block(net, is_training, end_point, layers_config, layers, idx)
                get_shape = net.get_shape().as_list()
                print('{} : {}'.format(end_point, get_shape))

                end_points[end_point] = net
                if self._final_endpoint == end_point: return net, end_points, layers

                # Mixed 3c: Inception block
                end_point = 'Mixed_3c'
                layers_config = {
                    'Branch_0': [128],
                    'Branch_1': [128, 192],
                    'Branch_2': [32, 96],
                    'Branch_3': [[1, 3, 3, 3, 1], [1, 1, 1, 1, 1], 64]
                }
                net, idx = inception_block(net, is_training, end_point, layers_config, layers, idx)
                get_shape = net.get_shape().as_list()
                print('{} : {}'.format(end_point, get_shape))

                end_points[end_point] = net
                if self._final_endpoint == end_point: return net, end_points, layers

                # 3x3x3 Max-pool, stride 2, 2, 2
                end_point = 'MaxPool3d_4a_3x3'
                layers[idx] = Pooling(net, end_point, kernel_size=[1, 3, 3, 3, 1],
                    strides=[1, 2, 2, 2, 1], padding='SAME', pool='MAX')
                net = layers[idx].forward(); idx += 1
                get_shape = net.get_shape().as_list()
                print('{} : {}'.format(end_point, get_shape))

                end_points[end_point] = net
                if self._final_endpoint == end_point: return net, end_points, layers

                # Mixed 4b: Inception block
                end_point = 'Mixed_4b'
                layers_config = {
                    'Branch_0': [192],
                    'Branch_1': [96, 208],
                    'Branch_2': [16, 48],
                    'Branch_3': [[1, 3, 3, 3, 1], [1, 1, 1, 1, 1], 64]
                }
                net, idx = inception_block(net, is_training, end_point, layers_config, layers, idx)
                get_shape = net.get_shape().as_list()
                print('{} : {}'.format(end_point, get_shape))

                end_points[end_point] = net
                if self._final_endpoint == end_point: return net, end_points, layers

                # Mixed 4c: Inception block
                end_point = 'Mixed_4c'
                layers_config = {
                    'Branch_0': [160],
                    'Branch_1': [112, 224],
                    'Branch_2': [24, 64],
                    'Branch_3': [[1, 3, 3, 3, 1], [1, 1, 1, 1, 1], 64]
                }
                net, idx = inception_block(net, is_training, end_point, layers_config, layers, idx)
                get_shape = net.get_shape().as_list()
                print('{} : {}'.format(end_point, get_shape))

                end_points[end_point] = net
                if self._final_endpoint == end_point: return net, end_points, layers

                # Mixed 4d: Inception block
                end_point = 'Mixed_4d'
                layers_config = {
                    'Branch_0': [128],
                    'Branch_1': [128, 256],
                    'Branch_2': [24, 64],
                    'Branch_3': [[1, 3, 3, 3, 1], [1, 1, 1, 1, 1], 64]
                }
                net, idx = inception_block(net, is_training, end_point, layers_config, layers, idx)
                get_shape = net.get_shape().as_list()
                print('{} : {}'.format(end_point, get_shape))

                end_points[end_point] = net
                if self._final_endpoint == end_point: return net, end_points, layers

                # Mixed 4e: Inception block
                end_point = 'Mixed_4e'
                layers_config = {
                    'Branch_0': [112],
                    'Branch_1': [144, 288],
                    'Branch_2': [32, 64],
                    'Branch_3': [[1, 3, 3, 3, 1], [1, 1, 1, 1, 1], 64]
                }
                net, idx = inception_block(net, is_training, end_point, layers_config, layers, idx)
                get_shape = net.get_shape().as_list()
                print('{} : {}'.format(end_point, get_shape))

                end_points[end_point] = net
                if self._final_endpoint == end_point: return net, end_points, layers

                # Mixed 4f: Inception block
                end_point = 'Mixed_4f'
                layers_config = {
                    'Branch_0': [256],
                    'Branch_1': [160, 320],
                    'Branch_2': [32, 128],
                    'Branch_3': [[1, 3, 3, 3, 1], [1, 1, 1, 1, 1], 128]
                }
                net, idx = inception_block(net, is_training, end_point, layers_config, layers, idx)
                get_shape = net.get_shape().as_list()
                print('{} : {}'.format(end_point, get_shape))

                end_points[end_point] = net
                if self._final_endpoint == end_point: return net, end_points, layers

                # 2x2x2 Max-pool, stride 2x2x2
                end_point = 'MaxPool3d_5a_2x2'
                layers[idx] = Pooling(net, end_point, kernel_size=[1, 2, 2, 2, 1],
                    strides=[1, 2, 2, 2, 1], padding='SAME', pool='MAX')
                net = layers[idx].forward(); idx += 1
                get_shape = net.get_shape().as_list()
                print('{} : {}'.format(end_point, get_shape))

                end_points[end_point] = net
                if self._final_endpoint == end_point: return net, end_points, layers

                # Mixed 5b: Inception block
                end_point = 'Mixed_5b'
                layers_config = {
                    'Branch_0': [256],
                    'Branch_1': [160, 320],
                    'Branch_2': [32, 128],
                    'Branch_3': [[1, 3, 3, 3, 1], [1, 1, 1, 1, 1], 128]
                }
                net, idx = inception_block(net, is_training, end_point, layers_config, layers, idx)
                get_shape = net.get_shape().as_list()
                print('{} : {}'.format(end_point, get_shape))

                end_points[end_point] = net
                if self._final_endpoint == end_point: return net, end_points, layers

                # Mixed 5c: Inception block
                end_point = 'Mixed_5c'
                layers_config = {
                    'Branch_0': [384],
                    'Branch_1': [192, 384],
                    'Branch_2': [48, 128],
                    'Branch_3': [[1, 3, 3, 3, 1], [1, 1, 1, 1, 1], 128]
                }
                net, idx = inception_block(net, is_training, end_point, layers_config, layers, idx)
                get_shape = net.get_shape().as_list()
                print('{} : {}'.format(end_point, get_shape))

                end_points[end_point] = net
                if self._final_endpoint == end_point: return net, end_points, layers

                # Logits
                end_point = 'Logits'
                with tf.variable_scope(end_point):
                    # 2x7x7 Average-pool, stride 1, 1, 1
                    layers[idx] = Pooling(net, 'Logits', kernel_size=[1, 2, 7, 7, 1],
                        strides=[1, 1, 1, 1, 1], padding='VALID', pool='AVG')
                    net = layers[idx].forward(); idx += 1
                    get_shape = net.get_shape().as_list()
                    print('{} / Average-pool3D: {}'.format(end_point, get_shape))
                    end_points[end_point + '_average_pool3d'] = net

                    # Dropout
                    net = tf.nn.dropout(net, self._keep)

                    # 1x1x1 Conv, stride 1
                    layers[idx] = Convolution(net, 'Conv3d_0c_1x1', self._num_classes,
                        kernel_size=1, stride=1, activation=None,
                        use_batch_norm=self._batch_norm, use_cross_replica_batch_norm=self._cross_replica_batch_norm,
                        is_training=is_training, num_cores=self._num_cores)
                    logits = layers[idx].forward(); idx += 1
                    get_shape = logits.get_shape().as_list()
                    print('{} / Conv3d_0c_1x1 : {}'.format(end_point, get_shape))

                    if self._spatial_squeeze:
                        # Removes dimensions of size 1 from the shape of a tensor
                        # Specify which dimensions have to be removed: 2 and 3
                        logits = tf.squeeze(logits, [2, 3], name='SpatialSqueeze')
                        get_shape = logits.get_shape().as_list()
                        print('{} / Spatial Squeeze : {}'.format(end_point, get_shape))

                averaged_logits = tf.reduce_mean(logits, axis=1)
                get_shape = averaged_logits.get_shape().as_list()
                print('{} / Averaged Logits : {}'.format(end_point, get_shape))

                end_points[end_point] = averaged_logits
                if self._final_endpoint == end_point: return averaged_logits, end_points, layers

                # Predictions
                end_point = 'Predictions'
                predictions = tf.nn.softmax(
                    averaged_logits)
                end_points[end_point] = predictions
                return predictions, end_points, layers

        def loss_function(self, label_placeholder, output_logits, **kwargs):
            self._loss = tf.losses.softmax_cross_entropy(
                logits=output_logits,
                onehot_labels=one_hot_labels,
                label_smoothing=kwargs['label_smoothing']
            )

            return self._loss

        def performance_metric(self, label_placeholder, output_logits, **kwargs):
            predictions = tf.argmax(output_logits, axis=1)
            ground_truth = tf.argmax(label_placeholder, axis=1)
            correct = tf.equal(predictions, ground_truth)
            self._perf = tf.reduce_mean(tf.cast(correct, tf.float32))

            return self._perf
