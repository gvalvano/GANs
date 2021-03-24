"""
Automatic Cardiac Diagnostic Challenge 2017 database. In total there are images of 100 patients, for which manual
segmentations of the heart cavity, myocardium and right ventricle are provided.
Database at: https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html
Atlas of the heart in each projection at: http://tuttops.altervista.org/ecocardiografia_base.html
"""
#  Copyright 2019 Gabriele Valvano
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import tensorflow as tf
from glob import glob
import numpy as np
import os
from math import pi
from idas.utils import print_yellow_text
# from tensorflow_addons.image import transform
from tensorflow_addons_image import transform

# -----------------------------------------------------------------------------------------------------------------
IMAGE_SIZE = (128, 256)
# root_dir = '/Users/gabrielevalvano/Sviluppo/python_projects/DATA/cityscapes/segmentation/processed'

# default_ids = {
#     'test': ['munster', 'lindau', 'frankfurt'],
#     'validation': ['krefeld', 'zurich', 'bremen'],
#     'train_sup': ['strasbourg', 'weimar', 'aachen', 'tubingen', 'jena', 'bochum', 'darmstadt', 'dusseldorf',
#                   'hamburg', 'cologne', 'monchengladbach', 'ulm', 'hanover', 'stuttgart', 'erfurt']
# }


class DatasetInterface(object):

    def __init__(self, root_dir, data_ids=None, input_size=(1024, 2048), n_classes=20, downsample_factor=1.0,
                 verbose=True):
        """
        Interface to the data set
        :param root_dir: (string) path to directory containing training data
        :param data_ids: (dict) dictionary with train, validation, test volume ids
        :param input_size: (int, int) input size for the neural network. It should be the same across all data sets.
        :param n_classes: (int) if 2, then collapse all the object in one channel. Defaults to 20 (all the classes)
        :param downsample_factor: (float) if greater than 1, downsample input data
        :param verbose: (bool) verbosity level
        """
        root_dir = os.path.join(root_dir, 'segmentation/processed')

        # if data_ids is None:
        #     data_ids = default_ids
        self.n_channels_in = 3
        self.verbose = verbose

        assert downsample_factor >= 1
        assert n_classes in [2, 20]  #  if 2, then collapse all the object into one channel, to have background/foreground
        self.downsample_factor = downsample_factor
        self.n_classes = n_classes
        self.input_size = [dim // self.downsample_factor for dim in input_size]

        # take train and validation samples inside 'train' sub-folder, test with samples under 'validation' sub-folder:
        sub_dir_list = ['train', 'train', 'validation']
        path_dict = dict()
        for d_set, subdir in zip(['train_sup', 'validation', 'test'], sub_dir_list):
            city_list = [d.rsplit('/')[-1] for d in glob(os.path.join(root_dir, subdir) + '/*')]
            filtered_city_list = [el for el in city_list if el in data_ids[d_set]]

            path_prefixes = []
            for city in filtered_city_list:
                path = os.path.join(root_dir, subdir) + '/{0}/{0}*_img.npy'.format(city)
                images = [d.rsplit('_img.npy')[0] for d in glob(path)]
                path_prefixes.extend(images)

            path_dict[d_set] = path_prefixes

        self.x_train_paths = path_dict['train_sup']
        self.x_validation_paths = path_dict['validation']
        self.x_test_paths = path_dict['test']

        assert len(self.x_train_paths) > 0
        assert len(self.x_validation_paths) > 0
        assert len(self.x_test_paths) > 0

    def _data_augmentation_ops(self, x, y):
        """ Data augmentation pipeline (to be applied on training samples)
        """
        # print_yellow_text('Rotation and translation skipped')
        # angles = tf.random.uniform((1, 1), minval=-pi / 2, maxval=pi / 2)
        # x = tfa.image.rotate(x, angles[0], interpolation='BILINEAR')
        # y = tfa.image.rotate(y, angles[0], interpolation='NEAREST')
        #
        # translations = tf.random.uniform((1, 2), minval=-0.1*self.input_size[0], maxval=0.1*self.input_size[0])
        # x = tfa.image.transform(x, translations, interpolation='BILINEAR')
        # y = tfa.image.transform(y, translations, interpolation='NEAREST')

        theta = tf.random.uniform((), minval=-pi / 2, maxval=pi / 2)
        tx = tf.random.uniform((), minval=-0.1 * self.input_size[0], maxval=0.1 * self.input_size[0])
        ty = tf.random.uniform((), minval=-0.1 * self.input_size[0], maxval=0.1 * self.input_size[0])
        transf_matrix = [tf.math.cos(theta), -tf.math.sin(theta), tx,
                         tf.math.sin(theta), tf.math.cos(theta), ty,
                         0.0, 0.0]

        x = transform(x, transf_matrix, interpolation='BILINEAR')
        y = transform(y, transf_matrix, interpolation='NEAREST')

        # image distortions:
        x = tf.image.random_brightness(x, max_delta=0.025)
        x = tf.image.random_contrast(x, lower=0.95, upper=1.05)

        # add noise as regularizer
        std = 0.02  # data are standardized
        noise = tf.random.normal(shape=tf.shape(input=x), mean=0.0, stddev=std)
        x = x + noise

        return x, y

    def _preprocess(self, x, y):
        x = tf.reshape(x, shape=[-1, IMAGE_SIZE[0], IMAGE_SIZE[1], self.n_channels_in])
        y = tf.reshape(y, shape=[-1, IMAGE_SIZE[0], IMAGE_SIZE[1], self.n_classes])
        if self.downsample_factor > 1:  # else keep default
            x = tf.map_fn(lambda _x: tf.image.resize(_x, self.input_size, method='bilinear'), elems=x)
            y = tf.map_fn(lambda _y: tf.image.resize(_y, self.input_size, method='nearest'), elems=y)
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        return x, y

    def data_parser(self, filename, standardize=False):
        """
        Given a subject, returns the sequence of frames for a random z coordinate
        :param filename: (str) path to the patient mri sequence
        :param standardize: (bool) if True, standardize input data
        :return: (array) = numpy array with the frames on the first dimension, s.t.: [None, width, height]
        """
        fname = filename.decode('utf-8') + '_img.npy'
        fname_mask = filename.decode('utf-8') + '_mask.npy'

        batch = np.load(fname).astype(np.float32)
        batch_mask = np.load(fname_mask).astype(np.float32)

        if self.n_classes == 2:  # else keep default
            batch_mask_left = batch_mask[..., :1]
            batch_mask_right = np.sum(batch_mask[..., 1:], axis=-1, keepdims=True)
            batch_mask = np.concatenate((batch_mask_left, batch_mask_right), axis=-1)
            assert np.max(batch_mask) == 1

        if standardize and self.verbose:
            print("Data won't be standardized, as they already have been pre-processed.")

        assert not np.any(np.isnan(batch))
        assert not np.any(np.isnan(batch_mask))

        return batch, batch_mask

    def dataset_pipeline(self, filenames, b_size, augment, standardize, repeat, shuffle_buffer_size=0):
        """ Builds loading and pre-processing pipeline for each dataset.
        :param filenames: dataset object
        :param b_size: batch size
        :param augment: if to perform data augmentation
        :param standardize: if to standardize the input data
        :param repeat: (bool) whether to repeat the input indefinitely
        :param shuffle_buffer_size: (int) buffer size for shuffling (if 0, no shuffle is applied)
        :return: dataset
        """
        dataset = tf.data.Dataset.from_tensor_slices(filenames)

        # read data from different files and parallelize this process
        parser = lambda name: tf.numpy_function(self.data_parser, [name, standardize], [tf.float32, tf.float32])
        dataset = dataset.map(parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(self._preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.cache()

        if augment:
            dataset = dataset.map(lambda x, y: self._data_augmentation_ops(x, y),
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if repeat:
            dataset = dataset.repeat()  # Repeat the input indefinitely
            if self.verbose:
                print_yellow_text(' --> Repeat the input indefinitely  = True', sep=False)

        # un-batch first, then batch the data
        dataset = dataset.unbatch()
        if shuffle_buffer_size > 0:
            dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True)
        dataset = dataset.batch(b_size, drop_remainder=True)

        # When model is training prefetch continue prepare data while GPU is busy
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset

    def get_data(self, b_size, augment=False, standardize=False, repeat=False):
        """ Returns iterators on the dataset.
        :param b_size: batch size
        :param augment: if to perform data augmentation
        :param standardize: if to standardize the input data
        :param repeat: (bool) whether to repeat the input indefinitely
        :return: train_init, valid_init, input_data, label
        """

        train_paths = tf.constant(self.x_train_paths)
        valid_paths = tf.constant(self.x_validation_paths)
        test_paths = tf.constant(self.x_test_paths)

        train_data = self.dataset_pipeline(train_paths, b_size, augment=augment, standardize=standardize,
                                           repeat=repeat, shuffle_buffer_size=300)
        valid_data = self.dataset_pipeline(valid_paths, b_size, augment=False, standardize=standardize,
                                           repeat=False, shuffle_buffer_size=0)
        test_data = self.dataset_pipeline(test_paths, b_size, augment=False, standardize=standardize,
                                          repeat=False, shuffle_buffer_size=0)

        return train_data, valid_data, test_data
