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
"""
Wrapper to the dataset interfaces
"""

# ACDC:
from data_interface.interfaces.acdc.acdc_paired_data_interface import DatasetInterface as ACDCPairedDataInterface
from data_interface.interfaces.acdc.acdc_unpaired_images_interface import \
    DatasetInterface as ACDCUnpairedImagesInterface
from data_interface.interfaces.acdc.acdc_unpaired_masks_interface import DatasetInterface as ACDCUnpairedMasksInterface

# Cityscapes:
from data_interface.interfaces.cityscapes.cs_paired_data_interface import DatasetInterface as CSPairedDataInterface
from data_interface.interfaces.cityscapes.cs_unpaired_images_interface import \
    DatasetInterface as CSUnpairedImagesInterface
from data_interface.interfaces.cityscapes.cs_unpaired_masks_interface import \
    DatasetInterface as CSUnpairedMasksInterface


class DatasetInterfaceWrapper(object):

    def __init__(self, augment, standardize, batch_size, input_size, num_threads, verbose=True):
        """
        Wrapper to the data set interfaces.
        :param augment: (bool) if True, perform data augmentation
        :param standardize: (bool) if True, standardize data as x_new = (x - mean(x))/std(x)
        :param batch_size: (int) batch size
        :param input_size: (int, int) tuple containing (image width, image height)
        :param num_threads: (int) number of parallel threads to run for CPU data pre-processing
        :param verbose: (bool) verbosity level
        """
        # class variables
        self.augment = augment
        self.standardize = standardize
        self.batch_size = batch_size
        self.input_size = input_size
        self.num_threads = num_threads
        self.verbose = verbose

    def get_acdc_paired_data(self, data_path, data_ids, n_classes=4, downsample_factor=1.0, repeat=False):
        """
        wrapper to ACDC data set. Gets input images and annotated masks.
        :param data_path: (str) path to data directory
        :param data_ids: (dict) dictionary with train, validation, test volume ids
        :param n_classes: (int) if 2, then collapse all the object in one channel. Defaults to 4 (left and right
                            ventricle, myocardium and background)
        :param downsample_factor: (float) if greater than 1, downsample input data
        :param repeat: (bool) whether to repeat the input indefinitely
        :return: dataset object
        """
        if self.verbose: print('Define input pipeline for paired data (image, mask)...')

        # initialize data set interfaces
        itf = ACDCPairedDataInterface(root_dir=data_path, data_ids=data_ids, input_size=self.input_size,
                                      verbose=self.verbose, n_classes=n_classes,
                                      downsample_factor=downsample_factor)

        train_data, valid_data, test_data = itf.get_data(
            b_size=self.batch_size,
            augment=self.augment,
            standardize=self.standardize,
            repeat=repeat
        )

        return train_data, valid_data, test_data

    def get_acdc_unpaired_images(self, data_path, data_ids, downsample_factor=1, repeat=False):
        """
        wrapper to ACDC data set. Gets unpaired images.
        :param data_path: (str) path to data directory
        :param data_ids: (dict) dictionary with train, validation, test volume ids
        :param downsample_factor: (float) if greater than 1, downsample input data
        :param repeat: (bool) whether to repeat the input indefinitely
        :return: dataset object
        """
        if self.verbose: print('Define input pipeline for unpaired images...')

        # initialize data set interfaces
        itf = ACDCUnpairedImagesInterface(root_dir=data_path, data_ids=data_ids, input_size=self.input_size,
                                          verbose=self.verbose, downsample_factor=downsample_factor)

        train_data, valid_data = itf.get_data(
            b_size=self.batch_size,
            augment=self.augment,
            standardize=self.standardize,
            repeat=repeat
        )

        return train_data, valid_data

    def get_acdc_unpaired_masks(self, data_path, data_ids, n_classes=4, downsample_factor=1.0, repeat=False):
        """
        wrapper to ACDC data set. Gets unpaired masks.
        :param data_path: (str) path to data directory
        :param data_ids: (dict) dictionary with train, validation, test volume ids
        :param n_classes: (int) if 2, then collapse all the object in one channel. Defaults to 4 (left and right
                            ventricle, myocardium and background)
        :param downsample_factor: (float) if greater than 1, downsample input data
        :param repeat: (bool) whether to repeat the input indefinitely
        :return: dataset object
        """
        if self.verbose: print('Define input pipeline for unpaired masks...')

        # initialize data set interfaces
        itf = ACDCUnpairedMasksInterface(root_dir=data_path, data_ids=data_ids, input_size=self.input_size,
                                         verbose=self.verbose, n_classes=n_classes,
                                         downsample_factor=downsample_factor)

        train_data, valid_data = itf.get_data(
            b_size=self.batch_size,
            augment=self.augment,
            standardize=self.standardize,
            repeat=repeat
        )

        return train_data, valid_data


    def get_cs_paired_data(self, data_path, data_ids, n_classes=20, downsample_factor=1.0, repeat=False):
        """
        wrapper to Cityscapes data set. Gets input images and annotated masks.
        :param data_path: (str) path to data directory
        :param data_ids: (dict) dictionary with train, validation, test volume ids
        :param n_classes: (int) if 2, then collapse all the object in one channel. Defaults to 20
        :param downsample_factor: (float) if greater than 1, downsample input data
        :param repeat: (bool) whether to repeat the input indefinitely
        :return: dataset object
        """
        if self.verbose: print('Define input pipeline for paired data (image, mask)...')

        # initialize data set interfaces
        itf = CSPairedDataInterface(root_dir=data_path, data_ids=data_ids, input_size=self.input_size,
                                    verbose=self.verbose, n_classes=n_classes,
                                    downsample_factor=downsample_factor)

        train_data, valid_data, test_data = itf.get_data(
            b_size=self.batch_size,
            augment=self.augment,
            standardize=self.standardize,
            repeat=repeat
        )

        return train_data, valid_data, test_data

    def get_cs_unpaired_images(self, data_path, data_ids, downsample_factor=1, repeat=False):
        """
        wrapper to Cityscapes data set. Gets unpaired images.
        :param data_path: (str) path to data directory
        :param data_ids: (dict) dictionary with train, validation, test volume ids
        :param downsample_factor: (float) if greater than 1, downsample input data
        :param repeat: (bool) whether to repeat the input indefinitely
        :return: dataset object
        """
        if self.verbose: print('Define input pipeline for unpaired images...')

        # initialize data set interfaces
        itf = CSUnpairedImagesInterface(root_dir=data_path, data_ids=data_ids, input_size=self.input_size,
                                        verbose=self.verbose, downsample_factor=downsample_factor)

        train_data, valid_data = itf.get_data(
            b_size=self.batch_size,
            augment=self.augment,
            standardize=self.standardize,
            repeat=repeat
        )

        return train_data, valid_data

    def get_cs_unpaired_masks(self, data_path, data_ids, n_classes=20, downsample_factor=1.0, repeat=False):
        """
        wrapper to Cityscapes data set. Gets unpaired masks.
        :param data_path: (str) path to data directory
        :param data_ids: (dict) dictionary with train, validation, test volume ids
        :param n_classes: (int) if 2, then collapse all the object in one channel. Defaults to 20
        :param downsample_factor: (float) if greater than 1, downsample input data
        :param repeat: (bool) whether to repeat the input indefinitely
        :return: dataset object
        """
        if self.verbose: print('Define input pipeline for unpaired masks...')

        # initialize data set interfaces
        itf = CSUnpairedMasksInterface(root_dir=data_path, data_ids=data_ids, input_size=self.input_size,
                                       verbose=self.verbose, n_classes=n_classes,
                                       downsample_factor=downsample_factor)

        train_data, valid_data = itf.get_data(
            b_size=self.batch_size,
            augment=self.augment,
            standardize=self.standardize,
            repeat=repeat
        )

        return train_data, valid_data


