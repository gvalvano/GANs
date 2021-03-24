import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from data_interface.utils_cityscapes.split_data import get_splits as get_cs_splits
from experiments_semi.vanilla_gan_base import BaseExperiment
import config as run_config


class Experiment(BaseExperiment):
    def __init__(self, run_id=None, config=None):
        """
        Extends BaseExperiment of LS-GAN
        """
        self.args = run_config.define_flags() if (config is None) else config

        # data specifics
        self.input_size = (128, 256)
        self.n_classes = 20

        # get volume ids for train, validation and test sets:
        self.data_ids = get_cs_splits()[self.args.n_sup_vols][self.args.split_number]

        # data pre-processing
        self.augment = self.args.augment  # perform data augmentation
        self.standardize = self.args.standardize  # perform data standardization

        # run the rest of the init
        super().__init__(run_id, self.args)

    def get_data(self):
        """ Define the dataset iterators for each task (supervised, unsupervised, mask discriminator, future prediction)
        They will be used in define_model().
        """
        if self.args.verbose:
            print('Dataset dir: \033[94m{0}\033[0m\n'.format(self.args.data_path))

        self.train_paired_data, self.valid_paired_data, self.test_paired_data = super(Experiment, self) \
            .get_cs_paired_data(data_path=self.args.data_path, data_ids=self.data_ids, repeat=False,
                                n_classes=self.n_classes)

        self.train_unpaired_images, self.valid_unpaired_images = super(Experiment, self) \
            .get_cs_unpaired_images(data_path=self.args.data_path, data_ids=self.data_ids, repeat=True)

        self.train_unpaired_masks, self.valid_unpaired_masks = super(Experiment, self) \
            .get_cs_unpaired_masks(data_path=self.args.data_path, data_ids=self.data_ids, repeat=True,
                                   n_classes=self.n_classes)


if __name__ == '__main__':
    print('\n' + '-' * 3)
    model = Experiment()
    model.build()
    model.train(n_epochs=2)
