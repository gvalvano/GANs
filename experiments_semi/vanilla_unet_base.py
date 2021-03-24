from abc import abstractmethod
from six.moves import range
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from data_interface.interfaces.dataset_wrapper import DatasetInterfaceWrapper
from architectures.unet import UNet
from idas.utils import ProgressBar
import config as run_config
from idas.metrics.tf_metrics import dice_coe, iou_coe, hausdorff_distance
from idas import utils
import time
import utils_tboard as tbu
from idas.utils import print_yellow_text
import utils_sql as usql
import numpy as np
from idas.tf_utils import from_one_hot_to_rgb


class BaseExperiment(DatasetInterfaceWrapper):
    def __init__(self, run_id=None, config=None):
        """
        :param run_id: (str) used when we want to load a specific pre-trained model. Default run_id is taken from
                config_file.py
        :param config: args object with configuration
        """

        self.args = run_config.define_flags() if (config is None) else config

        self.run_id = self.args.RUN_ID if (run_id is None) else run_id
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.CUDA_VISIBLE_DEVICE)
        self.verbose = self.args.verbose

        if self.verbose:
            print('\n----------')
            print('CUDA_VISIBLE_DEVICE: \033[94m{0}\033[0m\n'.format(str(self.args.CUDA_VISIBLE_DEVICE)))
            print('RUN_ID = \033[94m{0}\033[0m'.format(self.run_id))

        self.num_threads = self.args.num_threads

        # -----------------------------
        # Model hyper-parameters:
        self.lr = tf.Variable(self.args.lr, dtype=tf.float32, trainable=False, name='learning_rate')
        self.batch_size = self.args.b_size

        # -----------------------------
        # Report

        # path to save checkpoints and graph
        self.last_checkpoint_dir = \
            os.path.join(self.args.results_dir,
                         'results/{0}/checkpoints/'.format(self.args.dataset_name) + self.args.RUN_ID)
        self.checkpoint_dir = \
            os.path.join(self.args.results_dir,
                         'results/{0}/checkpoints/'.format(self.args.dataset_name) + self.args.RUN_ID)
        utils.safe_mkdir(self.checkpoint_dir)
        self.log_dir = os.path.join(
            self.args.results_dir, 'results/{0}/graphs/'.format(self.args.dataset_name) + self.args.RUN_ID + '/convnet')

        # verbosity
        self.skip_step = self.args.skip_step  # frequency of batch report
        self.train_summaries_skip = self.args.train_summaries_skip  # number of skips before writing train summaries
        self.tensorboard_on = self.args.tensorboard_on  # (bool) if you want to save tensorboard logs
        self.tensorboard_verbose = self.args.tensorboard_verbose if self.tensorboard_on else False  # (bool) save also
        #                                                                          # layers weights at the end of epoch
        # # epoch offset for first test:
        # offset = {'perc100': 180, 'perc25': 100, 'perc12p5': 30, 'perc10': 10, 'perc6': 10, 'perc3': 10}
        # self.ep_offset = offset[self.args.n_sup_vols]

        # -----------------------------
        # Other settings

        # Define global step for training e validation and counter for global epoch:
        self.g_epoch = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_epoch')

        # number of epochs before saving a model
        self.valid_offset = 20

        # training or test mode (needed for the behaviour of dropout, BN, ecc.)
        self.is_training = tf.Variable(True, dtype=tf.bool, trainable=False, name='is_training')
        self.switch_to_train_mode = lambda: self.is_training.assign(True)
        self.switch_to_test_mode = lambda: self.is_training.assign(False)

        # lr decay
        # self.decay_lr = self.lr.assign(tf.multiply(self.lr, 1.0), name='decay_lr')
        # self.update_lr = self.lr.assign(
        #     cyclic_learning_rate(self.g_epoch, step_size=20,
        #                          learning_rate=args.lr // 10, max_lr=args.lr,
        #                          mode='triangular', gamma=.997), name='update_lr')
        self.update_lr = self.lr.assign(self.args.lr, name='update_lr')

        self.optimizer = tf.keras.optimizers.Adam(self.args.lr)

        # -----------------------------
        # initialize wrapper to the data set
        super().__init__(augment=self.augment,
                         standardize=self.standardize,
                         batch_size=self.batch_size,
                         input_size=self.input_size,
                         num_threads=self.num_threads,
                         verbose=self.args.verbose)

        # -----------------------------
        # initialize placeholders for the class
        # data pipeline placeholders:
        self.global_seed = tf.Variable(initial_value=0, dtype=tf.int64, shape=(), trainable=False)
        self.train_paired_data = None
        self.valid_paired_data = None
        self.test_paired_data = None
        # model:
        self.segmentor = None
        # metrics:
        self.w_adv = self.args.adv_weight  # weight for the adversarial contributions
        self.train_dice_loss = None
        self.train_dice = None
        self.valid_dice_loss = None
        self.valid_dice = None
        self.valid_iou = None
        self.test_dice_loss = None
        self.test_dice = None
        self.test_iou = None
        self.all_metrics = None
        self.best_dice_ever = 0.0
        self.best_epoch_ever = 0
        # summaries:
        self.train_summaries = None
        self.valid_summaries = None
        self.test_summaries = None

        # -----------------------------
        # progress bar
        self.progress_bar = ProgressBar(update_delay=20)

    def build(self):
        """ Build the computation graph """
        if self.verbose:
            print('Building the computation graph...')
        self.get_data()
        self.define_model()
        self.metrics()
        self.add_summaries()

    @abstractmethod
    def get_data(self):
        """ Define the dataset iterators for each task.
        It must be implemented by the user. Iterators will be used in define_model().
        """
        raise NotImplementedError

    def define_model(self):
        """ Define model to use for the experiment"""
        # -------------
        # define segmentor:
        self.segmentor = UNet(output_channels=self.n_classes, num_filters=[32, 64, 128, 256, 512])

    def metrics(self):
        # train
        self.train_dice_loss = tf.keras.metrics.Mean(name='dice_loss')
        self.train_dice = tf.keras.metrics.Mean(name='dice')

        # validation
        self.valid_dice_loss = tf.keras.metrics.Mean(name='dice_loss')
        self.valid_dice = tf.keras.metrics.Mean(name='dice')
        self.valid_iou = tf.keras.metrics.Mean(name='iou')

        # Test
        self.test_dice_loss = tf.keras.metrics.Mean(name='dice_loss')
        self.test_dice = tf.keras.metrics.Mean(name='dice')
        self.test_iou = tf.keras.metrics.Mean(name='iou')

        # all
        self.all_metrics = [
            self.train_dice_loss, self.train_dice, self.valid_dice_loss, self.valid_dice,
            self.valid_iou, self.test_dice_loss, self.test_dice, self.test_iou
        ]

    def reset_metrics(self):
        for metric in self.all_metrics:
            metric.reset_states()

    def add_summaries(self):
        self.train_summaries = {
            'dice_loss': {'type': 'scalar', 'value': self.train_dice_loss},
            'dice': {'type': 'scalar', 'value': self.train_dice},
        }
        self.valid_summaries = {
            'dice_loss': {'type': 'scalar', 'value': self.valid_dice_loss},
            'dice': {'type': 'scalar', 'value': self.valid_dice},
            'iou': {'type': 'scalar', 'value': self.valid_iou},
        }
        self.test_summaries = {
            'dice_loss': {'type': 'scalar', 'value': self.test_dice_loss},
            'dice': {'type': 'scalar', 'value': self.test_dice},
            'iou': {'type': 'scalar', 'value': self.test_iou},
        }

    def write_summaries(self, stype, writer, step):

        assert stype in ['train', 'validation', 'test']
        summary_dict = {
            'train': self.train_summaries,
            'validation': self.valid_summaries,
            'test': self.test_summaries}

        for name, content in zip(summary_dict[stype].keys(), summary_dict[stype].values()):
            if name is not None:
                with writer.as_default():
                    if content['type'] == 'scalar':
                        tf.summary.scalar("{0}/{1}".format(stype, name), content['value'].result(), step=step)
                    elif content['type'] == 'histogram':
                        tf.summary.histogram("{0}/{1}".format(stype, name), content['value'].result(), step=step)
                    elif content['type'] == 'image':
                        tf.summary.image("{0}/{1}".format(stype, name), content['value'], step=step,
                                         max_outputs=3, description=None)
                    else:
                        raise TypeError('Unsupported type for Validation summary')

    @staticmethod
    def add_image_summary(stype, writer, name_value_dict, step):
        for name, value in zip(name_value_dict.keys(), name_value_dict.values()):
            if value is not None:
                with writer.as_default():
                    tf.summary.image("{0}/{1}".format(stype, name), value, step=step, max_outputs=3, description=None)

    @staticmethod
    def add_histogram_summary(stype, writer, name_value_dict, step):
        for name, value in zip(name_value_dict.keys(), name_value_dict.values()):
            if value is not None:
                with writer.as_default():
                    tf.summary.histogram("{0}/{1}".format(stype, name), value, step=step)

    # @staticmethod
    # def add_binary_noise(mask, pred):
    #
    #     return tf.stop_gradient(mask), tf.stop_gradient(expected_pred)

    def wrapper_supervised_loss(self, images, labels, training, verbose=False):
        predictions = self.segmentor(images, training=training)
        soft_predictions = tf.nn.softmax(predictions)
        iou = iou_coe(output=soft_predictions[..., 1:], target=labels[..., 1:], axis=(1, 2))
        dice = dice_coe(output=soft_predictions[..., 1:], target=labels[..., 1:], axis=(1, 2))
        loss = 1.0 - dice

        if verbose:
            return loss, dice, iou, soft_predictions
        return loss, dice, iou

    def train_one_epoch(self, writer, step, paired_data):

        t_step = 0
        training_state = True

        # shape = [self.batch_size, self.input_size[0], self.input_size[1], self.n_classes]
        # old_fake_mask = tf.zeros(shape=shape)
        for images, labels in paired_data:

            # self.progress_bar.monitor_progress()
            with tf.GradientTape() as sup_tape:

                # ------------
                # labelled data (supervised cost):
                sup_loss, dice, iou = self.wrapper_supervised_loss(images, labels, training=training_state)
                self.train_dice_loss.update_state(sup_loss)
                self.train_dice.update_state(dice)

                t_step += 1

            gradients = sup_tape.gradient(sup_loss, self.segmentor.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.segmentor.trainable_variables))

        self.write_summaries('train', writer, step=step)
        writer.flush()

        # detach progress bar and update last time of arrival:
        # self.progress_bar.detach()
        # self.progress_bar.update_lta(time.time() - start_time)
        return step + 1

    # @tf.function
    def validate(self, writer, step, paired_data):

        log_id, log_max = 0, 8
        segmentation_logs = [[], [], []]
        v_step = 0
        training_state = False
        for images, labels in paired_data:

            # ------------
            # labelled data (supervised cost):
            sup_loss, dice, iou, predicted_mask = \
                self.wrapper_supervised_loss(images, labels, training=training_state, verbose=True)
            self.valid_dice_loss.update_state(sup_loss)
            self.valid_dice.update_state(dice)
            self.valid_iou.update_state(iou)

            # add images to tensorboard log
            if log_id < log_max:
                log_id += 1

                # -----------------
                # supervised segmentation
                segmentation_logs[0].append(255 * images[0])
                segmentation_logs[1].append(from_one_hot_to_rgb(predicted_mask)[0, ...])
                segmentation_logs[2].append(from_one_hot_to_rgb(labels)[0, ...])

                v_step += 1

        with writer.as_default():
            tf.summary.image("validation/input_pred_gt",
                             tbu.plot_custom_image_grid(segmentation_logs, title='Input - Predicted - GT',
                                                        max_rows=log_max), step=step)
        self.write_summaries('validation', writer, step=step)
        writer.flush()
        return step + 1

    # @tf.function
    def test(self, writer, paired_data, step=0):

        print_yellow_text('Testing the model...')

        training_state = False
        log_id, log_max = 0, 15
        segmentation_logs = [[], [], []]

        # ----------------
        # initialize a dictionary with the metrics
        metrics = {'dice': dict(), 'iou': dict(), 'hausdorff_distance': dict()}
        metrics['dice']['global'] = list()
        metrics['iou']['global'] = list()
        for ch in range(self.n_classes):
            metrics['dice'][ch] = list()
            metrics['iou'][ch] = list()
            metrics['hausdorff_distance'][ch] = list()

        for images, labels in paired_data:

            # ------------
            # labelled data (supervised cost):
            sup_loss, dice, iou, predicted_mask = \
                self.wrapper_supervised_loss(images, labels, training=training_state, verbose=True)
            self.test_dice_loss.update_state(sup_loss)
            self.test_dice.update_state(dice)
            self.test_iou.update_state(iou)

            metrics['dice']['global'].append(dice.numpy())
            metrics['iou']['global'].append(iou.numpy())
            for ch in range(self.n_classes):
                dice_ch = dice_coe(output=tf.expand_dims(predicted_mask[..., ch], axis=-1),
                                   target=tf.expand_dims(labels[..., ch], axis=-1))
                iou_ch = iou_coe(output=tf.expand_dims(predicted_mask[..., ch], axis=-1),
                                 target=tf.expand_dims(labels[..., ch], axis=-1))
                hd_ch = hausdorff_distance(predicted_mask[..., ch], labels[..., ch])
                metrics['dice'][ch].append(dice_ch.numpy())
                metrics['iou'][ch].append(iou_ch.numpy())
                metrics['hausdorff_distance'][ch].append(hd_ch)

            # add images to tensorboard log
            if log_id < log_max:
                log_id += 1

                # -----------------
                # supervised segmentation
                segmentation_logs[0].append(255 * images[0])
                segmentation_logs[1].append(from_one_hot_to_rgb(predicted_mask)[0, ...])
                segmentation_logs[2].append(from_one_hot_to_rgb(labels)[0, ...])

        self.write_summaries('test', writer, step=step)

        with writer.as_default():
            tf.summary.image("test/input_pred_gt",
                             tbu.plot_custom_image_grid(segmentation_logs, title='Input - Predicted - GT',
                                                        max_rows=log_max), step=step)

        writer.flush()
        # -----------------------------------
        # report to the database:
        db_name = self.args.sql_db_name
        print_yellow_text('Adding results to database with name: {0}'.format(db_name), sep=False)
        db_entries = {
            'run_id': self.run_id, 'n_sup_vols': self.args.n_sup_vols, 'split_number': self.args.split_number,
            'config': self.args,
            'experiment_type': self.args.experiment_type, 'dataset_name': self.args.dataset_name,
            'input_size': self.input_size, 'epoch': self.best_epoch_ever,
            'dice_list': metrics['dice']['global'],
            'iou_list': metrics['iou']['global'],
            'avg_dice': np.mean(np.array(metrics['dice']['global'])),
            'avg_iou': np.mean(np.array(metrics['iou']['global'])),
            'std_dice': np.std(np.array(metrics['dice']['global'])),
            'std_iou': np.std(np.array(metrics['iou']['global'])),
            'dice_list_per_class': [np.array(metrics['dice'][ch]) for ch in range(self.n_classes)],
            'iou_list_per_class': [np.array(metrics['iou'][ch]) for ch in range(self.n_classes)],
            'avg_dice_per_class': [np.mean(np.array(metrics['dice'][ch])) for ch in range(self.n_classes)],
            'std_dice_per_class': [np.std(np.array(metrics['dice'][ch])) for ch in range(self.n_classes)],
            'avg_iou_per_class': [np.mean(np.array(metrics['iou'][ch])) for ch in range(self.n_classes)],
            'std_iou_per_class': [np.std(np.array(metrics['iou'][ch])) for ch in range(self.n_classes)],
            'avg_hd_per_class': [np.mean(np.array(metrics['hausdorff_distance'][ch])) for ch in range(self.n_classes)],
            'std_hd_per_class': [np.std(np.array(metrics['hausdorff_distance'][ch])) for ch in range(self.n_classes)]}
        usql.add_db_entry(entries=db_entries, table_name=self.args.table_name, database=db_name)
        print_yellow_text('Done.\n', sep=False)

        return True

    def load_pre_trained_model(self, ckp_dir, verbose=True):
        try:
            s_ckp_dir = os.path.join(ckp_dir, 'segmentor')
            self.segmentor.load_weights(os.path.join(s_ckp_dir, 'checkpoint'))
            if verbose:
                col_start = utils.Colors.OKGREEN
                col_end = utils.Colors.ENDC
                print('{0}Loading pre-trained model from: \033[94m{2}\033[0m{1}'.format(col_start, col_end, ckp_dir))
        except ValueError:
            if verbose:
                col_start = utils.Colors.FAIL
                col_end = utils.Colors.ENDC
                print('{0}No pre-trained model available: start training from scratch{1}'.format(col_start, col_end))

    def save_model(self, ckp_dir):
        s_ckp_dir = os.path.join(ckp_dir, 'segmentor')
        utils.safe_mkdir(ckp_dir)
        utils.safe_mkdir(s_ckp_dir)
        self.segmentor.save_weights(os.path.join(s_ckp_dir, 'checkpoint'))

    def maybe_save_best_model(self, epoch):
        if self.valid_dice.result() > self.best_dice_ever:
            print('New best model... saving weights')
            self.save_model(ckp_dir=self.checkpoint_dir)
            self.best_dice_ever = self.valid_dice.result()
            self.best_epoch_ever = epoch

    def train(self, n_epochs):
        """ The train function alternates between training one epoch and evaluating """
        if self.verbose:
            print("\nStarting network training... Number of epochs to train: \033[94m{0}\033[0m".format(n_epochs))
            print("Tensorboard verbose mode: \033[94m{0}\033[0m".format(self.tensorboard_verbose))
            print("Tensorboard dir: \033[94m{0}\033[0m".format(self.log_dir))

        # with tf.profiler.experimental.Profile('logdir'):

        # load pre-trained model, if available
        self.load_pre_trained_model(self.checkpoint_dir)

        writer = tf.summary.create_file_writer(self.log_dir)

        t_step, v_step = tf.cast(0, tf.int64), tf.cast(0, tf.int64)
        trained_epochs = 0
        template = '\033[31m{0}\033[0m |  Loss: {1:.4f}  |  Accuracy: {2:.4f}  |  Took: {3:.3f} seconds'
        for epoch in range(n_epochs):
            ep_str = str(epoch + 1) if (trained_epochs == 0) else '({0}+) '.format(trained_epochs) + str(epoch + 1)
            print('_' * 40 + '\n\033[1;33mEPOCH {0}\033[0m - \033[94m{1}\033[0m : '.format(ep_str, self.run_id))

            # Reset the metrics at the start of the next epoch
            self.reset_metrics()

            start_time = time.time()
            t_step = self.train_one_epoch(writer, t_step,
                                          paired_data=self.train_paired_data)
            print(template.format('TRAIN:      ', self.train_dice_loss.result(), self.train_dice.result(),
                                  time.time() - start_time))

            if epoch > 50:
                start_time = time.time()
                v_step = self.validate(writer, v_step,
                                       paired_data=self.valid_paired_data)
                print(template.format('VALIDATION: ', self.valid_dice_loss.result(), self.valid_dice.result(),
                                      time.time() - start_time))

            # if self.maybe_apply_early_stopping(): break
            if epoch > self.valid_offset:
                self.maybe_save_best_model(epoch)

            # increase global epoch counter
            self.g_epoch = self.g_epoch + 1

        # save last model
        self.save_model(ckp_dir=self.checkpoint_dir + '/last_model')

        # load best model and do a test:
        self.load_pre_trained_model(self.checkpoint_dir, verbose=True)
        self.test(writer, paired_data=self.test_paired_data)
