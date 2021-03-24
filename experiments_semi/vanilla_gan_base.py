from abc import abstractmethod
from six.moves import range
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from data_interface.interfaces.dataset_wrapper import DatasetInterfaceWrapper
from architectures.unet import UNet
from architectures.discriminator import Discriminator
from idas.utils import ProgressBar
import config as run_config
from idas.metrics.tf_metrics import dice_coe, iou_coe, hausdorff_distance
from idas import utils
import time
import gan_losses
import utils_tboard as tbu
from idas.utils import print_yellow_text
import utils_sql as usql
import numpy as np
from idas.tf_utils import from_one_hot_to_rgb
from architectures.replay_buffer import ReplayBuffer


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
        self.use_exp_replay = self.args.use_experience_replay

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
        self.replay_buffer_dir = os.path.join(
            self.args.results_dir, 'results/{0}/replay_buffers/'.format(self.args.dataset_name))

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

        self.generator_optimizer = tf.keras.optimizers.Adam(self.args.lr)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(self.args.lr_disc)
        if self.args.lr_disc != self.args.lr_disc:
            print_yellow_text("Using different learning rate for generator (LR = {0}) and discriminator (LR = {1})"
                              .format(self.args.lr, self.args.lr_disc), sep=False)

        gan_types = {'lsgan': gan_losses.LeastSquareGAN,
                     'wgan': gan_losses.WassersteinGAN,
                     'nsatgan': gan_losses.NonSaturatingGAN}
        self.gan = gan_types[self.args.gan]

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
        self.train_unpaired_images = None
        self.valid_unpaired_images = None
        self.train_unpaired_masks = None
        self.valid_unpaired_masks = None
        # model:
        self.segmentor = None
        self.disc_interface = None
        self.discriminator = None
        self.n_disc_levels = None
        self.replay_buffer = None
        # metrics:
        self.w_adv = self.args.adv_weight  # weight for the adversarial contributions
        self.train_dice_loss = None
        self.train_dice = None
        self.train_w_dynamic = None
        self.train_adv_generator_loss = None
        self.train_adv_discriminator_real_loss = None
        self.train_adv_discriminator_fake_loss = None
        self.train_disc_replay_loss = None
        self.train_gen_replay_loss = None
        self.valid_dice_loss = None
        self.valid_dice = None
        self.valid_iou = None
        self.valid_adv_generator_loss = None
        self.valid_adv_discriminator_real_loss = None
        self.valid_adv_discriminator_fake_loss = None
        self.test_dice_loss = None
        self.test_dice = None
        self.test_iou = None
        self.test_adv_generator_loss = None
        self.test_adv_discriminator_loss = None
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
        """ Define the dataset iterators for each task (supervised, unsupervised, mask discriminator, future prediction)
        It must be implemented by the user. Iterators will be used in define_model().
        """
        raise NotImplementedError

    def define_model(self):
        """ Define model to use for the experiment"""
        # -------------
        # define segmentor and discriminator:
        self.segmentor = UNet(output_channels=self.n_classes, num_filters=[32, 64, 128, 256, 512])

        num_filters = [32, 64, 128, 128, 256]
        self.n_disc_levels = len(num_filters)
        self.discriminator = Discriminator(num_filters,
                                           use_spectral_norm=self.args.use_spectral_norm,
                                           use_instance_noise=True, noise_step=self.g_epoch)

    def metrics(self):
        # train
        self.train_dice_loss = tf.keras.metrics.Mean(name='dice_loss')
        self.train_dice = tf.keras.metrics.Mean(name='dice')
        self.train_w_dynamic = tf.keras.metrics.Mean(name='w_dynamic')
        self.train_adv_generator_loss = tf.keras.metrics.Mean(name='adv_generator')
        self.train_adv_discriminator_real_loss = tf.keras.metrics.Mean(name='adv_discriminator_real')
        self.train_adv_discriminator_fake_loss = tf.keras.metrics.Mean(name='adv_discriminator_fake')

        # train experience replay
        self.train_disc_replay_loss = tf.keras.metrics.Mean(name='train_disc_replay_loss')
        self.train_gen_replay_loss = tf.keras.metrics.Mean(name='train_gen_replay_loss')

        # validation
        self.valid_dice_loss = tf.keras.metrics.Mean(name='dice_loss')
        self.valid_dice = tf.keras.metrics.Mean(name='dice')
        self.valid_iou = tf.keras.metrics.Mean(name='iou')
        self.valid_adv_generator_loss = tf.keras.metrics.Mean(name='adv_generator')
        self.valid_adv_discriminator_real_loss = tf.keras.metrics.Mean(name='adv_discriminator_real')
        self.valid_adv_discriminator_fake_loss = tf.keras.metrics.Mean(name='adv_discriminator_fake')

        # Test
        self.test_dice_loss = tf.keras.metrics.Mean(name='dice_loss')
        self.test_dice = tf.keras.metrics.Mean(name='dice')
        self.test_iou = tf.keras.metrics.Mean(name='iou')
        self.test_adv_generator_loss = tf.keras.metrics.Mean(name='adv_generator')
        self.test_adv_discriminator_loss = tf.keras.metrics.Mean(name='adv_discriminator')

        # all
        self.all_metrics = [
            self.train_dice_loss, self.train_dice, self.valid_dice_loss, self.valid_dice,
            self.valid_iou, self.test_dice_loss, self.test_dice, self.test_iou, self.train_w_dynamic,
            self.train_adv_generator_loss, self.train_adv_discriminator_real_loss,
            self.train_adv_discriminator_fake_loss, self.valid_adv_generator_loss,
            self.valid_adv_discriminator_real_loss, self.valid_adv_discriminator_fake_loss,
            self.test_adv_generator_loss, self.test_adv_discriminator_loss,
            self.train_disc_replay_loss, self.train_gen_replay_loss,
        ]

    def reset_metrics(self):
        for metric in self.all_metrics:
            metric.reset_states()

    def add_summaries(self):
        self.train_summaries = {
            'dice_loss': {'type': 'scalar', 'value': self.train_dice_loss},
            'dice': {'type': 'scalar', 'value': self.train_dice},
            'w_dynamic.': {'type': 'scalar', 'value': self.train_w_dynamic},
            'adv_generator': {'type': 'scalar', 'value': self.train_adv_generator_loss},
            'adv_discriminator_real': {'type': 'scalar', 'value': self.train_adv_discriminator_real_loss},
            'adv_discriminator_fake': {'type': 'scalar', 'value': self.train_adv_discriminator_fake_loss},
            # 'train_disc_replay_loss': {'type': 'scalar', 'value': self.train_disc_replay_loss},
            # 'train_gen_replay_loss': {'type': 'scalar', 'value': self.train_gen_replay_loss},
        }
        self.valid_summaries = {
            'dice_loss': {'type': 'scalar', 'value': self.valid_dice_loss},
            'dice': {'type': 'scalar', 'value': self.valid_dice},
            'iou': {'type': 'scalar', 'value': self.valid_iou},
            'adv_generator': {'type': 'scalar', 'value': self.valid_adv_generator_loss},
            'adv_discriminator_real': {'type': 'scalar', 'value': self.valid_adv_discriminator_real_loss},
            'adv_discriminator_fake': {'type': 'scalar', 'value': self.valid_adv_discriminator_fake_loss},
        }
        self.test_summaries = {
            'dice_loss': {'type': 'scalar', 'value': self.test_dice_loss},
            'dice': {'type': 'scalar', 'value': self.test_dice},
            'iou': {'type': 'scalar', 'value': self.test_iou},
            'adv_generator_global': {'type': 'scalar', 'value': self.test_adv_generator_loss},
            'adv_discriminator_global': {'type': 'scalar', 'value': self.test_adv_discriminator_loss},
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

    def wrapper_supervised_loss(self, images, labels, training, verbose=False):
        predictions = self.segmentor(images, training=training)
        soft_predictions = tf.nn.softmax(predictions)
        iou = iou_coe(output=soft_predictions[..., 1:], target=labels[..., 1:], axis=(1, 2))
        dice = dice_coe(output=soft_predictions[..., 1:], target=labels[..., 1:], axis=(1, 2))
        loss = 1.0 - dice

        if verbose:
            return loss, dice, iou, soft_predictions
        return loss, dice, iou

    def wrapper_adversarial_loss(self, unpaired_images, unpaired_masks, training):

        fake_masks = self.segmentor(unpaired_images, training=training)
        soft_fake_masks = tf.nn.softmax(fake_masks)
        pred_fake = self.discriminator(soft_fake_masks, training=training)
        pred_real = self.discriminator(unpaired_masks, training=training)

        # -------------------------
        # Classic adversarial loss:
        gen_loss = self.gan.generator_loss(pred_fake)

        disc_loss_real = self.gan.discriminator_real_loss(pred_real)
        disc_loss_fake = self.gan.discriminator_fake_loss(pred_fake)

        # -------------------------
        # return results:
        return gen_loss, disc_loss_real, disc_loss_fake

    def train_one_epoch(self, writer, step, paired_data, unpaired_images, unpaired_masks):

        dataset = tf.data.Dataset.zip((paired_data, unpaired_images, unpaired_masks)).batch(1)

        # setup progress bar
        # self.progress_bar.attexp_ls_gan_asymmetric.pyach()
        # self.progress_bar.monitor_progress()
        # start_time = time.time()

        t_step = 0
        eps = 1e-12
        training_state = True

        # shape = [self.batch_size, self.input_size[0], self.input_size[1], self.n_classes]
        # old_fake_mask = tf.zeros(shape=shape)
        for (images, labels), unpaired_images, unpaired_masks in dataset:

            images, labels, unpaired_images, unpaired_masks = \
                images[0], labels[0], unpaired_images[0], unpaired_masks[0]

            # self.progress_bar.monitor_progress()
            with tf.GradientTape() as sup_tape, \
                    tf.GradientTape() as gen_tape, \
                    tf.GradientTape() as disc_tape:

                # ------------
                # labelled data (supervised cost):
                sup_loss, dice, iou = self.wrapper_supervised_loss(images, labels, training=training_state)
                self.train_dice_loss.update_state(sup_loss)
                self.train_dice.update_state(dice)

                # ------------
                # unpaired data (adversarial cost):
                gen_loss, disc_loss_real, disc_loss_fake = \
                    self.wrapper_adversarial_loss(unpaired_images, unpaired_masks, training=training_state)

                disc_loss = disc_loss_real + disc_loss_fake
                self.train_adv_generator_loss.update_state(gen_loss)
                self.train_adv_discriminator_real_loss.update_state(disc_loss)
                self.train_adv_discriminator_fake_loss.update_state(disc_loss)
                # self.train_adv_generator_local_loss.update_state(gen_local_loss)

                # compute adversarial costs
                gen_loss = gen_loss
                disc_loss = disc_loss

                # define a dynamic weight for the losses
                w_dynamic = self.w_adv * tf.abs(tf.stop_gradient(sup_loss / (gen_loss + eps)))
                self.train_w_dynamic.update_state(w_dynamic)
                gen_loss = w_dynamic * gen_loss
                disc_loss = w_dynamic * disc_loss

                t_step += 1

            gradients = sup_tape.gradient(sup_loss, self.segmentor.trainable_variables)
            self.generator_optimizer.apply_gradients(zip(gradients, self.segmentor.trainable_variables))

            gradients = gen_tape.gradient(gen_loss, self.segmentor.trainable_variables)
            self.generator_optimizer.apply_gradients(zip(gradients, self.segmentor.trainable_variables))

            # if disc_loss > gen_loss:  # this condition is to train discriminator and generator together
            gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
            self.discriminator_optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables))

        self.write_summaries('train', writer, step=step)
        writer.flush()

        # detach progress bar and update last time of arrival:
        # self.progress_bar.detach()
        # self.progress_bar.update_lta(time.time() - start_time)
        return step + 1

    # @tf.function
    def validate(self, writer, step, paired_data, unpaired_images, unpaired_masks):

        dataset = tf.data.Dataset.zip((paired_data, unpaired_images, unpaired_masks)).batch(1)
        log_id, log_max = 0, 8
        segmentation_logs = [[], [], []]
        v_step = 0
        training_state = False
        for (images, labels), unpaired_images, unpaired_masks in dataset:

            images, labels, unpaired_images, unpaired_masks = \
                images[0], labels[0], unpaired_images[0], unpaired_masks[0]

            # ------------
            # labelled data (supervised cost):
            sup_loss, dice, iou, predicted_mask = \
                self.wrapper_supervised_loss(images, labels, training=training_state, verbose=True)
            self.valid_dice_loss.update_state(sup_loss)
            self.valid_dice.update_state(dice)
            self.valid_iou.update_state(iou)

            # ------------
            # unpaired data (adversarial cost):
            gen_loss, disc_loss_real, disc_loss_fake = \
                self.wrapper_adversarial_loss(unpaired_images, unpaired_masks, training=training_state)

            disc_loss = disc_loss_real + disc_loss_fake
            self.valid_adv_generator_loss.update_state(gen_loss)
            self.valid_adv_discriminator_real_loss.update_state(disc_loss)
            self.valid_adv_discriminator_fake_loss.update_state(disc_loss)

            # add images to tensorboard log
            if log_id < log_max:
                log_id += 1

                # -----------------
                # supervised segmentation
                segmentation_logs[0].append(255 * images[0])
                segmentation_logs[1].append(from_one_hot_to_rgb(predicted_mask)[0, ...])
                segmentation_logs[2].append(from_one_hot_to_rgb(labels)[0, ...])

                if log_id == 1:
                    # add histograms to tensorboard logs:
                    block0_l0_weights = self.discriminator.blocks[0].layers[0].trainable_weights[0]
                    block0_l1_weights = self.discriminator.blocks[0].layers[2].trainable_weights[0]
                    block0_l0_act = self.discriminator.blocks[0].layers[1](unpaired_masks)
                    block0_l1_act = self.discriminator.blocks[0].layers[3](block0_l0_act)

                    valid_dict = {'block0_l0': block0_l0_weights, 'block0_l0_act': block0_l0_act,
                                  'block0_l1': block0_l1_weights, 'block0_l1_act': block0_l1_act}

                    self.add_histogram_summary('validation', writer, valid_dict, step=step)

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
            d_ckp_dir = os.path.join(ckp_dir, 'discriminator')
            self.segmentor.load_weights(os.path.join(s_ckp_dir, 'checkpoint'))
            self.discriminator.load_weights(os.path.join(d_ckp_dir, 'checkpoint'))
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
        d_ckp_dir = os.path.join(ckp_dir, 'discriminator')
        utils.safe_mkdir(ckp_dir)
        utils.safe_mkdir(s_ckp_dir)
        utils.safe_mkdir(d_ckp_dir)
        self.segmentor.save_weights(os.path.join(s_ckp_dir, 'checkpoint'))
        self.discriminator.save_weights(os.path.join(d_ckp_dir, 'checkpoint'))

    def maybe_save_best_model(self, epoch):
        if self.valid_dice.result() > self.best_dice_ever:
            print('New best model... saving weights')
            self.save_model(ckp_dir=self.checkpoint_dir)
            self.best_dice_ever = self.valid_dice.result()
            self.best_epoch_ever = epoch

    def initialize_replay_buffer(self, capacity, prioritized, tau):
        utils.safe_mkdir(self.replay_buffer_dir)
        self.replay_buffer = ReplayBuffer(data_spec={'input_size': self.input_size, 'dtype': np.float16},
                                          patience=20, capacity=capacity,
                                          prioritized_replay=prioritized, prioritized_tau=tau,
                                          buffer_name=os.path.join(self.replay_buffer_dir, self.args.RUN_ID))

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

        # initialise replay buffer:
        if self.use_exp_replay:
            self.initialize_replay_buffer(capacity=2 * n_epochs, prioritized=True, tau=n_epochs/2)

        template = '\033[31m{0}\033[0m |  Loss: {1:.4f}  |  Accuracy: {2:.4f}  |  Took: {3:.3f} seconds'
        for epoch in range(n_epochs):
            ep_str = str(epoch + 1) if (trained_epochs == 0) else '({0}+) '.format(trained_epochs) + str(epoch + 1)
            print('_' * 40 + '\n\033[1;33mEPOCH {0}\033[0m - \033[94m{1}\033[0m : '.format(ep_str, self.run_id))

            # Reset the metrics at the start of the next epoch
            self.reset_metrics()

            start_time = time.time()
            t_step = self.train_one_epoch(writer, t_step,
                                          paired_data=self.train_paired_data,
                                          unpaired_images=self.train_unpaired_images,
                                          unpaired_masks=self.train_unpaired_masks)
            print(template.format('TRAIN:      ', self.train_dice_loss.result(), self.train_dice.result(),
                                  time.time() - start_time))

            if epoch >= self.valid_offset:
                start_time = time.time()
                v_step = self.validate(writer, v_step,
                                       paired_data=self.valid_paired_data,
                                       unpaired_images=self.valid_unpaired_images,
                                       unpaired_masks=self.valid_unpaired_masks)
                print(template.format('VALIDATION: ', self.valid_dice_loss.result(), self.valid_dice.result(),
                                      time.time() - start_time))

                # if self.maybe_apply_early_stopping(): break
                self.maybe_save_best_model(epoch)

            # increase global epoch counter
            self.g_epoch = self.g_epoch + 1

        # save last model
        self.save_model(ckp_dir=self.checkpoint_dir + '/last_model')

        # load best model and do a test:
        self.load_pre_trained_model(self.checkpoint_dir, verbose=True)
        self.test(writer, paired_data=self.test_paired_data)
