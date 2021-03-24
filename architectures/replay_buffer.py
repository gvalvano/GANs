import tensorflow as tf
import numpy as np
import os
from math import pi
from tensorflow_addons_image import transform
from idas.utils import safe_mkdir
from glob import glob


class ReplayBuffer(object):
    """Abstract base class for TF-Agents replay buffer."""

    def __init__(self, data_spec, capacity, patience, prioritized_replay=False, buffer_name='replay_buffer', **kwargs):
        """Initializes the replay buffer.

        Args:
          data_spec: Specs describing a single item that can be stored in this buffer. Must contain at least input_size
          capacity: number of elements that the replay buffer can hold.
          patience: number of epochs to wait before using replay
          buffer_name: name of the pickle file that will contain the buffer
        """
        self.data_spec = data_spec
        self.capacity = capacity
        self.patience = patience
        self.prioritized_replay = prioritized_replay
        if self.prioritized_replay:
            self.prioritized_condition = 'exponential'
            self.prioritized_tau = tf.cast(kwargs['prioritized_tau'], tf.float32)

        self.ds_name = buffer_name
        safe_mkdir(self.ds_name)

        self.buffer_is_empty = True

    def add_batch(self, real_items, fake_items, step='', **kwargs):
        """Adds a batch of items to the replay buffer.
        step can be both an int and a string
        """
        add_condition_true = True
        if self.prioritized_replay:
            replay_step = tf.cast(kwargs['replay_step'], tf.float32)
            add_probability = np.exp(- replay_step/self.prioritized_tau)
            add_condition_true = np.random.uniform(0.0, 1.0) < add_probability

        if self.buffer_is_empty or add_condition_true:
            # add elements to the buffer

            files = glob(self.ds_name + '/*.npy')
            if self.capacity <= len(files):
                # reached maximum capacity --> remove older files:
                f_name = np.random.choice(files)
                os.remove(f_name)

            # add element to the buffer:
            num = len([el for el in files if el.rsplit('/')[-1].startswith(step)])  # counter
            f_name = os.path.join(self.ds_name, '{0}_replay_{1}.npy'.format(step, num))
            np.save(f_name, fake_items.numpy().astype(self.data_spec['dtype']))

            f_name = os.path.join(self.ds_name, '{0}_real_{1}.npy'.format(step, num))
            np.save(f_name, real_items.numpy().astype(self.data_spec['dtype']))

            self.buffer_is_empty = False

    def as_dataset(self, augment=True):
        """Creates and returns a dataset that returns entries from the buffer."""
        parser = lambda name: tf.numpy_function(lambda x: np.load(x).astype(np.float32), [name], [tf.float32])

        from idas.utils import print_yellow_text
        replay_paths = glob(self.ds_name + '/*_replay_*.npy')
        fake_replays = tf.constant(replay_paths)
        fake_replays = tf.data.Dataset.from_tensor_slices(fake_replays)
        fake_replays = fake_replays.shuffle(buffer_size=len(replay_paths), reshuffle_each_iteration=True)
        fake_replays = fake_replays.map(parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        real_paths = glob(self.ds_name + '/*_real_*.npy')
        real_images = tf.constant(real_paths)
        real_images = tf.data.Dataset.from_tensor_slices(real_images)
        real_images = real_images.shuffle(buffer_size=len(real_paths), reshuffle_each_iteration=True)
        real_images = real_images.map(parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        fake_replays = fake_replays.batch(1)
        fake_replays = fake_replays.map(lambda x: x[0], num_parallel_calls=tf.data.experimental.AUTOTUNE)

        real_images = real_images.batch(1)
        real_images = real_images.map(lambda x: x[0], num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if augment:
            fake_replays = fake_replays.map(lambda x: self._augment(x),
                                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
            real_images = real_images.map(lambda x: self._augment(x),
                                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # When model is training prefetch continue prepare data while GPU is busy
        fake_replays = fake_replays.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        real_images = real_images.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return fake_replays, real_images

    def clear(self):
        """ remove the buffer """
        os.remove(self.ds_name)

    def _augment(self, image, interpolation='BILINEAR'):
        """ Data augmentation pipeline: add roto-translations """

        theta = tf.random.uniform((), minval=-pi / 2, maxval=pi / 2)
        tx = tf.random.uniform((),
                               minval=-0.1 * self.data_spec['input_size'][0],
                               maxval=0.1 * self.data_spec['input_size'][0])
        ty = tf.random.uniform((),
                               minval=-0.1 * self.data_spec['input_size'][0],
                               maxval=0.1 * self.data_spec['input_size'][0])
        transf_matrix = [tf.math.cos(theta), -tf.math.sin(theta), tx,
                         tf.math.sin(theta), tf.math.cos(theta), ty,
                         0.0, 0.0]

        image = transform(image, transf_matrix, interpolation=interpolation)
        return image
