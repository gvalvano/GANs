from typing import List, Any, Tuple
import tensorflow as tf
from tensorflow import keras as k
from architectures.layers.spectral_norm import SpectralNormalization
from architectures.layers.noise_layers import InstanceNoise

ortho_init = tf.keras.initializers.Orthogonal()


class Discriminator(k.Model):

    def __init__(self, num_filters: List[int],
                 use_spectral_norm=True,
                 use_instance_noise=False,
                 noise_step=None,
                 **kwargs: Any):
        super().__init__()
        assert len(num_filters) >= 1

        if use_instance_noise:
            amplitude = 1/10  # = -20db w.r.t. actual signal (binary mask, at most 1)
            rate = 1/400  # anneal completely in 400 ep.
            self.maybe_add_noise = InstanceNoise(mean=0.0, stddev=amplitude ** 2,
                                                 annealing_rate=rate, step=noise_step)
        else:
            self.maybe_add_noise = k.layers.Lambda(lambda x: x)

        self.blocks = [self.build_spectral_block(f, name=f'block_{i}', **kwargs) for i, f in enumerate(num_filters)] \
            if use_spectral_norm \
            else [self.build_block(f, name=f'block_{i}', **kwargs) for i, f in enumerate(num_filters)]

        self.downsample_ops = [
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='same', name=f'downsample_{i}')
            for i in range(len(num_filters))]

        # output layer with final activation linear
        self.output_layer = self.build_output_layer(**kwargs)

    @staticmethod
    def build_block(num_filters: int, name: str, **kwargs: Any) -> tf.keras.Sequential:
        return k.Sequential([
            k.layers.Conv2D(filters=num_filters, kernel_size=4, padding='same',
                                   kernel_initializer=ortho_init,  **kwargs),
            k.layers.Activation('elu'),
            k.layers.Conv2D(filters=num_filters, kernel_size=4, padding='same',
                                   kernel_initializer=ortho_init,  **kwargs),
            k.layers.Activation('elu'),
        ], name=name)

    @staticmethod
    def build_spectral_block(num_filters: int, name: str, **kwargs: Any) -> tf.keras.Sequential:
        return k.Sequential([
            SpectralNormalization(
                k.layers.Conv2D(filters=num_filters, kernel_size=4, padding='same',
                                kernel_initializer=ortho_init, **kwargs)),
            k.layers.Activation('elu'),
            SpectralNormalization(
                k.layers.Conv2D(filters=num_filters, kernel_size=4, padding='same',
                                kernel_initializer=ortho_init, **kwargs)),
            k.layers.Activation('elu'),
        ], name=name)

    @staticmethod
    def build_output_layer(**kwargs):
        layer_stack = [tf.keras.layers.Conv2D(filters=16, kernel_size=4, padding='same',
                                              kernel_initializer=ortho_init, **kwargs),
                       tf.keras.layers.Activation('elu'),
                       tf.keras.layers.GlobalAveragePooling2D(),
                       tf.keras.layers.Dense(units=1)]
        return tf.keras.Sequential(layer_stack, name='bottleneck')

    def call(self, x: tf.Tensor) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        """
        Call the encoder
        """
        y = self.maybe_add_noise(x)
        for block, downsample in zip(self.blocks, self.downsample_ops):
            # extract features:
            y = block(y)
            # downsample to the next level
            y = downsample(y)
        global_pred = self.output_layer(y)  # output is linear
        return global_pred
