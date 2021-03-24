from typing import List, Tuple
import tensorflow as tf
from tensorflow import keras as k
from architectures.layers.textures_layer import TexturesLayer
from architectures.layers.noise_layers import InstanceNoise
from architectures.layers.distance_transform import ConcatDistanceTransform
from architectures.layers.coord import CoordinateChannel2D
import numpy as np


class DiscInterface(k.Model):

    def __init__(self,
                 input_size: List[int],
                 use_instance_noise=False,
                 **kwargs):

        super().__init__()

        # ---------------------------------------------
        # input interface

        
        if use_instance_noise:
            # amplitude = 1 - 1 / np.sqrt(2)  # --> texture with half the power of the signal, which is a binary pixel
            amplitude = 1/10  # = -20db w.r.t. actual signal (binary mask, at most 1)
            rate = 1/300  # anneal completely in 300 ep.
            self.add_noise = InstanceNoise(mean=0.0, stddev=amplitude ** 2, annealing_rate=rate,
                                           step=kwargs['noise_step'])
        else:
            self.add_noise = k.layers.Lambda(lambda x: x)

        self.processing_steps = [
            self.add_noise
        ]

    def call(self, x: tf.Tensor, training) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        """
        Call the encoder
        """
        y = x
        for step in self.processing_steps:
            y = step(y)
        return y
