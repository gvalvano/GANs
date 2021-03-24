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


def instance_noise(input_tensor, mean=0.0, stddev=1.0, truncated=False, scope="InstanceNoiseLayer"):
    """
    Gradient reversal layer. In forward pass, the input tensor gets through unchanged,
    in the backward pass the gradients are propagated with opposite sign.
    :param input_tensor: (tensor) input tensor
    :param mean: (float) mean for the gaussian noise
    :param stddev: (float) standard deviation for the gaussian noise
    :param truncated: (bool) if the noisy output must be thresholded to be positive and smaller then 1 (range [0, 1])
    :param scope: (str) variable scope for the layer
    :return:
    """

    @tf.custom_gradient
    def _truncate(incoming):
        """ Threshold the noisy output to be in the range [0, 1] """
        def grad(g): return g
        with tf.compat.v1.variable_scope("truncate_op"):
            forward_pass = tf.clip_by_value(incoming, 0.0, 1.0)
        return forward_pass, grad

    with tf.compat.v1.variable_scope(scope):
        # add small noise to the input image:
        noise = tf.random.normal(shape=tf.shape(input=input_tensor), mean=mean, stddev=stddev, dtype=tf.float32)
        forward_pass = input_tensor + noise

    if truncated:
        forward_pass = _truncate(forward_pass)

    return forward_pass


def label_smoothing(input_tensor, is_training, prob=0.1, mode='flip_sign', scope="LabelNoiseLayer"):
    """
    Gradient reversal layer. In forward pass, the input tensor gets through unchanged,
    in the backward pass the gradients are propagated with opposite sign.
    :param input_tensor: (tensor) input tensor
    :param is_training: (bool or tf.placeholder) training mode
    :param prob: (float) probability of perturbing prediction
    :param mode: (str) noise modality. Allowed values in ['flip_sign', 'flip_label'] to flip sign or one-hot label in
                    the prediction (i.e. input_tensor)
    :param scope: (str) variable scope for the layer
    :return:
    """

    assert mode in ['flip_sign', 'flip_label']

    @tf.custom_gradient
    def _flip_sign(incoming):
        """ Threshold the noisy output to be in the range [0, 1] """
        def grad(g): return g
        with tf.compat.v1.variable_scope("flip_sign"):
            c = tf.random.uniform([1], minval=0.0, maxval=1.0, dtype=tf.float32)[0]
            forward_pass = tf.cond(pred=tf.greater(c, prob), true_fn=lambda: incoming, false_fn=lambda: -1.0 * incoming)
        return forward_pass, grad

    @tf.custom_gradient
    def _flip_label(incoming):
        """Threshold the noisy output to be in the range [0, 1] """
        def grad(g): return g
        with tf.compat.v1.variable_scope("flip_label"):
            c = tf.random.uniform([1], minval=0.0, maxval=1.0, dtype=tf.float32)[0]
            forward_pass = tf.cond(pred=tf.greater(c, prob), true_fn=lambda: incoming, false_fn=lambda: tf.abs(incoming - 1.0))
        return forward_pass, grad

    with tf.compat.v1.variable_scope(scope):
        if mode == 'flip_sign':
            forward_pass = _flip_sign(input_tensor)
        elif mode == 'flip_label':
            forward_pass = _flip_label(input_tensor)

    forward_pass = tf.cond(pred=is_training, true_fn=lambda: forward_pass, false_fn=lambda: input_tensor)

    return forward_pass


class InstanceNoise(tf.keras.layers.Layer):
    def __init__(self, mean=0.0, stddev=1.0, annealing_rate=0.0, step=0, truncated=False, scope="InstanceNoiseLayer"):
        """
        Layer for InstanceNoise
        """
        super(InstanceNoise, self).__init__()
        self.mean = mean
        self.stddev = stddev
        self.truncated = truncated
        self.scope = scope
        self.step = step
        self.annealing_rate = annealing_rate
        assert 0.0 <= annealing_rate < 1.0

    def call(self, inputs, **kwargs):
        """ call to the layer
        :param inputs: incoming tensor
        :return:
        """
        i_noise = instance_noise(inputs, mean=self.mean, stddev=self.stddev, truncated=self.truncated, scope=self.scope)
        noise = i_noise - inputs
        alpha = tf.maximum((1.0 - tf.cast(self.step, tf.float32) * self.annealing_rate), 0.0)
        output = inputs * tf.stop_gradient((1 + alpha * noise))
        return output


class LabelSmoothing(tf.keras.layers.Layer):
    def __init__(self, is_training, prob=0.1, mode='flip_sign', scope="LabelNoiseLayer"):
        """
        Layer for LabelSmoothing
        """
        super(LabelSmoothing, self).__init__()
        self.is_training = is_training
        self.prob = prob
        self.mode = mode
        self.scope = scope

    def call(self, inputs, **kwargs):
        """ call to the layer
        :param inputs: incoming tensor
        :return:
        """
        return label_smoothing(inputs, is_training=self.is_training, prob=self.prob, mode=self.mode, scope=self.scope)
