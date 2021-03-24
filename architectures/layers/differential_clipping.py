"""
Custom layer that clips inputs during forward pass and copies the gradients during backward pass
"""
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


@tf.custom_gradient
def differential_clipping_layer(incoming, scope="Clipping"):
    """
    Layer for Clipping. In forward pass, it applies a clipping to the incoming tensor,
    in the backward pass is simply copies the gradients from the output to the input.
    :param incoming: (tensor) incoming tensor
    :param scope: (str) variable scope for the layer
    :return:
    """
    def grad(g):
        return g

    with tf.compat.v1.variable_scope(scope):
        forward_pass = tf.clip_by_value(incoming, clip_value_min=0.01, clip_value_max=0.99)

    return forward_pass, grad


class DifferentialClipping(tf.keras.layers.Layer):
    def __init__(self):
        """
        Layer for Clipping. In forward pass, it applies a clipping to the incoming tensor,
        in the backward pass is simply copies the gradients from the output to the input.
        :return: clipped values
        """
        super(DifferentialClipping, self).__init__()

    def call(self, inputs, **kwargs):
        """ call to the layer
        :param inputs: incoming tensor
        :return:
        """
        return differential_clipping_layer(inputs, scope="Clipping")
