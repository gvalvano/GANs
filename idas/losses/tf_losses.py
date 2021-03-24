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
from idas.metrics.tf_metrics import dice_coe, generalized_dice_coe, shannon_binary_entropy
from idas.models.hyperbolic.hyp_ops import tf_poinc_dist_sq


def dice_loss(output, target, axis=(1, 2, 3), smooth=1e-12):
    """ Returns Soft Sørensen–Dice loss """
    return 1.0 - dice_coe(output, target, axis=axis, smooth=smooth)


def generalized_dice_loss(output, target, axis=(1, 2, 3), smooth=1e-12):
    """ Returns the Generalized Soft Sørensen–Dice loss """
    return 1.0 - generalized_dice_coe(output, target, axis=axis, smooth=smooth)


def weighted_softmax_cross_entropy(y_pred, y_true, num_classes, eps=1e-12):
    """
    Define weighted cross-entropy function for classification tasks. Applies softmax on y_pred.
    :param y_pred: tensor [None, width, height, n_classes]
    :param y_true: tensor [None, width, height, n_classes]
    :param eps: (float) small value to avoid division by zero
    :param num_classes: (int) number of classes
    :return:
    """

    n = [tf.reduce_sum(input_tensor=tf.cast(y_true[..., c], tf.float32)) for c in range(num_classes)]
    n_tot = tf.reduce_sum(input_tensor=n)

    weights = [n_tot / (n[c] + eps) for c in range(num_classes)]

    y_pred = tf.reshape(y_pred, (-1, num_classes))
    y_true = tf.cast(tf.reshape(y_true, (-1, num_classes)), dtype=tf.float32)
    softmax = tf.nn.softmax(y_pred)

    w_cross_entropy = -tf.reduce_sum(input_tensor=tf.multiply(y_true * tf.math.log(softmax + eps), weights), axis=[1])
    loss = tf.reduce_mean(input_tensor=w_cross_entropy, name='weighted_softmax_cross_entropy')
    return loss


def weighted_cross_entropy(y_pred, y_true, num_classes, eps=1e-12):
    """
    Define weighted cross-entropy function for classification tasks. Assuming y_pred already probabilistic.
    :param y_pred: tensor [None, width, height, n_classes]
    :param y_true: tensor [None, width, height, n_classes]
    :param eps: (float) small value to avoid division by zero
    :param num_classes: (int) number of classes
    :return:
    """

    n = [tf.reduce_sum(input_tensor=tf.cast(y_true[..., c], tf.float32)) for c in range(num_classes)]
    n_tot = tf.reduce_sum(input_tensor=n)

    weights = [n_tot / (n[c] + eps) for c in range(num_classes)]

    y_pred = tf.reshape(y_pred, (-1, num_classes))
    y_true = tf.cast(tf.reshape(y_true, (-1, num_classes)), dtype=tf.float32)

    w_cross_entropy = -tf.reduce_sum(input_tensor=tf.multiply(y_true * tf.math.log(y_pred + eps), weights), axis=[1])
    loss = tf.reduce_mean(input_tensor=w_cross_entropy, name='weighted_cross_entropy')
    return loss


def shannon_binary_entropy_loss(incoming, axis=(1, 2), unscaled=True, smooth=1e-12):
    """
    Evaluates shannon entropy on a binary mask. The last index contains one-hot encoded predictions.
    :param incoming: incoming tensor (one-hot encoded). On the first dimension there is the number of samples (typically
                the batch size)
    :param axis: axis containing the input dimension. Assuming 'incoming' to be a 4D tensor, axis has length 2: width
                and height; if 'incoming' is a 5D tensor, axis should have length of 3, and so on.
    :param unscaled: The computation does the operations using the natural logarithm log(). To obtain the actual entropy
                value one must scale this value by log(2) since the entropy should be computed in base 2 (hence log2()).
                However, one may desire using this function in a loss function to train a neural net. Then, the log(2)
                is just a multiplicative constant of the gradient and could be omitted for efficiency reasons. Turning
                this flag to False allows for exact actual entropy evaluation; default behaviour is True.
    :param smooth: This small value will be added to the numerator and denominator.
    :return:
    """
    return shannon_binary_entropy(incoming, axis=axis, unscaled=unscaled, smooth=smooth)


def contrastive_loss(y_pred, y_true, num_classes, margin=1.0):
    """ Euclidian distance between the two sets of tensors """

    y_pred = tf.reshape(y_pred, (-1, num_classes))
    y_true = tf.reshape(y_true, (-1, num_classes))

    # This would be the average distance between classes in Euclidian space
    distances = tf.sqrt(tf.reduce_sum(input_tensor=tf.math.squared_difference(y_pred - y_true), axis=1))

    # label here is {0,1} for neg, pos. pairs in the contrastive loss
    loss = tf.reduce_mean(input_tensor=tf.cast(y_true, distances.dtype) * tf.square(distances) +
                          (1. - tf.cast(y_true, distances.dtype)) *
                          tf.square(tf.maximum(margin - distances, 0.)),
                          name='contrastive_loss')
    return loss


def hyperbolic_contrastive_loss(y_pred, y_true, num_classes, margin=1.0, radius=1.0):
    """ Hyperbolic distance between the two sets of tensors """

    y_pred = tf.reshape(y_pred, (-1, num_classes))
    y_true = tf.reshape(y_true, (-1, num_classes))

    # This would be the average distance between classes on the Poincaré disk:
    distances = tf_poinc_dist_sq(y_pred, y_true, c=radius)

    # label here is {0,1} for neg, pos. pairs in the contrastive loss
    loss = tf.reduce_mean(input_tensor=tf.cast(y_true, distances.dtype) * tf.square(distances) +
                          (1. - tf.cast(y_true, distances.dtype)) *
                          tf.square(tf.maximum(margin - distances, 0.)),
                          name='hyperbolic_contrastive_loss')
    return loss
