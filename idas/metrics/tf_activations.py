import tensorflow as tf


def softmax_nd(target, axis, name=None):
    """
    Multi dimensional softmax,
    refer to https://github.com/tensorflow/tensorflow/issues/210
    compute softmax along the dimension of target
    the native softmax only supports batch_size x dimension
    """
    with tf.compat.v1.name_scope(name, 'softmax', values=[target]):
        max_axis = tf.reduce_max(input_tensor=target, axis=axis, keepdims=True)
        target_exp = tf.exp(target - max_axis)
        normalize = tf.reduce_sum(input_tensor=target_exp, axis=axis, keepdims=True)
        sft = target_exp / normalize
        return sft
