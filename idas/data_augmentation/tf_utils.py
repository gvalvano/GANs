import tensorflow as tf


def flip_tensors_left_right(list_of_tensors, probability=0.5):
    """Randomly flip a list of tensors (i.e. image and mask).
    """
    assert 0.0 <= probability <= 1.0

    uniform_random = tf.random.uniform([], 0, 1.0)
    flip_cond = tf.less(uniform_random, probability)
    augmented = []
    for tensor in list_of_tensors:
        augmented.append(tf.cond(pred=flip_cond, true_fn=lambda: tf.image.flip_left_right(tensor), false_fn=lambda: tensor))

    return augmented


def flip_tensors_up_down(list_of_tensors, probability=0.5):
    """Randomly flip a list of tensors (i.e. image and mask).
    """
    assert 0.0 <= probability <= 1.0

    uniform_random = tf.random.uniform([], 0, 1.0)
    flip_cond = tf.less(uniform_random, probability)
    augmented = []
    for tensor in list_of_tensors:
        augmented.append(tf.cond(pred=flip_cond, true_fn=lambda: tf.image.flip_up_down(tensor), false_fn=lambda: tensor))

    return augmented
