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
import numpy as np


def get_shape(tensor):
    """ It returns the static shape of a tensor when available, otherwise returns its dynamic shape.
        .Example
          |  a.set_shape([32, 128])  # static shape of a is [32, 128]
          |  a.set_shape([None, 128])  # first dimension of a is determined dynamically
    """
    static_shape = tensor.shape.as_list()
    dynamic_shape = tf.unstack(tf.shape(input=tensor))
    dims = [s[1] if s[0] is None else s[0]
            for s in zip(static_shape, dynamic_shape)]
    return dims


def reshape_tensor(tensor, dims_list):
    """ General purpose reshape function to collapse any list of dimensions.
        .Example
        We want to convert a Tensor of rank 3 to a tensor of rank 2 by collapsing the second and third dimensions
        into one:
          |  b = tf.placeholder(tf.float32, [None, 10, 32])
          |  shape = get_shape(b)
          |  b = tf.reshape(b, [shape[0], shape[1] * shape[2]])
        With this function, we can easily write:
          |  b = tf.placeholder(tf.float32, [None, 10, 32])
          |  b = reshape(b, [0, [1, 2]])  # hence: collapse [1, 2] into the same dimension, leave 0 dimension unchanged
    """
    shape = get_shape(tensor)
    dims_prod = []
    for dims in dims_list:
        if isinstance(dims, int):
            dims_prod.append(shape[dims])
        elif all([isinstance(shape[d], int) for d in dims]):
            dims_prod.append(np.prod([shape[d] for d in dims]))
        else:
            dims_prod.append(tf.reduce_prod(input_tensor=[shape[d] for d in dims]))
    tensor = tf.reshape(tensor, dims_prod)
    return tensor


def from_one_hot_to_rgb(incoming, palette=None, no_background=False, background_color='black'):
    """ Assign a different color to each class in the input tensor """
    assert background_color in ['black', 'white']
    bgd_color = {
        'black': (0, 0, 0),
        'white': (255, 255, 255)
    }
    if palette is None:
        palette = [bgd_color[background_color],
                   [215, 48, 39],
                   [69, 117, 180],
                   [255, 255, 191],
                   [253, 174, 97],
                   [171, 217, 233],
                   [244, 109, 67],
                   [116, 173, 209],
                   [254, 224, 144],
                   [224, 243, 248],
                   [140, 81, 10],
                   [1, 102, 94],
                   [191, 129, 45],
                   [53, 151, 143],
                   [197, 27, 125],
                   [77, 146, 33],
                   [222, 119, 174],
                   [184, 225, 134],
                   [241, 182, 218],
                   [118, 42, 131],
                   [231, 212, 232],
                   [179, 88, 6],
                   [178, 24, 43],
                   [158, 188, 218],
                   [189, 189, 189],
                   [140, 150, 198],
                   [115, 115, 115],
                   [255, 255, 217],
                   [255, 255, 217]]

    if no_background:
        palette.pop(0)
    palette = np.array(palette, np.uint8)

    with tf.compat.v1.name_scope('from_one_hot_to_rgb'):
        _, W, H, _ = get_shape(incoming)
        palette = tf.constant(palette, dtype=tf.uint8)
        class_indexes = tf.argmax(input=incoming, axis=-1)

        class_indexes = tf.reshape(class_indexes, [-1])
        color_image = tf.gather(palette, class_indexes)
        color_image = tf.reshape(color_image, [-1, W, H, 3])

        color_image = tf.cast(color_image, dtype=tf.float32)

    return color_image


def add_histogram(writer, tag, values, step, bins=1000):
    """
    Logs the histogram of a list/vector of values.
    From: https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
    """

    # Create histogram using numpy
    counts, bin_edges = np.histogram(values, bins=bins)

    # Fill fields of histogram proto
    hist = tf.compat.v1.HistogramProto()
    hist.min = float(np.min(values))
    hist.max = float(np.max(values))
    hist.num = int(np.prod(values.shape))
    hist.sum = float(np.sum(values))
    hist.sum_squares = float(np.sum(values ** 2))

    # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
    # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
    # Therefore we drop the start of the first bin
    bin_edges = bin_edges[1:]

    # Add bin edges and counts
    for edge in bin_edges:
        hist.bucket_limit.append(edge)
    for c in counts:
        hist.bucket.append(c)

    # Create and write Summary
    summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=tag, histo=hist)])
    writer.add_summary(summary, step)
