import matplotlib  # set the matplotlib backend so figures can be saved in the background
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import tensorflow as tf
import numpy as np


def safe_rgb(image, rescale=True, eps=1e-16):
    """ Safely convert input image to RGB """
    if rescale:
        image = (image - np.min(image)) / (np.ptp(image) + eps)

    if image.shape[-1] == 1:
        return np.stack((np.squeeze(image, axis=-1),) * 3, axis=-1)
    elif image.shape[-1] == 3:
        return image
    elif image.shape[-1] == 4:
        return image[..., :3]
    else:
        raise Exception("Input tensor has shape: {0}, while expected shape has last dimension in: [1, 3, 4]"
                        .format(image.shape))


def plot_custom_image_grid(img_list, title='', max_rows=5, rescale=True):
    """
    Returns an image compatible with tf.image.summary() containing the images in img_list all tiled together.
    :param img_list: list of lists with images
    :param title: title for the plot
    :param max_rows: maximum number of samples to show
    :param rescale: rescale image between its min and max values. Can be a bool, or list of bool
    """
    n_rows, n_cols = len(img_list), len(img_list[0])
    n_rows = min(n_rows, max_rows)
    h, w = img_list[0][0].shape[0], img_list[0][0].shape[1]

    # if it's not a list, make rescale a list of bool
    if not isinstance(rescale, list):
        if not isinstance(rescale, bool):
            raise TypeError
        rescale = [rescale] * n_rows
    assert len(rescale) == n_rows

    # initialize and then fill empty array with input images:
    tiled = np.zeros((n_rows * h, n_cols * w, 3))

    # Create a figure to contain the image
    figure = plt.figure(figsize=(10*n_rows, 10*n_cols))

    # fill the empty array
    for i in range(n_rows):
        for j in range(n_cols):
            x = safe_rgb(img_list[i][j], rescale=rescale[i])
            tiled[i * h: (i + 1) * h, j * w: (j+1) * w, :] = x

    # Create a pyplot image and save to buffer:
    plt.title(title)
    plt.imshow(tiled, interpolation='nearest')  # cmap=plt.cm.binary)

    # Save the plot to a PNG in memory:
    memory_buffer = io.BytesIO()
    plt.axis('off')
    plt.savefig(memory_buffer, format='png', bbox_inches='tight', pad_inches=0)

    # Closing the figure prevents it from being displayed directly inside a notebook:
    plt.close(figure)

    memory_buffer.seek(0)

    # Convert PNG buffer to TF image:
    image = tf.image.decode_png(memory_buffer.getvalue(), channels=4)

    # Add the batch dimension:
    image = tf.expand_dims(image, 0)

    # return image compatible with tf.image.summary()
    return image


def plot_weight_histogram(values, title=''):
    """
    Returns an histogram of the given values
    """

    # Create a figure to contain the image
    figure = plt.figure(figsize=(30, 30))

    values = np.reshape(np.array(values), newshape=[-1])
    hist, bin_edges = np.histogram(values)
    plt.bar(bin_edges[:-1], hist, width=0.5, color='#0504aa', alpha=0.7)
    plt.xlim(min(bin_edges), max(bin_edges))
    plt.grid(axis='y', alpha=0.75)

    # Create a pyplot image and save to buffer:
    plt.title(title)
    plt.show()

    # Save the plot to a PNG in memory:
    memory_buffer = io.BytesIO()
    plt.axis('off')
    plt.savefig(memory_buffer, format='png', bbox_inches='tight', pad_inches=0)

    # Closing the figure prevents it from being displayed directly inside a notebook:
    plt.close(figure)

    memory_buffer.seek(0)

    # Convert PNG buffer to TF image:
    image = tf.image.decode_png(memory_buffer.getvalue(), channels=4)

    # Add the batch dimension:
    image = tf.expand_dims(image, 0)

    # return image compatible with tf.image.summary()
    return image
