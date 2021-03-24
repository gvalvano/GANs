import tensorflow as tf
import numpy as np
import cv2
from glob import glob
import os
from idas.data_utils.png_utils import get_png_image
from idas.utils import print_yellow_text, safe_mkdir


# ----------------------------------------------------------------------------------------------------------------
FINAL_SIZE = (128, 256)
root = '~/python_projects'
source_dir = root + '/DATA/cityscapes/segmentation/raw'
dest_dir = root + '/DATA/cityscapes/segmentation/processed'

N_CLASSES = 20  # number of actual classes, after removing labels to ignore
IMAGE_SIZE = (1024, 2048)  #  image size for data in cityscapes
# ----------------------------------------------------------------------------------------------------------------


def preprocess_labels(label):
    """ Returns labels from 0 to 19, where:
         - 0 is background (ignored classes),
         - from 1 to 19 we have the other classes.
    """
    ignore_label = 255
    mapping = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
               3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
               7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
               14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
               18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
               28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}

    label = np.array(label)
    label_copy = label.copy()
    for k, v in mapping.items():
        # set ignored classes to zero label, shift the remaining by 1:
        if v != ignore_label:
            label_copy[label == k] = v + 1
        else:
            label_copy[label == k] = 0

    return label_copy


def resize(image, new_size, interpolation):
    """ Resize the input images
    :param image: input images
    :param new_size: [int, int] output size, with shape (N, M)
    :param interpolation: interpolation type
    :return: resized images
    """
    target_height = new_size[0]
    target_width = new_size[1]

    resized = cv2.resize(image, (target_width, target_height), interpolation=interpolation)
    return resized


def one_hot_encode(y, nb_classes):
    y_shape = list(y.shape)
    y_shape.append(nb_classes)
    with tf.device('/cpu:0'):
        with tf.compat.v1.Session() as sess:
            res = sess.run(tf.one_hot(indices=y, depth=nb_classes))
    return res.reshape(y_shape)


def get_filenames(root_dir):
    path_dict = {'train': {'image': '', 'label': ''},
                 'validation': {'image': '', 'label': ''},
                 'test': {'image': '', 'label': ''}}

    n_images = 0
    for d_set in ['train', 'validation']:  # , 'test']:  # skip test set
        city_list = [d.rsplit('/')[-1] for d in glob(os.path.join(root_dir, '{0}/*'.format(d_set)))]

        img_list = []
        for city in city_list:
            path = os.path.join(root_dir, '{0}/{1}/{1}*leftImg8bit.png'.format(d_set, city))
            images = [d.rsplit('leftImg8bit.png')[0] for d in glob(path)]
            img_list.extend(images)

        path_dict[d_set]['image'] = [el + 'leftImg8bit.png' for el in img_list]
        path_dict[d_set]['label'] = [el + 'gtFine_labelIds.png' for el in img_list]
        n_images += len(img_list)

    return path_dict, n_images


def preprocess_data(fnames_dict, new_size=FINAL_SIZE, dest_dir='./preprocessed'):

    file_id = 0
    for d_set in ['train', 'validation']:  # , 'test']:  # skip test set

        # create destination directory, if it does not exist:
        dest_subdir = os.path.join(dest_dir, d_set)
        safe_mkdir(dest_subdir)

        # process and save images:
        for name_image, name_mask in zip(fnames_dict[d_set]['image'], fnames_dict[d_set]['label']):
            file_id += 1
            print('# {0} : {1}'.format(file_id, name_image))

            image = get_png_image(name_image)
            mask = get_png_image(name_mask)
            mask = preprocess_labels(mask)

            # cast
            image = image.astype(np.float32)
            mask = mask.astype(np.float32)

            # resize images on the major axis
            image = resize(image, new_size, interpolation=cv2.INTER_CUBIC)
            mask = resize(mask, new_size, interpolation=cv2.INTER_NEAREST)

            # # crop or pad image and annotations to be have final size
            # image = crop_or_pad_slice_center(image, new_size=FINAL_IMAGE_SIZE)
            # mask = crop_or_pad_slice_center(mask, new_size=FINAL_IMAGE_SIZE)

            # one-hot encoding of the annotations
            mask = one_hot_encode(mask, nb_classes=20)  # slice to remove alpha channel

            # standardize image
            image = image / np.max(np.abs(image))
            # image = image - np.mean(image)

            # save pre-processed file
            name_prefix = name_image.rsplit('_leftImg8bit.png')[0]
            name, city = name_prefix.rsplit('/')[-1], name_prefix.rsplit('/')[-2]
            city_subdir = os.path.join(dest_subdir, city)
            safe_mkdir(city_subdir)
            np.save(os.path.join(city_subdir, name + '_img.npy'), image)
            np.save(os.path.join(city_subdir, name + '_mask.npy'), mask.astype(np.uint8))


def main():

    print('\nBuilding data sets...')
    print_yellow_text('Nota Bene: the file will only process train and validation set as not all the data for the test '
                      'are available.')

    # create numpy dataset:
    fnames, tot_images = get_filenames(source_dir)
    print_yellow_text('Total number of images = {0}'.format(tot_images))
    preprocess_data(fnames, new_size=FINAL_SIZE, dest_dir=dest_dir)

    print_yellow_text('\nDone.\n', sep=False)


if __name__ == '__main__':
    main()
