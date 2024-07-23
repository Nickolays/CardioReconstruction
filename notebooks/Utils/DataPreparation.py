"""
            Prepare Data for CNN
"""
import numpy as np
import tensorflow as tf
import os

""" TO DO: 
        1. Create constructor for all formats preparation, not only for MHD format

"""


# class Preparation():
#     def __init__(self):
#         self.img_format = None
#         ...


class PreparationMHD():
    def __init__(self, main_path_to_data, image_size, subset='training', validation_split=0.2, dtype=np.float16):
        """

        :param main_path_to_data:
        :param image_size
        :param subset:
        :param validation_split:
        :param dtype:

        :return np.ArrayIteration for flow method into NN
        """
        self.path_main = main_path_to_data
        self.image_size = image_size
        self.subset = subset
        self.validation_split = 0.2
        self.dtype = dtype
        #
        self.path_train = os.path.join(self.path_main, 'training')
        self.path_test = os.path.join(self.path_main, 'testing')
        self.img_format = 'mhd'


def _prepare_imgs(self):
    """  """
    # Train
    train_image_filepaths = self.get_image_filepaths
    train_mask_filepaths = self.get_image_filepaths
    # Test
    test_image_filepaths = self.get_image_filepaths
    test_mask_filepaths = self.get_image_filepaths


def __get_image_filepaths(self):
    """  """
    # Train
    =
    =
    # Test
    =
    =


    # Получить все пути
    # Train
    train_image_filepaths = get_image_filepaths(train_path, img_format, as_mask=False)
    train_mask_filepaths = get_image_filepaths(train_path, img_format, as_mask=True)
    # Test
    test_image_filepaths = get_image_filepaths(test_path, img_format, as_mask=False)
    test_mask_filepaths = get_image_filepaths(test_path, img_format, as_mask=True)

    # Загрузить данные
    X_train = get_data(train_image_filepaths, img_shape=IMAGE_SIZE)
    y_train = get_data(train_mask_filepaths, img_shape=IMAGE_SIZE, as_mask=True)

    X_test = get_data(test_image_filepaths, img_shape=IMAGE_SIZE)
    y_test = get_data(test_mask_filepaths, img_shape=IMAGE_SIZE, as_mask=True)
