import os
import random

import cv2
import numpy as np
import torch
from skimage.draw import disk



class Compose(object):
    """ Composes several transforms together.
    For example:
    >>> transforms.Compose([
    >>>     transforms.RandomBackground(),
    >>>     transforms.CenterCrop(127, 127, 3),
    >>>  ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, rendering_images, bounding_box=None):
        for t in self.transforms:
            if t.__class__.__name__ == 'RandomCrop' or t.__class__.__name__ == 'CenterCrop':
                rendering_images = t(rendering_images, bounding_box)
            else:
                rendering_images = t(rendering_images)

        return rendering_images


class ToTensor(object):
    """
    Convert a PIL Image or numpy.ndarray to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, rendering_images):
        assert (isinstance(rendering_images, np.ndarray))
        array = np.transpose(rendering_images, (0, 3, 1, 2))
        # handle numpy array
        tensor = torch.from_numpy(array)

        # put it from HWC to CHW format
        return tensor.float()


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, rendering_images):
        assert (isinstance(rendering_images, np.ndarray))
        rendering_images -= self.mean
        rendering_images /= self.std

        return rendering_images