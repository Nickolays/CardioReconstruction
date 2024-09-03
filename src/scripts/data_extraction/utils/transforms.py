import torch
import numpy as np


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
    
    
def bgr_to_gray(self, bgr):
        """
        Convert a RGB image to a grayscale image
            Differences from cv2.cvtColor():
                1. Input image can be float
                2. Output image has three repeated channels, other than a single channel

        Args:
            bgr: Image in BGR format
                 Numpy array of shape (h, w, 3)

        Returns:
            gs: Grayscale image
                Numpy array of the same shape as input; the three channels are the same
        """
        ch = 0.114 * bgr[:, :, 0] + 0.587 * bgr[:, :, 1] + 0.299 * bgr[:, :, 2]
        gs = np.dstack((ch, ch, ch))
        return gs