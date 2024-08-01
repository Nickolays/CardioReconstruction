import pytest
from PIL import Image
import pandas as pd
import sys
import os
import io

# pytest app import fix
dynamic_path = os.path.abspath('.')
print(dynamic_path)

sys.path.append(dynamic_path)

import torch

from src.scripts.DataLoaders import HeartLoader, get_image_filepaths
from src.scripts.models.U2Net import U2Net
from src.scripts.DataLoaders import HeartLoader


################################ Fixtures #####################################################

@pytest.fixture
def get_paths():
    """
    # Fixture to return a file object of the test image used for testing.
    """
    # files = {'file': open('./tests/test_image.jpg', 'rb')}
    # return(files)
    imgs_paths = get_image_filepaths("data/train/train")
    return imgs_paths

# @pytest.fixture
def predicting():
    """ Comlile model and download weights """
    weights_path = "results/U2Net/u2net-cardio_segmentation_1.pt" 
    input_image = torch.randn((2, 3, 256, 256)).cuda()
    model = U2Net().cuda()
    model.eval()
    model.load_state_dict(torch.load(weights_path))
    predict = model(input_image)[-1]

    return predict

################################ Test #####################################################

def test_initialize_models():
    """
    Test to check if all the models are loading correctly.
    """
    model_sample_model = U2Net()
    assert model_sample_model is not None
    assert os.path.exists("results/U2Net/u2net-cardio_segmentation.pt" )

def test_initialize_dataloader():
    """
    
        Loader should return a np.float32 (wo tensor yet), and output shape have form (bs, chnls, weight, height), 
    for image and mask. And also the output shape of both images should have the same size: (256, 256)
    """
    image_size = (256, 256)
    imgs_paths = ['src/tests/test_frame_1.jpg', 'src/tests/test_frame_2.png']  # get_paths()
    dataloader = HeartLoader(imgs_paths, imgs_paths, image_size)
    image, mask = dataloader[1]

    assert image.ndim == 3
    assert mask.ndim == 3
    assert image.shape[1:] == image_size
    assert mask.shape[1:] == image_size

def test_predict():
    """  """
    predict = predicting()
    
    assert predict.shape == (2, 1, 256, 256)

# def test_train_step():

