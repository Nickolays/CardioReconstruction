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
    imgs_paths = get_image_filepaths("src/tests")
    return imgs_paths

@pytest.fixture
def get_model():
    """ Comlile model and download weights """
    model = U2Net()
    weights_path = "results/U2Net/u2net-cardio_segmentation.pt" 
    model.load_state_dict(torch.load(weights_path))
    return model

# @pytest.fixture
# def input_image():
#     """
#     Fixture to return a PIL image object of the test image used for testing.
#     """
#     input_image = Image.open('./tests/test_image.jpg').convert("RGB")
#     return(input_image)

# @pytest.fixture
# def predictions():
#     """
#     Fixture to return the predictions and label names for the test image.
#     """get_paths()= 
#     model = YOLO("./models/sample_model/yolov8n.pt")
#     predictions = model.predict(source=input_image)
#     return(predictions, model.model.names)


################################ Test #####################################################

def test_initialize_models():
    """
    Test to check if all the models are loading correctly.
    """
    model_sample_model = U2Net
    assert model_sample_model is not None
    assert os.path.exists("results/U2Net/u2net-cardio_segmentation.pt" )

def test_initialize_dataloader():
    """
    
        Loader should return a np.float32 (wo tensor yet), and output shape have form (bs, chnls, weight, height), 
    for image and mask. And also the output shape of both images should have the same size: (256, 256)
    """
    image_size = (256, 256)
    imgs_paths = get_paths()
    dataloader = HeartLoader(imgs_paths, imgs_paths, image_size)
    image, mask = dataloader(1)

    assert image.ndim() == 4
    assert mask.ndim() == 4
    assert image.shape[1:] == image_size
    assert mask.shape[1:] == image_size

def test_predict():
    """  """
    input_image = torch.randn((2, 3, 256, 256))
    model = get_model()
    predict = model(input_image)[-1]
    assert predict
    assert predict.shape == (2, 1, 256, 256)

# def test_train_step():



# def test_transform_predict_to_df(predictions):
#     """
#     Test the function 'transform_predict_to_df' which converts the predictions from the YOLO model to a pandas DataFrame.
#     It takes in two arguments:
#         predictions: A list of dictionaries returned by the YOLO model
#         label_names: A list of class labels for the YOLO model
#     It returns a DataFrame with columns:
#         'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'
#     Asserts:
#         - The returned object is a DataFrame
#         - The columns of the DataFrame are as expected
#         - The DataFrame contains at least one object of class 'dog'
#     """
#     predictions, label_names = predictions
#     predict_bbox = transform_predict_to_df(predictions, label_names)
#     # Check if the returned object is an instance of pd.DataFrame
#     assert isinstance(predict_bbox, pd.DataFrame)
#     # Check if the returned DataFrame has the correct columns
#     assert set(predict_bbox.columns) == set(['xmin', 'ymin', 'xmax','ymax', 'confidence', 'class', 'name'])
#     assert 'dog' in predict_bbox.name.tolist()

# def test_get_model_predict(input_image):
#     """
#     Test to check if the function 'get_model_predict' is returning a DataFrame object with the correct columns and number of rows.
#     It also checks if the returned object is an instance of pd.DataFrame
#     """
#     model_sample_model = YOLO("./models/sample_model/yolov8n.pt")
#     predictions = get_model_predict(model_sample_model, input_image)
#     # Check if the returned object is an instance of pd.DataFrame
#     assert isinstance(predictions, pd.DataFrame)
#     # Check if the returned DataFrame has the correct columns
#     assert set(predictions.columns) == set(['xmin', 'ymin', 'xmax','ymax', 'confidence', 'class', 'name'])
#     # Check if the returned DataFrame has more than one row
#     assert len(predictions) > 1

# def test_add_bboxs_on_img(input_image, predictions):
#     """
#     Test to check if the function 'add_bboxs_on_img' is adding bounding boxes on the image and returning the image object.
#     """
#     predictions, label_names = predictions
#     predict_bbox = transform_predict_to_df(predictions, label_names)
#     image_with_bbox = add_bboxs_on_img(input_image, predict_bbox)
#     assert isinstance(image_with_bbox, Image.Image)





# def test_get_image_from_bytes(test_image):
#     """
#     Test to check if the function 'get_image_from_bytes' is converting the binary image data to a PIL image object.
#     """
#     binary_image = test_image['file'].read()
#     output = get_image_from_bytes(binary_image)
#     assert isinstance(output, Image.Image) and output.mode == "RGB"

# def test_get_bytes_from_image(input_image):
#     """
#     Test to check if the function 'get_bytes_from_image' is converting the PIL image object to binary image data.
#     """
#     output = get_bytes_from_image(input_image)
#     assert isinstance(output, io.BytesIO)