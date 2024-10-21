import numpy as np
import matplotlib.pyplot as plt
import torch

from skimage import measure
from scipy.interpolate import RegularGridInterpolator

import os, cv2
from tqdm import tqdm
from natsort import natsorted

import warnings
warnings.filterwarnings('ignore')



def interpolate(input_image, method="linear"):

    # assert 

    pixelsize_old = 1
    slice_thickness_old = 10

    pixelsize_new = 1
    slice_thickness_new = 5

    x_old = np.linspace(0, (input_image.shape[1]-1)*pixelsize_old, input_image.shape[1])
    y_old = np.linspace(0, (input_image.shape[2]-1)*pixelsize_old, input_image.shape[2])
    z_old = np.arange(0, (input_image.shape[0]))*slice_thickness_old

    my_interpolating_object = RegularGridInterpolator((z_old, x_old, y_old), input_image, method=method, bounds_error=False)

    x_new = np.round(input_image.shape[1]*pixelsize_old/pixelsize_new).astype('int')
    y_new = np.round(input_image.shape[2]*pixelsize_old/pixelsize_new).astype('int')
    z_new = np.arange(z_old[0], z_old[-1], slice_thickness_new)

    # pts is the new grid
    pts = np.indices((len(z_new), x_new, y_new)).transpose((1, 2, 3, 0))
    pts = pts.reshape(1, len(z_new)*x_new*y_new, 1, 3).reshape(len(z_new)*x_new*y_new, 3)
    pts = np.array(pts, dtype=float)
    pts[:, 1:3] = pts[:, 1:3]*pixelsize_new
    pts[:, 0] = pts[:, 0]*slice_thickness_new + z_new[0]

        # Interpolate
    interpolated_data = my_interpolating_object(pts)
    interpolated_data = interpolated_data.reshape(len(z_new), x_new, y_new)

    interpolated_data_16bit = (np.round(((interpolated_data - interpolated_data.min())/(interpolated_data.max() - interpolated_data.min())) * 255.0)).astype(np.uint8)

    # Use marching cubes to obtain the surface mesh of these ellipsoids
    verts, faces, normals, values = measure.marching_cubes(interpolated_data_16bit, 0)

    return verts, faces, normals, values


def prepare_input(img, image_size=(512, 512)):
    # 
    if img.shape[:2] != image_size:
        img = cv2.resize(img, image_size)

    img = img / 255
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    img = img.astype(np.float32)
    img = torch.from_numpy(img).cuda()

    return img

def prepare_output(pred):
    if len(pred) > 1:
        pred = pred[-1]
    if pred.ndim == 4:
        pred = pred[0]

    pred = pred.cpu().detach().numpy()
    pred = sigmoid(pred)
    return np.transpose(pred, (1, 2, 0))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
