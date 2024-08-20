import numpy as np
import matplotlib.pyplot as plt
# import SimpleITK as sitk
# import keras.backend as K
# import tensorflow as tf

from skimage import measure
from scipy.interpolate import RegularGridInterpolator

import os, cv2
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')


def get_image_filepaths(main_path, img_format, as_mask=False):
    
    # Сюда всё запишем
    filepaths = []
    # Итерируемся по всем пациентам
    for address, dirs, files in os.walk(main_path):
        # Все названия внутри каждого пациента
        for name in files:
            # Выбираем нужный формат
            if img_format in name:
                # Добавляем путь до изображения или маски
                if os.path.exists(os.path.join(address, name)):
                    filepaths.append(os.path.join(address, name))

    # assert len(files) == len(filepaths), f"There're another format of files, .txt, for instance. File is: {filepaths}"
    
    return filepaths


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




# def get_image_filepaths(main_path, img_format, as_mask=False):
    
#     # Сюда всё запишем
#     filepaths = []
#     # Итерируемся по всем пациентам
#     for address, dirs, files in os.walk(main_path):
#         # Все названия внутри каждого пациента
#         for name in files:
#             # Выбираем нужный формат
#             if img_format in name:
#                 # Добавляем путь до изображения или маски
#                 if as_mask:
#                     if 'gt' in name:
#                         filepaths.append(os.path.join(address, name))
#                 else:
#                     if 'gt' not in name and 'sequence' not in name:
#                         filepaths.append(os.path.join(address, name))
                
#     return filepaths

# def get_data(paths: list, img_shape, as_mask=False):
#     """ img_shape - Output image size. Without channels. """
#     n_photo = len(paths)  # Batch size
#     if as_mask:
#         # images = np.empty(shape=(n_photo, img_shape[0], img_shape[1], 1), dtype=np.float16)
#         images = np.empty(shape=(n_photo, img_shape[0], img_shape[1], 1), dtype=np.float32)
#     else:
#         images = np.empty(shape=(n_photo, img_shape[0], img_shape[1], 1), dtype=np.float32)

#     for i, path in tqdm(enumerate(paths), total=n_photo):
#         # Open img
#         image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#         # Resize
#         image = cv2.resize(image.astype(np.uint8), dsize=img_shape, interpolation=cv2.INTER_LINEAR)  # INTER_CUBIC
#         # Convert to float
#         image = image / image.max()
#         # Choose only 1 class from 3  (LV_Endo)
#         if as_mask == True:
#             image = np.where((image > 0.2) & (image < 0.75), 1, 0)
#         # Add to array
#         if as_mask:
#             # images[i, :, :, 0] = image.astype(np.float16)
#             images[i, :, :, 0] = image.astype(np.float32)
#         else:
#             images[i, :, :, 0] = image.astype(np.float32)

#     return images

# def get_mhd_data(paths: list, img_shape, as_mask=False):
#     """ img_shape - Output image size. Without channels. """
#     n_photo = len(paths)      # Batch size
#     if as_mask:
#         images = np.empty(shape=(n_photo, img_shape[0], img_shape[1], 1), dtype=np.float16)
#     else:
#         images = np.empty(shape=(n_photo, img_shape[0], img_shape[1], 1), dtype=np.float32)
    
#     for i, path in tqdm(enumerate(paths), total=n_photo):
#         # Open mhd
#         itk_image = sitk.ReadImage(path)
#         image = sitk.GetArrayViewFromImage(itk_image)
        
#         # Crop image
#         image = image[:, 150:-150, 50:-50]
#         # Reshape
#         image = image.reshape((image.shape[1], image.shape[2], 1))
#         # Choose only 1 class from 3  (LV_Endo)
#         if as_mask == True:
#             image = np.where(image == 1, 1, 0)
#         # Resize
#         image = cv2.resize(image.astype(np.uint8), dsize=img_shape, interpolation=cv2.INTER_LINEAR)  # INTER_CUBIC
        
#         # Convert to float
#         image = image / image.max()
#         # Add to array
#         if as_mask:
#             images[i, :, :, 0] = image.astype(np.float16)
#         else:
#             images[i, :, :, 0] = image.astype(np.float32)
    
#     return images

# # def prepare_imgs():

# @tf.function
# def dice_coef(y_true, y_pred):
#     '''(2*|X & Y|) / (|X| + |Y|)'''
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     return (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))

# @tf.function
# def dice_loss(y_true, y_pred):
#     return 1. - dice_coef(y_true, y_pred)
