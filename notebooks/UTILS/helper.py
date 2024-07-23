import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import keras.backend as K
import tensorflow as tf
import skimage
import nibabel as nib

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
                if as_mask:
                    if 'gt' in name and 'sequence' not in name:
                        filepaths.append(os.path.join(address, name))
                else:
                    if 'gt' not in name and 'sequence' not in name:
                        filepaths.append(os.path.join(address, name))
                
    return filepaths

def get_data(paths: list, img_shape, as_mask=False):
    """ img_shape - Output image size. Without channels. """
    n_photo = len(paths)      # Batch size
    if as_mask:
        images = np.empty(shape=(n_photo, img_shape[0], img_shape[1], 1), dtype=np.float16)
    else:
        images = np.empty(shape=(n_photo, img_shape[0], img_shape[1], 1), dtype=np.float32)
    
    for i, path in tqdm(enumerate(paths), total=n_photo):
        # Open mhd
        itk_image = sitk.ReadImage(path)
        image = sitk.GetArrayViewFromImage(itk_image)
        # Crop image
        # image = image[:, 150:-150, 50:-50]
        # Reshape
        image = image.reshape((image.shape[1], image.shape[2], 1))
        # Choose only 1 class from 3  (LV_Endo)
        # if as_mask == True:
        #     image = np.where(image == 1, 1, 0)
        # Resize
        image = cv2.resize(image.astype(np.uint8), dsize=img_shape, interpolation=cv2.INTER_LINEAR)  # INTER_CUBIC
        
        # Convert to float
        image = image / image.max()
        # Add to array
        if as_mask:
            images[i, :, :, 0] = image.astype(np.float16)
        else:
            images[i, :, :, 0] = image.astype(np.float32)
    
    return images

def read_mhd(path):
    # Open mhd
    print(path)
    itk_image = sitk.ReadImage(path)
    image = sitk.GetArrayViewFromImage(itk_image)
    # Reshape
    image = image.reshape((image.shape[1], image.shape[2], 1))

    return image

def read_nifti(path):
    #
    nii_img = nib.load(path)
    nii_data = nii_img.get_fdata()
    # Rotate image
    nii_data = cv2.rotate(nii_data, cv2.ROTATE_90_CLOCKWISE)

    return np.expand_dims(nii_data, axis=-1)

def get_img_txt_path(inp_path, format='jpeg'):
    """ Вернёт список с путями до фото и контрольными точками в папке """
    # Find all objects in folder
    objects_name = os.listdir(inp_path)
    # Define new lists
    image_paths = []
    descriptions_paths = []
    #
    for obj_name in objects_name:
        if format in obj_name:
            image_paths.append(os.path.join(inp_path, obj_name))
        elif 'txt' in obj_name:
            descriptions_paths.append(os.path.join(inp_path, obj_name))
        else:
            print('obj_name')

    return sorted(image_paths, reverse=False), descriptions_paths


def format2jpeg(in_paths, in_format, out_path, output_shape, is_mask, start_indx=0):
    """ Remove all folder """
    # Find all paths in folder
    paths = get_image_filepaths(in_paths, in_format, as_mask=is_mask)
    #
    for i, path in tqdm(enumerate(paths)):
        #
        i += start_indx

        if in_format == "nii":
            img = read_nifti(path)

        elif in_format == "mhd":
            img = read_mhd(path)

        else:
            print(in_format)
            break

        # Obtain needed image shape
        img = skimage.transform.resize(img, output_shape=output_shape)
        # Save image
        # name = os.path.join(out_path, f"{str(i+1).zfill(4)}.jpeg")
        name = os.path.join(out_path, f"{str(i).zfill(4)}.jpeg")

        skimage.io.imsave(name, img)

    # return True