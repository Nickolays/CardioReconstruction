import numpy as np
import tensorflow as tf
import os, cv2
from tqdm import tqdm
import SimpleITK as sitk



def get_data(paths: list, img_shape, as_mask=False):
    """ img_shape - Output image size. Without channels. """
    n_photo = len(paths)  # Batch size
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
        if as_mask == True:
            image = np.where(image == 1, 1, 0)
        # Resize
        image = cv2.resize(image.astype(np.uint8), dsize=img_shape, interpolation=cv2.INTER_LINEAR)  # INTER_CUBIC

        # Convert to float
        image = image / image.max()
        # Add to array
        images[i, :, :, 0] = image.astype(np.float32)

    return images



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
                    if 'gt' in name:
                        filepaths.append(os.path.join(address, name))
                else:
                    if 'gt' not in name and 'sequence' not in name:
                        filepaths.append(os.path.join(address, name))

    return filepaths

