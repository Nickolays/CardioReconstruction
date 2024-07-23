import tensorflow as tf
import numpy as np
import SimpleITK as sitk
import os, cv2
from tqdm import tqdm
import random

import warnings
warnings.filterwarnings('ignore')


class CustomDataGen(tf.keras.utils.Sequence):

    def __init__(self, df, X_col, y_col,
                 batch_size,
                 input_size=(224, 224, 3),
                 shuffle=True):
        self.df = df.copy()
        self.X_col = X_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle

        self.n = len(self.df)
        self.n_name = df[y_col['name']].nunique()
        self.n_type = df[y_col['type']].nunique()

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __get_input(self, path, bbox, target_size):
        xmin, ymin, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']

        image = tf.keras.preprocessing.image.load_img(path)
        image_arr = tf.keras.preprocessing.image.img_to_array(image)

        image_arr = image_arr[ymin:ymin + h, xmin:xmin + w]
        image_arr = tf.image.resize(image_arr, (target_size[0], target_size[1])).numpy()

        return image_arr / 255.

    def __get_output(self, label, num_classes):
        return tf.keras.utils.to_categorical(label, num_classes=num_classes)

    def __get_data(self, batches):
        # Generates data containing batch_size samples

        path_batch = batches[self.X_col['path']]
        bbox_batch = batches[self.X_col['bbox']]

        name_batch = batches[self.y_col['name']]
        type_batch = batches[self.y_col['type']]

        X_batch = np.asarray([self.__get_input(x, y, self.input_size) for x, y in zip(path_batch, bbox_batch)])

        y0_batch = np.asarray([self.__get_output(y, self.n_name) for y in name_batch])
        y1_batch = np.asarray([self.__get_output(y, self.n_type) for y in type_batch])

        return X_batch, tuple([y0_batch, y1_batch])

    def __getitem__(self, index):
        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches)
        return X, y

    def __len__(self):
        return self.n // self.batch_size



"""         1 Version   Нумпай загрузка    """

MAIN_PATH_TRAIN = r'J:\Data\training'
PATHE_PATIENT_TRAIN = os.listdir(MAIN_PATH_TRAIN)
IMG_FORMAT = "mhd"
def get_patient_paths(index):
    """   """
    # get_patient_paths
    # patient = PATHES_TRAIN[index]
    # paths = os.listdir(patient)
    #
    x_paths = []
    y_paths = []

    for patient in PATHE_PATIENT_TRAIN:
        for name in os.listdir(patient):
            # Выбираем нужный формат
            if IMG_FORMAT in name:
                # Добавляем путь до изображения или маски
                if 'gt' in name:
                    y_paths.append(os.path.join(MAIN_PATH_TRAIN, name))
                else:   # if 'gt' not in name and 'sequence' not in name:
                    x_paths.append(os.path.join(MAIN_PATH_TRAIN, name))
    return x_paths, y_paths

# Output paths data have shape: (patients, 4, n, n, 1)


class DataGenMHD:

    def __init__(self, image_size, batch_size,):

        self.image_size = image_size
        self.batch_size = batch_size

    def __get_patient_paths(self, index):
        return self.x_paths[index], self.y_paths[index]

    def __generate_paths(self):

        self.x_paths = []
        self.y_paths = []

    def __read_mhd(self, paths, as_mask=False):
        n_photo = len(paths)  # Batch size. Have to be 4
        if as_mask:
            images = np.empty(shape=(n_photo, self.image_size[0], self.image_size[1], 1), dtype=np.float16)
        else:
            images = np.empty(shape=(n_photo, self.image_size[0], self.image_size[1], 1), dtype=np.float32)
        #
        for i, path in tqdm(enumerate(paths), total=n_photo):
            # Open mhd
            itk_image = sitk.ReadImage(path)
            image = sitk.GetArrayViewFromImage(itk_image)
            # Reshape
            image = image.reshape((image.shape[1], image.shape[2], 1))
            # Choose only 1 class from 3  (LV_Endo)
            if as_mask == True:
                image = np.where(image == 1, 1, 0)
            # Resize
            image = cv2.resize(image.astype(np.uint8), dsize=self.image_size, interpolation=cv2.INTER_LINEAR)  # INTER_CUBIC

            # Convert to float
            image = image / image.max()
            # Add to array
            if as_mask:
                images[i, :, :, 0] = image.astype(np.float16)
            else:
                images[i, :, :, 0] = image.astype(np.float32)

        return images

    def __get_data(self, x_paths, y_paths):
        X = self.__read_mhd(x_paths)
        y = self.__read_mhd(y_paths)

        return X, y

    def __getitem__(self, index):
        x_path, y_path = self.__get_patient_paths(index)
        return self.__get_data(x_path, y_path)

    def __len__(self):
        return len(self.x_paths) // (self.batch_size * 4)

    def on_epoch_end(self):
        if self.shuffle:
            self.patients = self.patients.shuffle()



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


def get_data(paths: list, img_shape, as_mask=False):
    """ img_shape - Output image size. Without channels. """
    n_photo = len(paths)  # Batch size
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
        if as_mask == True:
            image = np.where(image == 1, 1, 0)
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

