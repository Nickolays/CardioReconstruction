import torch
import torch.nn as nn
import cv2, os
import numpy as np
from typing import List
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


class HeartLoader(torch.utils.data.Dataset):
    def __init__(self, imgs_paths, masks_paths, image_size=(512, 512)):
        """ Simple image torch Dataset for images and masks in photo format """
        assert len(imgs_paths) == len(masks_paths)

        assert isinstance(imgs_paths, List)
        assert isinstance(imgs_paths[0], str)
        assert os.path.exists(imgs_paths[0])

        assert isinstance(masks_paths, List)
        assert isinstance(masks_paths[0], str)
        assert os.path.exists(masks_paths[0])

        # Save
        self.imgs_paths = imgs_paths
        self.masks_paths = masks_paths
        self.image_size = image_size
        
    def choose_mask(self, mask):
        """ We need only 1 type """
        return np.where(mask == 255, 1., 0.).astype(np.float32)
            
    def __getitem__(self, index):

        img = cv2.imread(self.imgs_paths[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 
        if img.shape[:2] != self.image_size:
            img = cv2.resize(img, dsize=(self.image_size))
        img = self.preprocess(img)
        # img = self.normalize(img)
        img = np.transpose(img, (2, 0, 1))


        mask = cv2.imread(self.masks_paths[index], 0)   # 0 means Like a grayscale
        if mask.shape[:2] != self.image_size:
            mask = cv2.resize(mask, dsize=(self.image_size))
        # mask = preprocess(mask)
        mask = self.choose_mask(mask)
        # mask = self.posprocess(mask)
        mask = np.expand_dims(mask, 0)
        
        # mask = mask.squeeze(0)

        return img.astype(np.float32), mask.astype(np.float32)

    def __len__(self):
        return len(self.imgs_paths)

    def preprocess(self, data):
        # Resize or something else the same
        return data / 255

    def posprocess(self, data):
        # mask = mask.squeeze(0)
        pass


# Restore images to suitable images of opencv style
def ImgForPlot(img):
    # img = np.einsum('ijk->jki', img)
    # img = (127.5*(img+1)).astype(np.uint8)
    try:
        img = img.cpu().detach().numpy()
        return np.transpose(img, (1, 2, 0))
    except:
        print("Already numpy.array")
        return np.transpose(img, (1, 2, 0))