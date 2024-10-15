import torch
import torch.nn as nn
import cv2, os
import numpy as np
from typing import List
from natsort import natsorted
import warnings
warnings.filterwarnings('ignore')


def get_image_filepaths(main_path, img_format):
    
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
    
    return natsorted(filepaths)


class HeartLoader(torch.utils.data.Dataset):
    def __init__(self, imgs_paths:list, masks_paths:list, labels:list, image_size=(512, 512)):
        """ Simple image torch Dataset for images and masks in photo format 
        
         Current version only for CAMUS dataset 
        """
        assert len(imgs_paths) == len(masks_paths)

        assert isinstance(imgs_paths, List)
        assert isinstance(imgs_paths[0], str)
        assert os.path.exists(imgs_paths[0])

        assert isinstance(masks_paths, List)
        assert isinstance(masks_paths[0], str)
        assert os.path.exists(masks_paths[0])

        assert isinstance(labels, list)

        # Save
        self.imgs_paths = imgs_paths
        self.masks_paths = masks_paths
        self.labels = labels  # list of label's indecies.   1 - LV, 2 - LA, 3 - RV, 4 - RA
        self.image_size = image_size

        # self.label_names = [for i in ]
        
    def choose_mask(self, mask):
        """ We need only 1 type 
         We choose a new format of """
        # assert isinstance(mask_class, str)
        # assert mask_class in (["LV", "LA", "RV", "RA"])

        # if mask_class == "LV":
        mask = np.where(mask > 0, 1., 0.).astype(np.float32)
        # elif mask_class == "RV":
        #     mask = np.where(mask == 255, 1., 0.).astype(np.float32)
        return mask
            
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


class LeftCamusDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 image_paths:list, 
                 LA_masks_paths:list, 
                 # image_size=(512, 512), 
                 label_smooth=None, 
                 transforms=None):
        """ Temporarily Simple image torch.Dataset for images and masks(LV+LA)
         Current version only for CAMUS dataset. We have 2 path to masks (LV and LA).
        """
        self.image_paths = image_paths
        self.la_paths = LA_masks_paths
        # self.img_size = image_size
        self.label_smooth = label_smooth
        self.transforms = transforms

    def __getitem__(self, index):
        """ Тут тупняк в том, что в LV есть ещё 1 датасет (EchoNet), но в нём есть и камус тоже, 
        поэтому такие костыли """
        img_path = self.image_paths[index]
        la_path = self.la_paths[index]
        # 
        image = self.load_image(img_path, as_mask=False)
        masks = self.concat_mask(la_path)
        # Apply transforms
        if self.transforms:
            # image, masks = self.transforms(image, masks)  # It doesn't work, I don't know why
            image = self.transforms(image)
            masks = self.transforms(masks)
        return image, masks
    
    def __len__(self):
        return len(self.image_paths)
    
    def load_image(self, path, as_mask=False):
        if as_mask:
            img = cv2.imread(path, 0)
            if self.label_smooth: img = np.where(img > 0, 1-self.label_smooth, 0.)
            else: img = np.where(img > 0, 1., 0.)
        else:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255
        return img.astype(np.float32)

    def concat_mask(self, la_path):
        """ Load both images and concat them """
        assert os.path.exists(la_path)
        # Read LA mask
        la_mask = self.load_image(la_path, as_mask=True)
        # Replace path to LV and load mask
        lv_mask = self.load_image(la_path.replace("echoLA", "echoLV"), as_mask=True)
        if la_mask.ndim == 2:
            la_mask = np.expand_dims(la_mask, -1)
            lv_mask = np.expand_dims(lv_mask, -1)
        return np.concat([lv_mask, la_mask], axis=-1)


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