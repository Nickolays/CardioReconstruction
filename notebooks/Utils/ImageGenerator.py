import tensorflow as tf
import numpy as np
import pandas as pd
import os, cv2
from tqdm import tqdm


class ImageSegmentationGenerator(tf.keras.utils.Sequence):
    def __init__(self, paths:list, out_shape:tuple, batch_size:int, features:list, shuffle:bool=True, as_gray=True):  
        """
        paths: Maybe i should use the word 'dirs'
        out_shape: Needed shape, default is (512, 512)
        """
        # 
        self.batch_size = batch_size
        self.out_shape = out_shape
        self.shuffle = shuffle
        self.features = features
        # Obtain all names of photo to pandas DataFrame
        self.paths = pd.DataFrame({
            'images': self.read_path(paths[0]),  # image_paths,
            'masks': self.read_path(paths[1]),   # masks_paths
        })
        # 
        self.n = len(self.paths)

    @tf.function
    def read_path(self, dir):
        """
        Collect all paths from the dir
        """
        names = os.listdir(dir)
        # Concat all names with current dir
        paths = [ os.path.join(dir, name) for name in names ]

        return sorted(paths, reverse=False)
    
    @tf.function
    def read_image(self, path):
        """ Open image """
        # Load the raw data from the file as a string
        raw = tf.io.read_file(path)
        # Convert the compressed string to a 3D uint8 tensor
        image = tf.io.decode_image(raw, channels=1)
        """ TO DO: RESHAPE FOR THE FUTURE"""
        # if image.shape != self.out_shape:
        #     image = tf.image.resize(image, (self.out_shape[0], self.out_shape[1]))   #.numpy()

        return image
    
    @tf.function
    def read_images(self, paths):
        """
        
        """
        images = []
        for path in paths:
            image = self.read_image(path)
            images.append(image)

        return tf.stack(images, axis=0)
    
    @tf.function
    def read_masks(self, paths):
        """  """
        masks = []
        for path in paths:
            mask = self.read_image(path)
            mask = tf.squeeze(mask, axis=-1)
            # Here we'll keep segmentation for every class
            cmaps = []
            # Condition for masks
            for feature in self.features:
                cmaps.append(
                    # Check features in mask
                    tf.math.equal(mask, feature),
                    # tf.math.reduce_all(tf.math.equal(mask, feature), axis=-1)
                )
            # Transform to tensor with channels and change bool type to float
            mask = tf.cast(tf.stack(cmaps, axis=-1), dtype=tf.float16)
            # Save it in list
            masks.append(mask)

        return tf.stack(masks, axis=0)  # Save it like batch

    @tf.function
    def on_epoch_end(self):
        if self.shuffle:
            # 
            self.paths = self.paths.sample(self.n)

    def __getitem__(self, index):
        """  """
        # 
        img_paths = self.paths['images'][index * self.batch_size:(index + 1) * self.batch_size]
        msk_paths = self.paths['masks'][index * self.batch_size:(index + 1) * self.batch_size]
        # 
        X = self.read_images(img_paths.to_list())
        y = self.read_masks(msk_paths.to_list())

        return X, y

    def __len__(self):
        return self.n // self.batch_size
    

