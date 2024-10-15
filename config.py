# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

from easydict import EasyDict as edict
import os

__C                                         = edict()
cfg                                         = __C

#
# Dataset Config
#
__C.DATASETS = edict()
__C.DATASETS.HEARTSEG = edict()
__C.DATASETS.HEARTSEG.IMG_ROOT = 'data/train/Heart/heart_seg/slices'
__C.DATASETS.HEARTSEG.TAXONOMY_FILE_PATH = 'data/train/HeartSeg_new.json'
__C.DATASETS.HEARTSEG.RENDERING_PATH = os.path.join(__C.DATASETS.HEARTSEG.IMG_ROOT, '%s/*.png')
__C.DATASETS.HEARTSEG.VOXEL_PATH = f'data/train/Heart/heart_seg/voxels/%s/model.npy'

__C.DATASETS.SHAPENET                       = edict()
__C.DATASETS.SHAPENET.TAXONOMY_FILE_PATH    = 'data/train/ShapeNet.json'
__C.DATASETS.SHAPENET.RENDERING_PATH        = '/home/hzxie/Datasets/ShapeNet/ShapeNetRendering/%s/%s/rendering/%02d.png'
__C.DATASETS.SHAPENET.VOXEL_PATH            = '/home/hzxie/Datasets/ShapeNet/ShapeNetVox32/%s/%s/model.binvox'
# __C.DATASETS.SHAPENET.TAXONOMY_FILE_PATH  = './datasets/PascalShapeNet.json'
# __C.DATASETS.SHAPENET.RENDERING_PATH      = '/home/hzxie/Datasets/ShapeNet/PascalShapeNetRendering/%s/%s/render_%04d.jpg'

# __C.DATASETS.PASCAL3D                       = edict()
# __C.DATASETS.PASCAL3D.TAXONOMY_FILE_PATH    = './datasets/Pascal3D.json'
# __C.DATASETS.PASCAL3D.ANNOTATION_PATH       = '/home/hzxie/Datasets/PASCAL3D/Annotations/%s_imagenet/%s.mat'
# __C.DATASETS.PASCAL3D.RENDERING_PATH        = '/home/hzxie/Datasets/PASCAL3D/Images/%s_imagenet/%s.JPEG'
# __C.DATASETS.PASCAL3D.VOXEL_PATH            = '/home/hzxie/Datasets/PASCAL3D/CAD/%s/%02d.binvox'
# __C.DATASETS.PIX3D                          = edict()
# __C.DATASETS.PIX3D.TAXONOMY_FILE_PATH       = './datasets/Pix3D.json'
# __C.DATASETS.PIX3D.ANNOTATION_PATH          = '/home/hzxie/Datasets/Pix3D/pix3d.json'
# __C.DATASETS.PIX3D.RENDERING_PATH           = '/home/hzxie/Datasets/Pix3D/img/%s/%s.%s'
# __C.DATASETS.PIX3D.VOXEL_PATH               = '/home/hzxie/Datasets/Pix3D/model/%s/%s/%s.binvox'

#
# Dataset
#
__C.DATASET                                 = edict()
__C.DATASET.MEAN                            = [0.5, 0.5, 0.5]
__C.DATASET.STD                             = [0.5, 0.5, 0.5]
__C.DATASET.TRAIN_DATASET                   = 'HeartSeg'  # 'ShapeNet'
__C.DATASET.TEST_DATASET                    = 'HeartSeg'  # 'ShapeNet'
# __C.DATASET.TEST_DATASET                  = 'Pascal3D'
# __C.DATASET.TEST_DATASET                  = 'Pix3D'

#
# Common
#
__C.CONST                                   = edict()
__C.CONST.DEVICE                            = '0'
__C.CONST.RNG_SEED                          = 0
__C.CONST.IMG_W                             = 256       # Image width for input
__C.CONST.IMG_H                             = 256       # Image height for input
__C.CONST.N_VOX                             = 64   # 32
__C.CONST.BATCH_SIZE                        = 8
__C.CONST.N_VIEWS_RENDERING                 = 9         # Dummy property for Pascal 3D
__C.CONST.CROP_IMG_W                        = 128       # Dummy property for Pascal 3D
__C.CONST.CROP_IMG_H                        = 128       # Dummy property for Pascal 3D

#
# Directories
#
__C.DIR                                     = edict()
__C.DIR.OUT_PATH                            = 'results/Pix2Vox/outputs'
__C.DIR.IOU_SAVE_PATH = os.path.join(__C.DIR.OUT_PATH, 'iou_scores.xlsx')
__C.DIR.RANDOM_BG_PATH                      = 'home/hzxie/Datasets/SUN2012/JPEGImages'
__C.DIR.CHECKPOINTS                         = 'results/Pix2Vox'

#
# Network
#
__C.NETWORK = edict()
__C.NETWORK.LEAKY_VALUE                     = .2
__C.NETWORK.TCONV_USE_BIAS                  = False
__C.NETWORK.LOSS_FUNC                       = 'BCELoss'
__C.NETWORK.MODEL_SIZE                      = 128    # 32
__C.NETWORK.USE_EP2V                        = True
__C.NETWORK.USE_REFINER                     = True
__C.NETWORK.USE_MERGER                      = True

#
# Training
#
__C.TRAIN                                   = edict()
__C.TRAIN.RESUME_TRAIN                      = False
__C.TRAIN.NUM_WORKER                        = 4             # number of data workers
__C.TRAIN.NUM_EPOCHS                        = 50
__C.TRAIN.BRIGHTNESS                        = .4
__C.TRAIN.CONTRAST                          = .4
__C.TRAIN.SATURATION                        = .4
__C.TRAIN.NOISE_STD                         = .1
__C.TRAIN.RANDOM_BG_COLOR_RANGE             = [[225, 255], [225, 255], [225, 255]]
__C.TRAIN.POLICY                            = 'adam'        # available options: sgd, adam
__C.TRAIN.EPOCH_START_USE_REFINER           = 0
__C.TRAIN.EPOCH_START_USE_MERGER            = 0
__C.TRAIN.ENCODER_LEARNING_RATE             = 1e-3
__C.TRAIN.DECODER_LEARNING_RATE             = 1e-3
__C.TRAIN.REFINER_LEARNING_RATE             = 1e-3
__C.TRAIN.MERGER_LEARNING_RATE              = 1e-4
__C.TRAIN.ENCODER_LR_MILESTONES             = [150]
__C.TRAIN.DECODER_LR_MILESTONES             = [150]
__C.TRAIN.REFINER_LR_MILESTONES             = [150]
__C.TRAIN.MERGER_LR_MILESTONES              = [150]
__C.TRAIN.BETAS                             = (.9, .999)
__C.TRAIN.MOMENTUM                          = .9
__C.TRAIN.GAMMA                             = .5
__C.TRAIN.SAVE_FREQ                         = 10            # weights will be overwritten every save_freq epoch
__C.TRAIN.UPDATE_N_VIEWS_RENDERING          = False

#
# Testing options
#
__C.TEST = edict()
__C.TEST.RANDOM_BG_COLOR_RANGE = [[240, 240], [240, 240], [240, 240]]
__C.TEST.VOXEL_THRESH = [.2, .3, .4, .5, .6, .7, .8, .9]
__C.TEST.TEST_NETWORK = False
__C.TEST.VOL_OR_RENDER_SAVE = 'volume'