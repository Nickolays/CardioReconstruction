from time import time
import logging
import torch
import numpy as np
import pandas as pd
import os, json

from src.scripts.data_loaders_p2v import DATASET_LOADER_MAPPING, DatasetType
from src.scripts import data_transforms


if torch.cuda.is_available():
    device = 'cuda'
else: device = 'cpu'

# Set up loss functions
loss_func = torch.nn.BCELoss()   # DiceBCELoss, IoULoss


def test_net(cfg,
             epoch_idx=-1,
             test_data_loader=None,
             encoder=None,
             decoder=None,
             refiner=None,
             merger=None):

    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # Load taxonomies of dataset
    taxonomies = []
    with open(cfg.DATASETS[cfg.DATASET.TEST_DATASET.upper()].TAXONOMY_FILE_PATH, encoding='utf-8') as file:
        taxonomies = json.loads(file.read())
    taxonomies = {t['taxonomy_id']: t for t in taxonomies}

    # Set up data loader
    if test_data_loader is None:
        # Set up data augmentation
        IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
        CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
        test_transforms = data_transforms.Compose([
            # data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
            # utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
            data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
            data_transforms.ToTensor(),
        ])

        # dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
        dataset_loader = DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
        test_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader.get_dataset(
                DatasetType.TEST, cfg.CONST.N_VIEWS_RENDERING, test_transforms),
            batch_size=1,
            num_workers=cfg.CONST.NUM_WORKER,
            pin_memory=True,
            shuffle=False)

    # Set up loss functions
    # loss_func = utils.helpers.get_loss_function(cfg)
    # Testing loop
    n_samples = len(test_data_loader)
    test_iou = dict()

    # Switch models to evaluation mode
    encoder.eval()
    decoder.eval()
    refiner.eval()
    merger.eval()

    for sample_idx, (taxonomy_id, sample_name, rendering_images, ground_truth_volume) in enumerate(test_data_loader):
        taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
        sample_name = sample_name[0]

        with torch.no_grad():
            # Get data from data loader
            rendering_images = rendering_images.cuda()
            ground_truth_volume = ground_truth_volume.cuda()

            # Test the encoder, decoder, refiner and merger
            image_features = encoder(rendering_images)
            raw_features, generated_volume = decoder(image_features)

            if cfg.NETWORK.USE_MERGER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_MERGER:
                generated_volume = merger(raw_features, generated_volume)
            else:
                generated_volume = torch.mean(generated_volume, dim=1)
            encoder_loss = loss_func(generated_volume, ground_truth_volume) * 10

            if cfg.NETWORK.USE_REFINER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_REFINER:
                generated_volume = refiner(generated_volume)
                refiner_loss = loss_func(generated_volume, ground_truth_volume) * 10
            else:
                refiner_loss = encoder_loss

            # IoU per sample
            sample_iou = []
            for th in cfg.TEST.VOXEL_THRESH:
                _volume = torch.ge(generated_volume, th).float()
                intersection = torch.sum(_volume.mul(ground_truth_volume)).float()
                union = torch.sum(torch.ge(_volume.add(ground_truth_volume), 1)).float()
                sample_iou.append((intersection / union).item())

            # IoU per taxonomy
            if taxonomy_id not in test_iou:
                test_iou[taxonomy_id] = {'n_samples': 0, 'iou': []}
            test_iou[taxonomy_id]['n_samples'] += 1
            test_iou[taxonomy_id]['iou'].append(sample_iou)

            # # Append generated volumes to TensorBoard
            # if cfg.TEST.VOL_OR_RENDER_SAVE.lower() == 'render':
            #     if test_writer and sample_idx < cfg.CONST.TEST_SAVE_NUMBER:
            #         # Volume Visualization
            #         rendering_views = utils.helpers.get_volume_views(generated_volume.cpu().numpy())
            #         test_writer.add_image('Model%02d/Reconstructed' % sample_idx, rendering_views, epoch_idx)
            #         rendering_views = utils.helpers.get_volume_views(ground_truth_volume.cpu().numpy())
            #         test_writer.add_image('Model%02d/GroundTruth' % sample_idx, rendering_views, epoch_idx)
            # elif cfg.TEST.VOL_OR_RENDER_SAVE.lower() == 'volume':
            #     # if test_writer and sample_idx < cfg.CONST.TEST_SAVE_NUMBER:
            #     utils.helpers.save_test_volumes_as_np(cfg, generated_volume, sample_idx, epoch_idx)
            # else:
            #     raise Exception(
            #         '[FATAL] %s Invalid input for save format %s. voxels' % (dt.now(), cfg.TEST.VOL_OR_RENDER_SAVE))

            # Print sample loss and IoU
            logging.info('Test[%d/%d] Taxonomy = %s Sample = %s EDLoss = %.4f RLoss = %.4f IoU = %s' %
                         (sample_idx + 1, n_samples, taxonomy_id, sample_name, encoder_loss.item(),
                          refiner_loss.item(), ['%.4f' % si for si in sample_iou]))

    if cfg.DIR.IOU_SAVE_PATH:
        df = pd.DataFrame(
            np.hstack((test_iou[taxonomy_id]['iou'], np.atleast_2d(np.max(test_iou[taxonomy_id]['iou'], axis=1)).T)),
            columns=[*cfg.TEST.VOXEL_THRESH, 'max_iou'])
        # writer = pd.ExcelWriter(, engine='xlsxwriter')
        # df.to_excel(writer, index=False)
        # writer.save()
        # df.to_excel(cfg.DIR.IOU_SAVE_PATH, index=False)
        

    mean_iou = []
    for taxonomy_id in test_iou:
        test_iou[taxonomy_id]['iou'] = np.mean(test_iou[taxonomy_id]['iou'], axis=0)
        mean_iou.append(test_iou[taxonomy_id]['iou'] * test_iou[taxonomy_id]['n_samples'])
    mean_iou = np.sum(mean_iou, axis=0) / n_samples

    # Print header
    print('============================ TEST RESULTS ============================')
    print('Taxonomy', end='\t')
    print('#Sample', end='\t')
    print('Baseline', end='\t')
    for th in cfg.TEST.VOXEL_THRESH:
        print('t=%.2f' % th, end='\t')
    print()
    # Print body
    for taxonomy_id in test_iou:
        print('%s' % taxonomies[taxonomy_id]['taxonomy_name'].ljust(8), end='\t')
        print('%d' % test_iou[taxonomy_id]['n_samples'], end='\t')
        if 'baseline' in taxonomies[taxonomy_id]:
            # print('%.4f' % taxonomies[taxonomy_id]['baseline']['%d-view' % cfg.CONST.N_VIEWS_RENDERING], end='\t\t')
            print('Ignoring baseline')
        else:
            print('N/a', end='\t\t')

        for ti in test_iou[taxonomy_id]['iou']:
            print('%.4f' % ti, end='\t')
        print()
    # Print mean IoU for each threshold
    print('Overall ', end='\t\t\t\t')
    for mi in mean_iou:
        print('%.4f' % mi, end='\t')
    print('\n')

    # Add testing results to TensorBoard
    max_iou = np.max(mean_iou)

    return max_iou