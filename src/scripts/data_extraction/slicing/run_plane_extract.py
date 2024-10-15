import glob
import os

from natsort import natsorted

from config import cfg
from plane_extract import run_slice_extraction
from multiprocessing import Pool, process
from itertools import repeat
import time


def main():

    paths = [os.path.join(cfg.DATA_IN.DATA_FOLDER, dir_name) for dir_name in os.listdir(cfg.DATA_IN.DATA_FOLDER) 
                                                                if dir_name.split('.')[-1] == 'vtk']
    all_data_paths = natsorted(paths)  # natsorted(glob.glob( + '*.vtk'))
    start = time.perf_counter()
    case_names = [file.split(os.sep)[-1].split('.')[0] for file in all_data_paths]

    save_paths = [os.path.join(cfg.DATA_IN.SAVE_FOLDER, 'heart_seg', 'heart_render', 'heart', case_name) for case_name
                  in
                  case_names]

    for path in save_paths:
        if not os.path.exists(path):
            os.makedirs(path)
    
    for data_path, save_path in zip(all_data_paths, save_paths):
        run_slice_extraction(cfg,  data_path, save_path, cfg.DATA_OUT.FANCY_PLOT)
    # with Pool(cfg.PARAMETERS.NUM_WORKERS) as pool:
    #     pool.starmap(run_slice_extraction,
    #                  zip(repeat(cfg), all_data_paths, save_paths, repeat(cfg.DATA_OUT.FANCY_PLOT)))

    finish = time.perf_counter()
    print(f'Finished in {finish - start} second(s)')


if __name__ == '__main__':
    main()

