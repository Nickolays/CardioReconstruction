import glob
import os

import pyvista as pv
from natsort import natsorted


def main():
    """ I'm change here glob to os lib for path reading """
    # Change these values
    data_folder = r'/home/suetin/Projects/VSCode/3DReconstructions/Final_models/Final_models_02'  # r'/path/to/your/vtkdata/'
    save_folder = r'e_pivox/datasets/'  # r'/path/to/your/data_save_folder'
    # 
    root, dirs, files = next(os.walk(data_folder))
    all_data_paths = natsorted(files)[1:]
    
    all_data_paths = natsorted(glob.glob(data_folder + '*.vtk'))
    print(all_data_paths)
    for file in all_data_paths:
        # print(file)
        # print(root, dirs)
        if dirs:
            file_path = os.path.join(root, dirs, file)
        else:
            file_path = os.path.join(root, file)
        case_mesh = pv.get_reader(file_path).read()
        case_number = file.split(os.sep)[-1].split('.')[0].split('_')[-1]
        filled_case_name = 'full_heart_mesh_' + str(int(case_number)).zfill(3)
        save_path = os.path.join(save_folder, 'heart_seg', 'heart_render', 'heart', filled_case_name)

        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        case_mesh.save(os.path.join(save_path, filled_case_name + '.vtk'), binary=True)
        print('Finished file:', file)


if __name__ == '__main__':
    main()
