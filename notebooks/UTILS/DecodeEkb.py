import numpy as np
import os
import matplotlib.pyplot as plt
import skimage


path = r'/home/suetin/Projects/VSCode/Apps/Echo_app/Data/12/12_0m'

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

    return image_paths, descriptions_paths


imgs, descripts = get_img_txt_path(path)


# Read Image
img = skimage.io.imread(imgs[0])
# Read txt
with open(descripts[0]) as f:
    # lines = f.read()
    lines2 = f.readlines()

# lines_np = np.loadtxt(descripts[0], dtype='str', delimiter='  ', skiprows=1, encoding='utf-8')
lines_np = np.genfromtxt(descripts[0], delimiter='  ', dtype='str')
print(lines_np.shape)
print(lines_np[0])
print(lines2[0])

arr = np.array([line.split(' ') for line in lines_np])
print(arr.shape)
print(arr[0])




# data = np.genfromtxt(descripts[0], dtype=str, delimiter="  ")
# print(data.shape)
# print(lines)
# print(lines2)
# print('\n', img.shape)
# print(len(lines))
# print(len(lines2))

# for line in lines[:15]:
#     print(line)

# plt.imshow(img)
#plt.show()