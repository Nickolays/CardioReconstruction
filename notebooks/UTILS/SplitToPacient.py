import os, skimage

main_path = r'/home/suetin/Projects/VSCode/ComputerVision/HeartSegmentation/test/images/images'
out_path = r'/home/suetin/Projects/VSCode/Apps/Echo_app/Data/CANVAS'

print(len(os.listdir(main_path)))  # 400

for i in range(0, 400, 4):
    # Read images