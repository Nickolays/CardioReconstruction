import numpy as np
import pandas as pd
import cv2, tqdm, os


def load_EchoNet(split_indx=0.15, contour='training'):
    """  """
    main_path = 'data/raw/EchoNet-Dynamic'
    label_df = pd.read_csv(os.path.join(main_path, 'VolumeTracings.csv'))
    label_df = label_df.sort_values(['FileName', 'Frame'])
    # Calc test index and choose needed part
    test_indx = int(label_df['FileName'].nunique() * split_indx)
    if contour == 'training':
        label_df[:test_indx]
    elif contour == 'testing':
        label_df[test_indx:]
    # 
    file_names = {}
    for filename in label_df['FileName'].unique():  # 
        file_names[filename] = os.path.join(os.path.join(main_path, 'Videos', filename))

    frames = []
    targets = []

    for i, (file_name, file_path) in tqdm(enumerate(file_names.items())):
        # print(file_name)
        cap = cv2.VideoCapture(file_path)
        df = label_df[label_df['FileName'] == file_name]
        frame_numbers = df['Frame'].unique()
        for f_num in frame_numbers:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_num-1)
            res, frame = cap.read()
            target = np.concat([df[['X1', 'Y1']], df[['X2', 'Y2']]], axis=0)
            # 
            frame, target = preprocessing_EchoNet(frame, target, input_shape=(512, 512))

            frames.append(frame)
            targets.append(target)

    return frames, targets

def preprocessing_EchoNet(frame, target, input_shape=(512, 512)):
    """ """
    
    mask = np.zeros_like(frame)

    points = np.array([[[xi, yi]] for xi, yi in target]).astype(np.int32)
    # points = points.sort_values(['x', 'y'])
    # print(points.shape)
    mask = cv2.fillPoly(mask, [points], color=[255,255,255])
    # mask = cv2.fillPoly(mask, [points], color=[255,255,255])
    # plt.imshow(mask);

    des = cv2.bitwise_not(mask)
    contour, hier = cv2.findContours(cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contour:
        cv2.drawContours(des, [cnt], 0, 255, -1)

    mask = cv2.bitwise_not(des)
    
    frame = cv2.resize(frame, input_shape)
    mask = cv2.resize(mask, input_shape)

    frame = frame / frame.max()
    frame = np.transpose(frame, (2, 0, 1)) 

    return frame, mask