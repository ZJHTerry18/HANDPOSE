# from 26-pose dataset to 10-pose dataset in TouchPose
import os
import shutil
from tqdm import tqdm

READ_PATH = 'D:\Workspace\Handpose\dataset\\train'
WRITE_PATH = 'D:\Workspace\Handpose\dataset\\train_10'

if not os.path.exists(WRITE_PATH):
    os.makedirs(WRITE_PATH)

files = os.listdir(READ_PATH)
convert_dict = {1:0, 5:1, 6:2, 10:3, 11:4, 14:5, 17:6, 20:7, 21:8, 25:9}

for file in tqdm(files):
    nameparts = file.split('_')
    poseid = int(nameparts[0])
    if poseid in convert_dict.keys():
        nameparts[0] = str(convert_dict[poseid])
        newfile = '_'.join(nameparts)
        shutil.copyfile(os.path.join(READ_PATH, file), os.path.join(WRITE_PATH, newfile))