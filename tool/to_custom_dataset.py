# from 26-pose dataset to 10-pose dataset in TouchPose
import os
import shutil
from tqdm import tqdm

READ_PATH = r'D:\Workspace\HANDPOSE\dataset\train_new'
WRITE_PATH = r'D:\Workspace\HANDPOSE\dataset\train_type_new'

if not os.path.exists(WRITE_PATH):
    os.makedirs(WRITE_PATH)

files = os.listdir(READ_PATH)
convert_dict = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 10:6, 11:7, 12:8, 13:9, 15:10, 16:11, 21:12}

for file in tqdm(files):
    nameparts = file.split('_')
    poseid = int(nameparts[0])
    if poseid in convert_dict.keys():
        nameparts[0] = str(convert_dict[poseid])
        newfile = '_'.join(nameparts)
        shutil.copyfile(os.path.join(READ_PATH, file), os.path.join(WRITE_PATH, newfile))