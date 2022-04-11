import pickle
import numpy as np
import os
from tqdm import tqdm
import os.path as osp

WRITE_PATH = 'D:/Workspace/LeapMotion/leapHandpose/leapHandpose/dataset_fp/leap/p0_fppred2'
finger_dict = {'t':4, 'i':9, 'm':14, 'r':19, 'p':24}

assert osp.exists(WRITE_PATH)

with open('result.pkl', 'rb') as f:
    data = pickle.load(f)
data = data.values

for i in tqdm(range(data.shape[0])):
    item = data[i]
    itpath = item[3].split('/')
    itlr = itpath[-3]
    itseq = itpath[-1][:-6]
    itfingerid = itpath[-1][-5]

    if itlr == 'right':
        item[1] = -item[1]
        item[2] = -item[2]
    
    with open(osp.join(WRITE_PATH, itlr, itseq + '.txt'), 'r') as fr:
        lines = fr.readlines()
    
    newline = ' '.join([str(round(item[i],3)) for i in range(3)]) + '\n'
    lines[finger_dict[itfingerid]] = newline
    with open(osp.join(WRITE_PATH, itlr, itseq + '.txt'), 'w') as fw:
        fw.writelines(lines)
    


