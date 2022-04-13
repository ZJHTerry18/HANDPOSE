import pickle
import numpy as np
import os
from tqdm import tqdm
import os.path as osp

FPPRED_RES_PATH = 'D:\Workspace\LeapMotion\leapHandpose\leapHandpose\dataset_fptype\\fingerprint_single\p0\\result.pkl'
WRITE_PATH = 'D:/Workspace/LeapMotion/leapHandpose/leapHandpose/dataset_fptype/leap/p0_fp'
finger_dict = [4,9,14,19,24]

HANDPOSE_DICT =  ["0 1 0 0 0", "0 0 1 0 0", "0 0 0 1 0", "0 0 0 0 1", "1 1 0 0 0",
"0 1 1 0 0", "0 1 0 1 0", "0 1 0 0 1", "0 0 1 1 0", "0 0 0 1 1",
"0 1 1 1 0", "0 0 1 1 1", "0 1 1 1 1"]

assert osp.exists(WRITE_PATH)

with open(FPPRED_RES_PATH, 'rb') as f:
    data = pickle.load(f)
data = data.values

for i in tqdm(range(data.shape[0])):
    item = data[i]
    itpath = item[3].split('/')
    itlr = itpath[-3]
    itseq = itpath[-1][:-6]
    itposeid = int(itpath[-2])
    touch_ind = [i for i, x in enumerate(HANDPOSE_DICT[itposeid].split()) if x == '1']
    itfingerid = int(itpath[-1][-5])

    item[1] = -item[1]
    if itlr == 'right':
        item[0] = -item[0]
        item[2] = -item[2]
    
    with open(osp.join(WRITE_PATH, itlr, itseq + '.txt'), 'r') as fr:
        lines = fr.readlines()
    
    newline = ' '.join([str(round(item[i],3)) for i in [1,0,2]]) + '\n'
    lines[finger_dict[touch_ind[itfingerid]]] = newline
    with open(osp.join(WRITE_PATH, itlr, itseq + '.txt'), 'w') as fw:
        fw.writelines(lines)
    


