import pickle
import numpy as np
import os
from tqdm import tqdm
import os.path as osp

FPPRED_RES_PATH = 'D:\Workspace\LeapMotion\leapHandpose\leapHandpose\dataset_fptype\\fingerprint_single\\p1\\result.pkl'
FPPRED_POS_PATH = 'D:\Workspace\LeapMotion\leapHandpose\leapHandpose\dataset_fptype\\fingerprint_single\\p1\\position_info_p1.pkl'
WRITE_PATH = 'D:/Workspace/LeapMotion/leapHandpose/leapHandpose/dataset_fptype/leap/p1_fp'
finger_dict = [4,9,14,19,24]
ppi = 225


HANDPOSE_DICT =  ["0 1 0 0 0", "0 0 1 0 0", "0 0 0 1 0", "0 0 0 0 1", "1 1 0 0 0",
"0 1 1 0 0", "0 1 0 1 0", "0 1 0 0 1", "0 0 1 1 0", "0 0 0 1 1",
"0 1 1 1 0", "0 0 1 1 1", "0 1 1 1 1"]

assert osp.exists(WRITE_PATH)

with open(FPPRED_RES_PATH, 'rb') as f:
    data = pickle.load(f)
data = data.values

with open(FPPRED_POS_PATH, 'rb') as f:
    posdata = pickle.load(f)


for i in tqdm(range(data.shape[0])):
    item = data[i]
    itpath = item[3].split('/')
    itlr = itpath[-3]
    itseq = itpath[-1][:-6]
    itposeid = int(itpath[-2])
    touch_ind = [i for i, x in enumerate(HANDPOSE_DICT[itposeid].split()) if x == '1']
    itfingerid = int(itpath[-1][-5])

    with open(osp.join(WRITE_PATH, itlr, itseq + '.txt'), 'r') as fr:
        lines = fr.readlines()
    
    item[1] = -item[1]
    if itlr == 'right':
        item[0] = -item[0]
        item[2] = -item[2]
    newangle = ' '.join([str(round(item[i],3)) for i in [1,0,2]]) + '\n'
    lines[finger_dict[touch_ind[itfingerid]]] = newangle

    # if itpath[-1][:-4] in posdata[itpath[-4]][itlr][itpath[-2]].keys():
    #     bbox = posdata[itpath[-4]][itlr][itpath[-2]][itpath[-1][:-4]]['bbox']
    #     tx = float(bbox[0] + bbox[2] / 2) / ppi * 2.54 * 10
    #     ty = float(bbox[1] + bbox[3] / 2) / ppi * 2.54 * 10
    #     newpos = ' '.join([str(round(-tx,3)), '0.0', str(round(ty,3))]) + '\n'
    #     lines[finger_dict[touch_ind[itfingerid]] + 1] = newpos

    with open(osp.join(WRITE_PATH, itlr, itseq + '.txt'), 'w') as fw:
        fw.writelines(lines)
    


