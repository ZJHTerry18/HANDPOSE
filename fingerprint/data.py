import os
import os.path as osp
import cv2
import numpy as np
import pickle
import copy
import torch
from tqdm import tqdm

LOAD_PATH = 'D:\Workspace\LeapMotion\leapHandpose\leapHandpose\\anglepredict\\fingerprint_type'
NUM_POSE = 10
IMS_PER_POSE = 150
N = 5

ids = ['p0']
hands = ['left', 'right']
poses = [str(i) for i in range(NUM_POSE)]

def get_neighbor(imgfilesp, imgseq, n):
    nbr_filesp = copy.deepcopy(imgfilesp)
    nbrseq = max(0, min(imgseq + n, IMS_PER_POSE - 1))
    nbr_filesp[1] = str(nbrseq).zfill(3)
    return nbr_filesp

def get_imgdata(path, imgfile):
    # img = cv2.imread(osp.join(path, imgfile))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(np.max(img))
    imgfile_sp = imgfile.split('_')
    imgseq = int(imgfile_sp[1])
    impath_sequence = [None] * N
    i = 0
    for n in range(-(N-1) // 2, (N-1) // 2 + 1):
        nbrfile = get_neighbor(imgfile_sp, imgseq, n)
        nbrfile = '_'.join(nbrfile)
        impath_sequence[i] = osp.join(path, nbrfile)
        i += 1
    
    return impath_sequence

    

if __name__ == "__main__":
    datapkg = []
    for id in ids:
        for hand in hands:
            for pose in poses:
                path = osp.join(LOAD_PATH, id, hand, pose)
                imgfiles = os.listdir(path)
                for file in tqdm(imgfiles):
                    path_sequence = get_imgdata(path, file)
                    datapkg.append({'path':osp.join(path, file), 'seqpath':path_sequence})
    pklname = 'fingerprint.pkl'
    with open(pklname, 'wb') as f:
        pickle.dump(datapkg, f)

