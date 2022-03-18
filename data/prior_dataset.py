from concurrent.futures import process
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.utils.data as data
import glob
import os

namespace = ['t','i','m','r','l']

class ivg_HD_naive(data.Dataset):
    def __init__(self, folder, is_training):
        super(ivg_HD_naive,self).__init__()
        self.hds = glob.glob(os.path.join(folder,'*.pkl'))
        self.len = len(self.hds)
        self.gap = 0
        # train_len = int(total_len * 0.7)
        # if is_training:
        #     self.len = train_len
        #     self.gap = 0
        # else:
        #     self.len = total_len - train_len
        #     self.gap = train_len
    
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        t_idx = self.gap + index
        file = self.hds[t_idx]
        with open(file, 'rb') as f:
            readin_data = pickle.load(f)
        # split the data
        cond = torch.from_numpy(readin_data['cond'])
        curl = torch.from_numpy(readin_data['curl']) # add the curl paras
        finger_ang = torch.zeros(len(namespace), 4) # 4 is the angle dim per finger
        for idx, name in enumerate(namespace):
            finger_ang[idx:idx+1,:] = torch.from_numpy(readin_data[name] * np.pi / 180)
        # equally process
        finger_ang_cor_s = torch.sin(finger_ang.reshape(-1,1))
        finger_ang_cor_c = torch.cos(finger_ang.reshape(-1,1))
        finger_ang_cor = torch.cat([finger_ang_cor_s, finger_ang_cor_c], dim = -1)
        return cond, curl, finger_ang_cor, finger_ang

if __name__ == '__main__':
    folder = '/Extra/panzhiyu/CollectHD'
    testD = ivg_HD_naive(folder, False)
    datalen = len(testD)
    testcond, testinfo, _ = testD[int(datalen * 0.3)]

    print('lls')

