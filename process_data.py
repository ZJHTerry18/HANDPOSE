import numpy as np
import os
import os.path as osp
import pickle
import re
from tqdm import tqdm
import glob

Folder = '/Extra/panzhiyu/HandData/'
files = os.listdir(Folder)
files.sort(key = lambda x: int(x.strip('.txt').replace('_','')))
SaveF = '/Extra/panzhiyu/CollectHD/'

namespace = ['t','i','m','r','l']

for f in tqdm(files):
    r = osp.join(Folder, f)
    with open(r, 'r') as handle:
        data = handle.readlines()
    saving_data = dict()
    finger_idx = -1
    for d in data:
        content = re.findall(r"\-?\d+\.?\d*",d)
        content = [float(x) for x in content]
        if len(content) == 5: # it is the conditional space
            saving_data['cond'] = np.array(content)
        elif len(content) == 2:           
            finger_idx = finger_idx + 1
            finger_info = [np.array(content)[np.newaxis,:]]
        elif len(content) == 1:
            finger_info.append(np.array(content)[np.newaxis,:])
        else:
            # saving the data once
            if 'finger_info' in locals():
                finger_info = np.concatenate(finger_info, axis=-1)
                saving_data[namespace[finger_idx]] = finger_info
                del finger_info
    
    save_name = osp.join(SaveF, f.replace('txt','pkl'))
    with open(save_name,'wb') as h:
        pickle.dump(saving_data,h)

        

