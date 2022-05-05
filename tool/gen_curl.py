import os
import numpy as np
from tqdm import tqdm
from dataload import load_dataset, load_data

DATA_PATH = '../dataset/test_new/txts'
CURLDATA_PATH = '../dataset/train_new_curl'

def curl_type(x):
    ind = [(2,3),(7,8),(12,13),(17,18),(22,23)]
    curl_num = 0
    for i1,i2 in ind:
        angles = x[i1-1][1] + x[i1][0] + x[i2][0]
        if angles > 120:
            curl_num += 1

    if curl_num == 0:
        return 0
    else:
        return 1


if __name__ == "__main__":
    if not os.path.exists(CURLDATA_PATH):
        os.makedirs(CURLDATA_PATH)
    
    files = os.listdir(DATA_PATH)
    curl_num = 0
    total_num = 0
    for file in tqdm(files):
        pose_id = int(file.split('_')[0])
        if pose_id in [0,1,5,6,14,15,16,21]:
            total_num += 1
        with open(os.path.join(DATA_PATH, file), 'r') as f:
            lines = f.readlines()
        dat = load_data(DATA_PATH, file)
        c = curl_type(dat)
        curlinfo = str(c) + '\n'
        lines.insert(2, curlinfo)
        curl_num += c
        # with open(os.path.join(CURLDATA_PATH, file), 'w') as f:
        #     f.writelines(lines)
    print(curl_num, total_num)