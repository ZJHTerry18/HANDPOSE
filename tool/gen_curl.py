import os
import numpy as np
from tqdm import tqdm
from dataload import load_dataset, load_data
from decomp import curl_type

DATA_PATH = '../dataset/test'
CURLDATA_PATH = '../dataset/test_curl'

if __name__ == "__main__":
    if not os.path.exists(CURLDATA_PATH):
        os.makedirs(CURLDATA_PATH)
    
    files = os.listdir(DATA_PATH)
    for file in tqdm(files):
        with open(os.path.join(DATA_PATH, file), 'r') as f:
            lines = f.readlines()
        dat = load_data(DATA_PATH, file)
        c = curl_type(dat)
        curlinfo = str(c) + '\n'
        lines.insert(2, curlinfo)
        with open(os.path.join(CURLDATA_PATH, file), 'w') as f:
            f.writelines(lines)