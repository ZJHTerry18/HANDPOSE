import os 
import os.path as osp
from tqdm import tqdm
from loguru import logger
import numpy as np

def load_data(path, datafile):
    with open(osp.join(path, datafile), 'r') as f:
        lines = f.readlines()
    dat = []
    title = datafile[:-4].split('_')
    title = [float(x) for x in title]
    dat.append(title)
    for line in lines[2:]:
        line = line.rstrip('\n').split()
        line = [float(x) for x in line]
        while len(line) < 3:
            line.append(0.0)
        dat.append(line)
    return np.array(dat)


def load_dataset(path):
    datafile_list = os.listdir(path)
    dataset = []
    logger.info('Loading dataset...')
    for datafile in tqdm(datafile_list):
        dat = load_data(path, datafile)
        dataset.append(dat)
    return dataset