from json import load
import os
import numpy as np
from dataload import load_dataset
from loguru import logger
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

DATA_PATH = '../dataset/train'

NUM_POSE = 26
HANDPOSE_DICT = ["1 0 0 0 0", "0 1 0 0 0", "0 0 1 0 0", "0 0 0 1 0", "0 0 0 0 1",
	"1 1 0 0 0", "0 1 1 0 0", 
	"1 0 1 0 0", "1 0 0 1 0", "1 0 0 0 1", "0 1 0 1 0", "0 1 0 0 1", "0 0 1 1 0", "0 0 0 1 1",
	"1 1 1 0 0", "0 1 1 1 0", "0 0 1 1 1", 
	"1 1 0 1 0", "1 0 1 1 0", "1 0 0 1 1", "1 1 0 0 1",
	"0 1 1 1 1", "1 0 1 1 1",
	"1 1 0 1 1", "1 1 1 1 0", 
	"1 1 1 1 1"]

def plot_hist(x, xtype, bins, ranges):
    plt.figure()
    n, bins, patches = plt.hist(x, bins=bins, range=ranges, density=True)
    plt.title("%s distribution" % (xtype))
    plt.show()

if __name__ == "__main__":
    dataset = load_dataset(DATA_PATH)
    ind = [4,9,14,19,24]
    
    pitchlist = []
    yawlist = []
    rolllist = []
    for x in dataset:
        pose_id = int(x[0][0])
        touch_ind = [i for i, x in enumerate(HANDPOSE_DICT[pose_id].split()) if x == '1']
        for i in touch_ind:
            pitch = x[ind[i]][0]
            yaw = x[ind[i]][1]
            roll = x[ind[i]][2]
            pitchlist.append(pitch)
            yawlist.append(yaw)
            rolllist.append(roll)
    
    plot_hist(np.array(pitchlist), xtype='pitch', bins=18, ranges=(-90,90))
    plot_hist(np.array(yawlist), xtype='yaw', bins=18, ranges=(-180,180))
    plot_hist(np.array(rolllist), xtype='roll', bins=18, ranges=(-90,90))
    
