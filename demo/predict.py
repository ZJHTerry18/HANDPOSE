import sys
sys.path.append('..')
sys.path.append('../methods')
import os
import os.path as osp
import numpy as np
from tqdm import tqdm
from loguru import logger
import cv2
import pickle
import argparse

#local import
from methods.knn import DATA_PATH, DATASET_PATH, HANDPOSE_DICT, kNNsearch, knncfg
from tool.dataload import load_dataset
from tool.handmodel import get_skeleton_from_data, transform_to_global, xzy_to_xyz, inverse_kinematics
from tool.visualize import vis, vis_pred

parser = argparse.ArgumentParser()
parser.add_argument('--id', type=str)
args = parser.parse_args()

# video configuration
FRAME_PATH = 'videos/test1'
VIDEO_FPS = 30.0

# handpose data configuration
HANDPOSE_DICT = knncfg.HANDPOSE_DICT
NUM_POSE = knncfg.NUM_POSE
DATA_FPS = 7.5
DATA_PATH = '../dataset/demo_type_test/pd/' + args.id
DATASET_PATH = knncfg.DATASET_PATH
SAVE_DIR = 'output/pd/' + args.id


def pred_interpolate(pred_res, gap):
    for i in range(len(pred_res)):
        if i % gap != 0:
            prev_dat = pred_res[(i // gap) * gap]
            next_dat = pred_res[(i // gap + 1) * gap]
            w_prev = 1.0 - float((i % gap)) / float(gap)
            w_next = 1.0 - w_prev
            pred_res[i] = {}
            pred_res[i]['data'] = np.zeros_like(prev_dat['data'])
            pred_res[i]['data'][0] = prev_dat['data'][0]
            pred_res[i]['data'][1:] = w_prev * prev_dat['data'][1:] + w_next * next_dat['data'][1:]
            pred_res[i]['local pose'] = w_prev * prev_dat['local pose'] + w_next * next_dat['local pose']
            pred_res[i]['global pose'] = w_prev * prev_dat['global pose'] + w_next * next_dat['global pose']



if __name__ == "__main__":
    if not osp.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    data_resource = load_dataset(DATASET_PATH)
    test_dataset = load_dataset(DATA_PATH)

    searchset = [[] for _ in range(NUM_POSE * 2)]
    for dat in data_resource:
        hand_id = int(dat[0][1])
        pose_id = int(dat[0][0])
        searchset[hand_id * NUM_POSE + pose_id].append(dat)
    
    intp_gap = int(VIDEO_FPS / DATA_FPS)
    pred_size = len(test_dataset) * intp_gap - intp_gap + 1
    pred_res = [None] * pred_size
    logger.info('Predicting...')
    for i, test_data in tqdm(enumerate(test_dataset)):
        hand_id = int(test_data[0][1])
        pose_id = int(test_data[0][0])
        seq = int(test_data[0][2])
        subset = searchset[hand_id * NUM_POSE + pose_id]
        touch_ind = [i for i, x in enumerate(HANDPOSE_DICT[pose_id].split()) if x == '1']

        pred_data = kNNsearch(test_data, subset, touch_ind, k=knncfg.K, metric=knncfg.METRIC)
        e_test_local, e_test_global = get_skeleton_from_data(test_data)
        e_pred_local, e_pred_global = get_skeleton_from_data(pred_data)
        e_pred_trans = transform_to_global(test_data, pred_data, e_test_global, e_pred_global)
        e_test_global = xzy_to_xyz(e_test_global)
        e_pred_trans = xzy_to_xyz(e_pred_trans)
        # vis(test_data, e_test_local, e_test_global, e_pred_local, e_pred_trans, None, None, show=True)

        pred_res[i * intp_gap] = {'data': pred_data, 'local pose': e_pred_local, 'global pose': e_pred_trans}
    
    pred_interpolate(pred_res, intp_gap)

    logger.info('Generating handpose frames...')
    for i, res in tqdm(enumerate(pred_res)):
        pred_data = res['data']
        hand_id = int(test_data[0][1])
        pose_id = int(test_data[0][0])
        figname = str(i).zfill(4) + '.jpg'
        
        e_pred_local = res['local pose']
        e_pred_global = res['global pose']

        # vis_pred(pred_data, e_pred_local, e_pred_global, show=False, save=True, save_dir=SAVE_DIR, save_fig=figname)
    
    with open(osp.join(SAVE_DIR, 'preddata.pkl'), 'wb') as f:
        pickle.dump(pred_res, f)


    