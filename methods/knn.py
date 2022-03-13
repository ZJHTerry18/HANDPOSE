import sys
sys.path.append("..")
import numpy as np
import os
from tqdm import tqdm
from loguru import logger
import os.path as osp
import math
from tool.handmodel import vis_and_save_result, save_video, get_skeleton_from_data
from tool.dataload import load_dataset

HANDPOSE_DICT = ["1 0 0 0 0", "0 1 0 0 0", "0 0 1 0 0", "0 0 0 1 0", "0 0 0 0 1",
	"1 1 0 0 0", "0 1 1 0 0", 
	"1 0 1 0 0", "1 0 0 1 0", "1 0 0 0 1", "0 1 0 1 0", "0 1 0 0 1", "0 0 1 1 0", "0 0 0 1 1",
	"1 1 1 0 0", "0 1 1 1 0", "0 0 1 1 1", 
	"1 1 0 1 0", "1 0 1 1 0", "1 0 0 1 1", "1 1 0 0 1",
	"0 1 1 1 1", "1 0 1 1 1",
	"1 1 0 1 1", "1 1 1 1 0", 
	"1 1 1 1 1"]
DATASET_PATH = '../dataset/train'
DATA_PATH = '../dataset/test'
NUM_POSE = 26
K = 10
METRIC = 'aL2'
WRITE_RESULT = True
SAVE_DIR = osp.join('..', 'results', '_'.join(['knn', str(K), METRIC]))

# def load_data(path, datafile):
#     with open(osp.join(path, datafile), 'r') as f:
#         lines = f.readlines()
#     dat = []
#     title = datafile[:-4].split('_')
#     title = [float(x) for x in title]
#     dat.append(title)
#     for line in lines[2:]:
#         line = line.rstrip('\n').split()
#         line = [float(x) for x in line]
#         while len(line) < 3:
#             line.append(0.0)
#         dat.append(line)
#     return np.array(dat)


# def load_dataset(path):
#     datafile_list = os.listdir(path)
#     dataset = []
#     logger.info('Loading dataset...')
#     for datafile in tqdm(datafile_list):
#         dat = load_data(path, datafile)
#         dataset.append(dat)
#     return dataset

def softmax(x):
    x = x - np.max(x)
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

def kNNsearch(x, dataset, touch_ind, k = 10, metric = 'aL2'):
    ind = [4,9,14,19,24]
    best_z = [None] * k
    min_dist = np.array([math.inf] * k)
    for z in dataset:
        if metric == 'aL2':
            dist = np.linalg.norm(x[ind][touch_ind] - z[ind][touch_ind])

        # to do: other metrics
        
        if dist < np.max(min_dist):
            rep_ind = np.argmax(min_dist)
            best_z[rep_ind] = z
            min_dist[rep_ind] = dist
    # w_arr = softmax(-min_dist)
    # print(min_dist, w_arr)
    # avg_best_z = np.sum(np.array([best_z[i] * w_arr[i] for i in range(len(w_arr))]), axis=0)
    # print(res)
    avg_best_z = np.average(np.array(best_z), axis=0)
    avg_best_dist = np.linalg.norm(x[ind][touch_ind] - avg_best_z[ind][touch_ind])
    if avg_best_dist < np.min(min_dist):
        res = avg_best_z
    else:
        res = best_z[np.argmin(min_dist)]
    return res


def angle_loss(test, pred, touch_ind):
    ind = [4,9,14,19,24]
    pitch_err = []
    yaw_err = []
    roll_err = []
    for i in touch_ind:
        j = ind[i]
        pitch_err.append(abs(test[j][0] - pred[j][0]))
        yaw_err.append(abs(test[j][1] - pred[j][1]))
        roll_err.append(abs(test[j][2] - pred[j][2]))
    pitchloss = np.average(np.array(pitch_err))
    yawloss = np.average(np.array(yaw_err))
    rollloss = np.average(np.array(roll_err))
    return list([pitchloss, yawloss, rollloss])


def loss(e_test, e_pred, touch_ind, all = True):
    e_dist = np.sqrt(np.sum(np.square(e_test - e_pred), axis=1))
    ind = []
    if all:
        ind = [1] + list(range(6,20))
    else:
        inddict = {0:[1,6,7], 1:[8,9,10], 2:[11,12,13], 3:[14,15,16], 4:[17,18,19]}
        for i in touch_ind:
            ind = ind + inddict[i]
    return np.average(e_dist[ind])


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
    
    logger.info('Predicting...')
    losses = dict([(k, []) for k in range(2 * NUM_POSE)])
    angle_losses = dict([(k, []) for k in range(2 * NUM_POSE)])
    all_losses = []
    all_angle_losses = []
    for test_data in tqdm(test_dataset):
        hand_id = int(test_data[0][1])
        pose_id = int(test_data[0][0])
        seq = int(test_data[0][2])
        subset = searchset[hand_id * NUM_POSE + pose_id]
        touch_ind = [i for i, x in enumerate(HANDPOSE_DICT[pose_id].split()) if x == '1']
        # print(touch_ind)
        pred = kNNsearch(test_data, subset, touch_ind, k=K, metric=METRIC)
        e_test_local, e_test_global = get_skeleton_from_data(test_data)
        e_pred_local, e_pred_global = get_skeleton_from_data(pred)
        lossval = loss(e_test_local, e_pred_local, touch_ind, all=True)
        losses[hand_id * NUM_POSE + pose_id].append(lossval)
        all_losses.append(lossval)
        angle_lossval = angle_loss(test_data, pred, touch_ind)
        angle_losses[hand_id * NUM_POSE + pose_id].append(angle_lossval)
        all_angle_losses.append(angle_lossval)

        save_figname = '_'.join([str(pose_id).zfill(2), str(hand_id), str(seq).zfill(3)]) + '.jpg'
        vis_and_save_result(hand_id, pose_id, e_test_local, e_test_global, e_pred_local, e_pred_global, 
            show=False, save=False, save_dir=SAVE_DIR, save_fig=save_figname)
    
    num = np.zeros(2 * NUM_POSE)
    avg_losses = np.zeros(2 * NUM_POSE)
    std_losses = np.zeros(2 * NUM_POSE)
    max_losses = np.zeros(2 * NUM_POSE)
    min_losses = np.zeros(2 * NUM_POSE)
    avg_pitch_losses = np.zeros(2 * NUM_POSE)
    std_pitch_losses = np.zeros(2 * NUM_POSE)
    avg_yaw_losses = np.zeros(2 * NUM_POSE)
    std_yaw_losses = np.zeros(2 * NUM_POSE)
    avg_roll_losses = np.zeros(2 * NUM_POSE)
    std_roll_losses = np.zeros(2 * NUM_POSE)
    for i in range(2 * NUM_POSE):
        num[i] = len(losses[i])
        avg_losses[i] = np.average(np.array(losses[i])) * 10.0
        std_losses[i] = np.std(np.array(losses[i])) * 10.0
        max_losses[i] = np.max(np.array(losses[i])) * 10.0
        min_losses[i] = np.min(np.array(losses[i])) * 10.0

        avg_anglelosses = np.average(np.array(angle_losses[i]), axis=0)
        std_anglelosses = np.std(np.array(angle_losses[i]), axis=0)
        print(avg_anglelosses, std_anglelosses)
        avg_pitch_losses[i] = avg_anglelosses[0]
        avg_yaw_losses[i] = avg_anglelosses[1]
        avg_roll_losses[i] = avg_anglelosses[2]
        std_pitch_losses[i] = std_anglelosses[0]
        std_yaw_losses[i] = std_anglelosses[1]
        std_roll_losses[i] = std_anglelosses[2]
        logger.info('Prediction %d\n data amount: %d\naverage loss:%.4f  std loss:%.4f  max loss:%.4f  min loss:%.4f\n' \
            'avg pitch err:%.4f  avg yaw err:%.4f  avg roll err:%.4f\nstd pitch err:%.4f  std yaw err:%.4f  std roll err:%.4f' % 
            (i, num[i], avg_losses[i], std_losses[i], max_losses[i], min_losses[i], 
            avg_pitch_losses[i], avg_yaw_losses[i], avg_roll_losses[i], std_pitch_losses[i], std_yaw_losses[i], std_roll_losses[i]))
    if WRITE_RESULT:
        with open(osp.join(SAVE_DIR, 'result.txt'), 'w') as f:
            for i in range(2 * NUM_POSE):
                f.write('Prediction %d\n data amount: %d\naverage loss:%.4f  std loss:%.4f  max loss:%.4f  min loss:%.4f\n' \
            'avg pitch err:%.4f  avg yaw err:%.4f  avg roll err:%.4f\nstd pitch err:%.4f  std yaw err:%.4f  std roll err:%.4f' % 
            (i, num[i], avg_losses[i], std_losses[i], max_losses[i], min_losses[i], 
            avg_pitch_losses[i], avg_yaw_losses[i], avg_roll_losses[i], std_pitch_losses[i], std_yaw_losses[i], std_roll_losses[i]))
    
    avg_loss = np.average(np.array(all_losses)) * 10.0
    std_loss = np.std(np.array(all_losses)) * 10.0
    avg_angle_loss = np.average(np.array(all_angle_losses), axis=0)
    std_angle_loss = np.std(np.array(all_angle_losses), axis=0)
    logger.info('Total:\naverage loss:%.4f  std loss:%.4f\navg pitch loss:%.4f  avg yaw loss:%.4f  avg roll loss:%.4f\n' \
        'std pitch loss:%.4f  std yaw loss:%.4f  std roll loss:%.4f' 
        % (avg_loss, std_loss, avg_angle_loss[0], avg_angle_loss[1], avg_angle_loss[2], std_angle_loss[0], std_angle_loss[1], std_angle_loss[2]))

    
    # save_video(SAVE_DIR, osp.join(SAVE_DIR, 'output.mp4'))
    
