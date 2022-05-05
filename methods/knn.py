import sys
sys.path.append("..")
import numpy as np
import os
from tqdm import tqdm
from loguru import logger
import os.path as osp
import math
from config import cfg
from tool.handmodel import get_skeleton_from_data, transform_to_global
from tool.visualize import vis, vis_fp
from tool.dataload import load_dataset
from loss import eloss_local, eloss_global, eloss_tip, loss_stat


HANDPOSE_DICT = cfg.HANDPOSE_DICT
NUM_POSE = cfg.NUM_POSE
DATASET_PATH = '../dataset/train_new/txts'
DATA_PATH = '../dataset/test_type_new'
K = 20
METRIC = 'aL1'
LOSS_TYPE = 1 # 0: EPE 1:EPE_v
SAVE_FIGURE = True
WRITE_RESULT = False
# SAVE_DIR = osp.join('..', 'results', '_'.join(['gbl', 'typeknn_new', str(K), METRIC]))
SAVE_DIR = '../dataset/train_new/imgs'

def softmax(x):
    x = x - np.max(x)
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

def kNNsearch(x0, dataset, touch_ind, k = 20, metric = 'aL1'):
    w_a = 1.0
    w_p = 0.33
    w_y = 0.33

    x = x0.copy()
    a_ind = [4,9,14,19,24]
    p_ind = [5,10,15,20,25]
    yaw_bx = x[a_ind][touch_ind][0,1]
    x_bx = x[p_ind][touch_ind][0,0]
    y_bx = x[p_ind][touch_ind][0,2]
    for i in touch_ind:
        x[a_ind[i], 1] -= yaw_bx
        x[p_ind[i], 0] -= x_bx
        x[p_ind[i], 2] -= y_bx
    best_z = [None] * k
    min_dist = np.array([math.inf] * k)
    for z0 in dataset:
        z = z0.copy()
        yaw_bz = z[a_ind][touch_ind][0,1]
        x_bz = z[p_ind][touch_ind][0,0]
        y_bz = z[p_ind][touch_ind][0,2]
        for i in touch_ind:
            z[a_ind[i], 1] -= yaw_bz
            z[p_ind[i], 0] -= x_bz
            z[p_ind[i], 2] -= y_bz
        if metric == 'aL2':
            dist = np.linalg.norm(x[a_ind][touch_ind] - z[a_ind][touch_ind])
        elif metric == 'aL1':
            # a_dist = np.sum(np.abs(x[a_ind][touch_ind] - z[a_ind][touch_ind]))
            a_dist = 3.0 * (
                w_p * np.sum(np.abs(x[a_ind][touch_ind][:,0] - z[a_ind][touch_ind][:,0])) + 
                w_y * np.sum(np.abs(x[a_ind][touch_ind][:,1] - z[a_ind][touch_ind][:,1])) + 
                (1.0 - w_p - w_y) * np.sum(np.abs(x[a_ind][touch_ind][:,2] - z[a_ind][touch_ind][:,2]))
            )
            p_dist = sum([np.sqrt((x[p_ind[i],0] - z[p_ind[i],0]) ** 2 + (x[p_ind[i],2] - z[p_ind[i],2]) ** 2) for i in touch_ind])
            dist = w_a * a_dist  + (1 - w_a) * p_dist

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
    avg_best_dist = np.linalg.norm(x[a_ind][touch_ind] - avg_best_z[a_ind][touch_ind])
    if avg_best_dist < np.min(min_dist):
        res = avg_best_z
    else:
        res = best_z[np.argmin(min_dist)]
    return res


def deform(test, pred, touch_ind, method = 'default'):
    pip_phi = np.zeros(5)
    pip_theta = np.zeros(5)
    iip_theta = np.zeros(5)
    dip_theta = np.zeros(5)
    pred_tip_angles = np.zeros((5,3))
    gt_tip_angles = np.zeros((5,3))

    pose_id = int(test[0][0])
    pred_new = pred.copy()

    for i in touch_ind:
        pip_phi[i] = pred[i * 5 + 1][0]
        pip_theta[i] = pred[i * 5 + 1][1]
        iip_theta[i] = pred[i * 5 + 2][0]
        dip_theta[i] = pred[i * 5 + 3][0]
        pred_tip_angles[i] = pred[i * 5 + 4]
        gt_tip_angles[i] = test[i * 5 + 4]
    
    err = gt_tip_angles - pred_tip_angles

    alpha = 0.0
    # 1-finger touch
    if pose_id in range(5):
        fin = touch_ind[0]
        err = gt_tip_angles[fin] - pred_tip_angles[fin]
        # not thumb
        if pose_id > 0:
            # pip_phi[fin] += err[1] # yaw
            if method == 'default':
                pip_theta[fin] += err[0]
            elif method == 'average':
                pip_theta[fin] += err[0] * alpha
                iip_theta[fin] += err[0] * 0.5 * (1.0 - alpha)
                dip_theta[fin] += err[0] * 0.5 * (1.0 - alpha)
        # thumb
        else:
            if method == 'default':
                pip_theta[fin] += err[0]
                # pip_phi[fin] += err[0] + err[1]
            elif method == 'average':
                pip_theta[fin] += err[0] * alpha
                iip_theta[fin] += err[0] * 0.5 * (1.0 - alpha)
                dip_theta[fin] += err[0] * 0.5 * (1.0 - alpha)
                # pip_phi[fin] += err[0] + err[1]
    # 2-finger touch
    else:
        err_pitch = err[:,0]
        err_yaw = err[:,1]
        avg_pitch_err = np.average(err_pitch[touch_ind])
        avg_yaw_err = np.average(err_yaw[touch_ind])
        pitch_delta = err_pitch - avg_pitch_err
        yaw_delta = err_yaw - avg_yaw_err
        # print(pitch_delta, yaw_delta)
        for i in touch_ind:
            if i > 0:
                # pip_phi[i] += yaw_delta[i] # yaw
                if method == 'default':
                    pip_theta[i] += pitch_delta[i]
                elif method == 'average':
                    pip_theta[i] += pitch_delta[i] * alpha
                    iip_theta[i] += pitch_delta[i] * 0.5 * (1.0 - alpha)
                    dip_theta[i] += pitch_delta[i] * 0.5 * (1.0 - alpha)
            else:
                if method == 'default':
                    pip_theta[i] += pitch_delta[i]
                    # pip_phi[fin] += pitch_delta[i] + yaw_delta[i]
                elif method == 'average':
                    pip_theta[i] += pitch_delta[i] * alpha
                    iip_theta[i] += pitch_delta[i] * 0.5 * (1.0 - alpha)
                    dip_theta[i] += pitch_delta[i] * 0.5 * (1.0 - alpha)
                    # pip_phi[fin] += pitch_delta[i] + yaw_delta[i]  

    for i in touch_ind:
        pred_new[i * 5 + 1][0] = pip_phi[i]
        pred_new[i * 5 + 1][1] = pip_theta[i]
        pred_new[i * 5 + 2][0] = iip_theta[i]
        pred_new[i * 5 + 3][0] = dip_theta[i]
        pred_new[i * 5 + 4] = gt_tip_angles[i]
    
    return pred_new


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
        # kNN
        pred = kNNsearch(test_data, subset, touch_ind, k=K, metric=METRIC)
        # deformation
        pred_def = deform(test_data, pred, touch_ind, method='average')
        e_test_local, e_test_global = get_skeleton_from_data(test_data)
        e_pred_local, e_pred_global = get_skeleton_from_data(pred)
        e_pred2_local, e_pred2_global = get_skeleton_from_data(pred_def)
        e_pred_trans = transform_to_global(test_data, pred, e_test_global, e_pred_global)
        e_pred2_trans = transform_to_global(test_data, pred, e_test_global, e_pred2_global)
        # lossval = eloss_local(e_test_local, e_pred2_local, touch_ind, all=(LOSS_TYPE == 0))
        # lossval = eloss_global(e_test_global, e_pred2_trans, touch_ind, all=(LOSS_TYPE == 0))
        lossval = eloss_tip(e_test_global, e_pred2_trans, touch_ind, all=(LOSS_TYPE == 0))
        losses[hand_id * NUM_POSE + pose_id].append(lossval)
        all_losses.append(lossval)
        angle_lossval = angle_loss(test_data, pred_def, touch_ind)
        angle_losses[hand_id * NUM_POSE + pose_id].append(angle_lossval)
        all_angle_losses.append(angle_lossval)

        if SAVE_FIGURE:
            save_figname = '_'.join([str(pose_id).zfill(2), str(hand_id), str(seq).zfill(3)]) + '.jpg'
            vis(test_data, e_test_local, e_test_global, e_test_local, e_test_global, 
                None, None, show=False, save=True, save_dir=SAVE_DIR, save_fig=save_figname)
            # save_fpfigname = save_figname[:-4] + 'fp.jpg'
            # vis_fp(test_data, pred, show=False, save=False, save_dir=SAVE_DIR, save_fig=save_fpfigname)
    
    loss_stat(NUM_POSE, all_losses, losses, all_angle_losses, angle_losses, WRITE_RESULT, SAVE_DIR)