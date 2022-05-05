import numpy as np
from loguru import logger
import csv
import os

def eloss_local(e_test, e_pred, touch_ind, all = True):
    e_dist = np.sqrt(np.sum(np.square(e_test - e_pred), axis=1))
    ind = []
    if all:
        ind = [1] + list(range(6,20))
    else:
        inddict = {0:[1,6,7], 1:[8,9,10], 2:[11,12,13], 3:[14,15,16], 4:[17,18,19]}
        for i in touch_ind:
            ind = ind + inddict[i]
    return np.average(e_dist[ind])

def eloss_global(e_test, e_pred, touch_ind, all=True):
    e_dist = np.sqrt(np.sum(np.square(e_test - e_pred), axis=1))
    ind = []
    if all:
        ind = list(range(20))
    else:
        inddict = {0:[0,1,6,7], 1:[2,8,9,10], 2:[3,11,12,13], 3:[4,14,15,16], 4:[5,17,18,19]}
        for i in touch_ind:
            ind = ind + inddict[i]
    return np.average(e_dist[ind])

def eloss_tip(e_test, e_pred, touch_ind, all=True):
    e_dist = np.sqrt(np.sum(np.square(e_test - e_pred), axis=1))
    ind = []
    if all:
        ind = list(range(20))
    else:
        inddict = {0:[7], 1:[10], 2:[13], 3:[16], 4:[19]}
        for i in touch_ind:
            ind = ind + inddict[i]
    return np.average(e_dist[ind])


def loss_stat(num_pose, all_losses, losses, all_angle_losses = None, angle_losses = None, write_result = None, save_dir = None):
    num = np.zeros(2 * num_pose)
    avg_losses = np.zeros(2 * num_pose)
    std_losses = np.zeros(2 * num_pose)
    max_losses = np.zeros(2 * num_pose)
    min_losses = np.zeros(2 * num_pose)
    avg_pitch_losses = np.zeros(2 * num_pose)
    std_pitch_losses = np.zeros(2 * num_pose)
    avg_yaw_losses = np.zeros(2 * num_pose)
    std_yaw_losses = np.zeros(2 * num_pose)
    avg_roll_losses = np.zeros(2 * num_pose)
    std_roll_losses = np.zeros(2 * num_pose)
    for i in range(2 * num_pose):
        num[i] = len(losses[i])
        avg_losses[i] = np.average(np.array(losses[i])) * 10.0
        std_losses[i] = np.std(np.array(losses[i])) * 10.0
        max_losses[i] = np.max(np.array(losses[i])) * 10.0
        min_losses[i] = np.min(np.array(losses[i])) * 10.0

        if angle_losses != None:
            avg_anglelosses = np.average(np.array(angle_losses[i]), axis=0)
            std_anglelosses = np.std(np.array(angle_losses[i]), axis=0)
            avg_pitch_losses[i] = avg_anglelosses[0]
            avg_yaw_losses[i] = avg_anglelosses[1]
            avg_roll_losses[i] = avg_anglelosses[2]
            std_pitch_losses[i] = std_anglelosses[0]
            std_yaw_losses[i] = std_anglelosses[1]
            std_roll_losses[i] = std_anglelosses[2]
        # logger.info('Prediction %d\n data amount: %d\naverage loss:%.4f  std loss:%.4f  max loss:%.4f  min loss:%.4f\n' \
        #     'avg pitch err:%.4f  avg yaw err:%.4f  avg roll err:%.4f\nstd pitch err:%.4f  std yaw err:%.4f  std roll err:%.4f' % 
        #     (i, num[i], avg_losses[i], std_losses[i], max_losses[i], min_losses[i], 
        #     avg_pitch_losses[i], avg_yaw_losses[i], avg_roll_losses[i], std_pitch_losses[i], std_yaw_losses[i], std_roll_losses[i]))
        logger.info('Prediction %d\n data amount: %d\naverage loss:%.4f  std loss:%.4f  max loss:%.4f  min loss:%.4f' % 
            (i, num[i], avg_losses[i], std_losses[i], max_losses[i], min_losses[i]))
    
    avg_loss = np.average(np.array(all_losses)) * 10.0
    std_loss = np.std(np.array(all_losses)) * 10.0
    if all_angle_losses is not None:
        avg_angle_loss = np.average(np.array(all_angle_losses), axis=0)
        std_angle_loss = np.std(np.array(all_angle_losses), axis=0)
    else:
        avg_angle_loss = np.zeros(3)
        std_angle_loss = np.zeros(3)
    # logger.info('Total:\naverage loss:%.4f  std loss:%.4f\navg pitch loss:%.4f  avg yaw loss:%.4f  avg roll loss:%.4f\n' \
    #     'std pitch loss:%.4f  std yaw loss:%.4f  std roll loss:%.4f' 
    #     % (avg_loss, std_loss, avg_angle_loss[0], avg_angle_loss[1], avg_angle_loss[2], std_angle_loss[0], std_angle_loss[1], std_angle_loss[2]))
    logger.info('Total:\naverage loss:%.4f  std loss:%.4f'
        % (avg_loss, std_loss))
    
    if write_result:
        with open(os.path.join(save_dir, 'result.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["pose", "data amount", "average loss", "std loss", "max loss", "min loss"])
            for i in range(2 * num_pose):
                writer.writerow([i, num[i], avg_losses[i], std_losses[i], max_losses[i], min_losses[i]])
            writer.writerow(["total data amount", "total average loss", "total std loss"])
            writer.writerow([np.sum(num), avg_loss, std_loss])
    
