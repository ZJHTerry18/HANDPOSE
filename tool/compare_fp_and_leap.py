import sys
sys.path.append('..')
import os
import os.path as osp
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from config import cfg
from dataload import load_data
from visualize import vis_fp

HANDPOSE_DICT = cfg.HANDPOSE_DICT
FP_PATH = '../dataset/test_type_2'
FP_IMAGE_PATH = cfg.FP_IMAGE_PATH
LEAP_PATH = '../dataset/test_type_wofp2'
SAVE_DIR = '../dataset/test_type_2/fp_and_leap'

finger_dict = [5,10,15,20,25]

fpfiles = os.listdir(FP_PATH)
leapfiles = os.listdir(LEAP_PATH)

def cal_err():
    pitch_errs = []
    pitch_errs_abs = []
    yaw_errs = []
    yaw_errs_abs = []
    roll_errs = []
    roll_errs_abs = []
    cnt = 0
    for file in fpfiles:
        if file in leapfiles:
            cnt += 1
            poseid = int(file.split('_')[0])
            # handid = int(file.split('_')[1])
            # seq = int(file.split('_')[2][:-4])

            touch_ind = [i for i, x in enumerate(HANDPOSE_DICT[poseid].split()) if x == '1']

            with open(osp.join(FP_PATH, file), 'r') as ffp:
                fplines = ffp.readlines()
            with open(osp.join(LEAP_PATH, file), 'r') as fleap:
                fllines = fleap.readlines()
            
            for i in touch_ind:
                fp_angles = fplines[finger_dict[i]].rstrip('\n').split()
                fp_angles = [float(x) for x in fp_angles[:3]]
                leap_angles = fllines[finger_dict[i]].rstrip('\n').split()
                leap_angles = [float(x) for x in leap_angles[:3]]

                pitch_errs.append(fp_angles[0] - leap_angles[0])
                yaw_errs.append(fp_angles[1] - leap_angles[1])
                roll_errs.append(fp_angles[2] - leap_angles[2])
                pitch_errs_abs.append(np.abs(fp_angles[0] - leap_angles[0]))
                yaw_errs_abs.append(np.abs(fp_angles[1] - leap_angles[1]))
                roll_errs_abs.append(np.abs(fp_angles[2] - leap_angles[2]))

    print(cnt)
    print("pitch avg err:", np.average(np.array(pitch_errs)))
    print("pitch err std:", np.std(np.array(pitch_errs)))
    print("yaw avg err:", np.average(np.array(yaw_errs)))
    print("yaw err std:", np.std(np.array(yaw_errs)))
    print("roll avg err:", np.average(np.array(roll_errs)))
    print("roll err std:", np.std(np.array(roll_errs)))

def plot_fp_angle_fig():
    for file in tqdm(fpfiles):
        if file in leapfiles:
            fp_data = load_data(FP_PATH, file)
            leap_data = load_data(LEAP_PATH, file)
            hand_id = int(fp_data[0][1])
            pose_id = int(fp_data[0][0])
            seq = int(fp_data[0][2])
            save_img = '_'.join([str(pose_id).zfill(2), str(hand_id), str(seq).zfill(3)]) + 'fp.jpg'
            vis_fp(fp_data, leap_data, show=False, save=True, save_dir=SAVE_DIR, save_fig=save_img)

def plot_pos_fig():
    pos_dict = [5,10,15,20,25]
    for file in tqdm(fpfiles):
        if file in leapfiles:
            fp_data = load_data(FP_PATH, file)
            leap_data = load_data(LEAP_PATH, file)
            hand_id = int(fp_data[0][1])
            pose_id = int(fp_data[0][0])
            seq = int(fp_data[0][2])
            touch_ind = [i for i, x in enumerate(HANDPOSE_DICT[pose_id].split()) if x == '1']

            fig = plt.figure()
            ax1 = fig.add_subplot(1,2,1)
            ax1.set_title('fingerprint')
            ax2 = fig.add_subplot(1,2,2)
            ax2.set_title('leap')
            for i in touch_ind:
                x1 = fp_data[pos_dict[i]][0]
                y1 = fp_data[pos_dict[i]][2]
                x2 = leap_data[pos_dict[i]][0]
                y2 = leap_data[pos_dict[i]][2]
                ax1.scatter(x1, y1, c='g', marker='o')
                ax1.text(x1, y1, str(i))
                ax2.scatter(x2, y2, c='g', marker='o')
                ax2.text(x2, y2, str(i))
            save_fig = '_'.join([str(pose_id).zfill(2), str(hand_id), str(seq).zfill(3)]) + 'pos.jpg'
            plt.savefig(osp.join(SAVE_DIR, save_fig))
            plt.close()


if __name__ == "__main__":
    if not osp.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    cal_err()
    plot_fp_angle_fig()
    # plot_pos_fig()


