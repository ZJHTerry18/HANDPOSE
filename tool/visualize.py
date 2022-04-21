import sys
import matplotlib
sys.path.append('..')
import numpy as np
import matplotlib.image as mpimg
import cv2
import matplotlib.pyplot as plt
import os
import os.path as osp
from config import cfg

HANDPOSE_DICT = cfg.HANDPOSE_DICT
NUM_POSE = cfg.NUM_POSE
IMG_PATH = cfg.FP_IMAGE_PATH


def plot_kp(ax,xp,yp,zp, touch_ind = None, finger_color = 'violet', palm_color = 'blue', linestyle = '-'):
	finger_dict = {0:7, 1:10, 2:13, 3:16, 4:19}
	untouch_id = list(range(20))
	for i in touch_ind:
		id = finger_dict[i]
		ax.scatter3D(xp[id], yp[id], zp[id], c='r', marker='v')
		untouch_id.remove(id)
	ax.scatter3D(xp[untouch_id], yp[untouch_id], zp[untouch_id], cmap="Greens")
	ax.plot(np.hstack((xp[0],xp[1])),
			np.hstack((yp[0],yp[1])),
			np.hstack((zp[0],zp[1])),
			ls=linestyle, color=finger_color)
	ax.plot(np.hstack((xp[0],xp[2])),
			np.hstack((yp[0],yp[2])),
			np.hstack((zp[0],zp[2])),
			ls=linestyle, color=palm_color)
	ax.plot(np.hstack((xp[0],xp[3])),
			np.hstack((yp[0],yp[3])),
			np.hstack((zp[0],zp[3])),
			ls=linestyle, color=palm_color)
	ax.plot(np.hstack((xp[0],xp[4])),
			np.hstack((yp[0],yp[4])),
			np.hstack((zp[0],zp[4])),
			ls=linestyle, color=palm_color)
	ax.plot(np.hstack((xp[0],xp[5])),
			np.hstack((yp[0],yp[5])),
			np.hstack((zp[0],zp[5])),
			ls=linestyle, color=palm_color)
	ax.plot(np.hstack((xp[1],xp[6:8])),
			np.hstack((yp[1],yp[6:8])),
			np.hstack((zp[1],zp[6:8])),
			ls=linestyle, color=finger_color)
	ax.plot(np.hstack((xp[2],xp[8:11])),
			np.hstack((yp[2],yp[8:11])),
			np.hstack((zp[2],zp[8:11])),
			ls=linestyle, color=finger_color)
	ax.plot(np.hstack((xp[3],xp[11:14])),
			np.hstack((yp[3],yp[11:14])),
			np.hstack((zp[3],zp[11:14])),
			ls=linestyle, color=finger_color)
	ax.plot(np.hstack((xp[4],xp[14:17])),
			np.hstack((yp[4],yp[14:17])),
			np.hstack((zp[4],zp[14:17])),
			ls=linestyle, color=finger_color)
	ax.plot(np.hstack((xp[5],xp[17:20])),
			np.hstack((yp[5],yp[17:20])),
			np.hstack((zp[5],zp[17:20])),
			ls=linestyle, color=finger_color)


def vis(data, e_local_test, e_global_test, e_local_res, e_global_res, e_local_res2 = None, e_global_res2 = None, show = False, save = False, save_dir = None, save_fig = None):
    hand_id = int(data[0][1])
    pose_id = int(data[0][0])
    seq = int(data[0][2])

    xp_local_test = e_local_test.T[0].T
    yp_local_test = e_local_test.T[1].T
    zp_local_test = e_local_test.T[2].T
    xp_global_test = e_global_test.T[0].T
    yp_global_test = e_global_test.T[1].T
    zp_global_test = e_global_test.T[2].T
    xp_local_res = e_local_res.T[0].T
    yp_local_res = e_local_res.T[1].T
    zp_local_res = e_local_res.T[2].T
    xp_global_res = e_global_res.T[0].T
    yp_global_res = e_global_res.T[1].T
    zp_global_res = e_global_res.T[2].T

    if (e_local_res2 is not None) and (e_global_res2 is not None):
        xp_local_res2 = e_local_res2.T[0].T
        yp_local_res2 = e_local_res2.T[1].T
        zp_local_res2 = e_local_res2.T[2].T
        xp_global_res2 = e_global_res2.T[0].T
        yp_global_res2 = e_global_res2.T[1].T
        zp_global_res2 = e_global_res2.T[2].T

    touch_ind = [i for i, x in enumerate(HANDPOSE_DICT[pose_id].split()) if x == '1']
	
    fig = plt.figure(figsize=plt.figaspect(0.5))
    axtest1 = fig.add_subplot(1,2,1, projection='3d')
    axtest2 = fig.add_subplot(1,2,2, projection='3d')
    axtest1.set_title('test local')
    axtest2.set_title('test global')
    axtest1.set_xlim3d([-10,10])
    axtest1.set_ylim3d([0,20])
    axtest1.set_zlim3d([-10,10])
    axtest2.set_xlim3d([-10,10])
    axtest2.set_ylim3d([10,30])
    axtest2.set_zlim3d([-10,10])
    axtest2.set_xlim3d([-10,10])
    axtest1.set_xticklabels([])
    axtest1.set_yticklabels([])
    axtest1.set_zticklabels([])
    axtest2.set_xticklabels([])
    axtest2.set_yticklabels([])
    axtest2.set_zticklabels([])
    axtest1.view_init(elev=45, azim=45)
    axtest2.view_init(elev=-45, azim=-90)
	
	# plot connections
    plot_kp(axtest1,xp_local_test, yp_local_test, zp_local_test, touch_ind)
    plot_kp(axtest2,xp_global_test, yp_global_test, zp_global_test, touch_ind)
    plot_kp(axtest1,xp_local_res, yp_local_res, zp_local_res, touch_ind, linestyle='--')
    plot_kp(axtest2,xp_global_res, yp_global_res, zp_global_res, touch_ind, linestyle='--')
    if (e_local_res2 is not None) and (e_global_res2 is not None):
        plot_kp(axtest1,xp_local_res2, yp_local_res2, zp_local_res2, touch_ind, linestyle=':', finger_color='green')
        plot_kp(axtest2,xp_global_res2, yp_global_res2, zp_global_res2, touch_ind, linestyle=':', finger_color='green')
    plt.axis('on')
    if show: 
        plt.show()
    if save:
        plt.savefig(osp.join(save_dir, save_fig))
    plt.close()


def vis_fp(data, pred, show = False, save = False, save_dir = None, save_fig = None):
    finger_dict = [4,9,14,19,24]
    hand_id = int(data[0][1])
    pose_id = int(data[0][0])
    seq = int(data[0][2])
    touch_ind = [i for i, x in enumerate(HANDPOSE_DICT[pose_id].split()) if x == '1']
	
    plt.figure()
    # plot fingerprint image
    handtype = 'left' if hand_id == 0 else 'right'
    imgname = str(pose_id).zfill(2) + '_' + str(seq).zfill(3) + '.png'
    img = cv2.imread(osp.join(IMG_PATH, handtype, str(pose_id), imgname))
    w = img.shape[0]
    h = img.shape[1]
    plt.imshow(img)

    celltext = [['-' for _ in range(5)] for i in range(6)]
    for i in touch_ind:
        lid = finger_dict[i]
        gt = [str(round(data[lid][k],3)) for k in range(3)]
        pr = [str(round(pred[lid][k],3)) for k in range(3)]
        for j in range(3):
            celltext[j][i] = gt[j]
            celltext[j+3][i] = pr[j]
    rows = ['fp_pitch', 'fp_yaw', 'fp_roll', 'leap_pitch', 'leap_yaw', 'leap_roll']
    collabels = ['0','1','2','3','4']
    if hand_id == 0:
        collabels.reverse()
        for i in range(6):
            celltext[i].reverse()
    plt.table(cellText=celltext, rowLabels=rows, colLabels=collabels, loc='bottom')
    plt.subplots_adjust(left=0.2, bottom=0.4)
    ax = plt.gca()
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_position(('data', 0))

    plt.axis('on')
    if show: 
        plt.show()
    if save:
        plt.savefig(osp.join(save_dir, save_fig))
    plt.close()