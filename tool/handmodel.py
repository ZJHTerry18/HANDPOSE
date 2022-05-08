import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import argparse
import os
import os.path as osp
from tqdm import tqdm
import cv2
from config import cfg

HANDPOSE_DICT = cfg.HANDPOSE_DICT
NUM_POSE = cfg.NUM_POSE


def in_angle(a, b, p):
	cosangle = np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))
	angle = np.arccos(cosangle)
	v = np.cross(a,b)
	# print(v)
	# if angle > np.pi * 0.5:
	# 	angle = angle - np.pi
	if np.dot(v,p) < 0:
		angle = -angle
	# print(angle)
	return angle

def rot_axis(v, k, theta):
	k = k / np.linalg.norm(k)
	v_rot = v * np.cos(theta) + k * (np.dot(k,v)) * (1 - np.cos(theta)) + np.cross(k,v) * np.sin(theta)
	return v_rot

def inverse_kinematics(e, direction, normal = None, handtype = 'left'):
    '''
    input: 
        e: 20*3 np.array, [x,y,z] of each joint
        direction: vector, the direction hand pointing at
        normal: vector, normal vector of palm
        handtype: 'left' or 'right'

    return:
        phi: 5-d np.array
        theta: 5*3 np.array
    '''
    import numpy.linalg as nl
    def angle(v1, v2):
        a = nl.norm(np.cross(v1,v2)) / (nl.norm(v1) * nl.norm(v2))
        return np.arcsin(a) * 180 / np.pi

    phi = np.zeros(5)
    theta = np.zeros((5,3))
    finger_joint_dict = {0:[0,1,6,7], 1:[2,8,9,10], 2:[3,11,12,13], 3:[4,14,15,16], 4:[5,17,18,19]}
    if normal == None:
        normal = np.cross((e[3] - e[0], e[2] - e[0])) if handtype == 'left' else -np.cross((e[3] - e[0], e[2] - e[0]))
    normal = normal / nl.norm(normal)
    direction = direction / nl.norm(direction)
    horiz = np.cross(direction, normal)

    for id in range(5):
        jlist = finger_joint_dict[id]

        e0 = e[jlist[0]] - e[0] if id > 0 else direction
        e1 = e[jlist[1]] - e[jlist[0]]
        e2 = e[jlist[2]] - e[jlist[1]]
        e3 = e[jlist[3]] - e[jlist[2]]

        v1 = np.cross(np.cross(normal, e1), normal)
        v1 = v1 / nl.norm(v1)

        phi[id] = angle(e0, v1)    
        if np.dot(horiz, e0) < np.dot(horiz, v1): 
            phi[id] = -phi[id]
        if handtype == 'right':
            phi[id] = -phi[id]
        theta[id][0] = angle(v1, e1)
        if np.dot(e1, normal) < 0: 
            theta[id][0] = -theta[id][0]
        theta[id][1] = angle(e1, e2)
        if np.dot(e1, normal) > np.dot(e2, normal):
            theta[id][1] = -theta[id][1]
        theta[id][2] = angle(e2, e3)
        if np.dot(e2, normal) > np.dot(e3, normal):
            theta[id][2] = -theta[id][2]
    

    return phi, theta

def get_skeleton_from_data(data):
	# default setting
	phi_I = 110 * np.pi / 180
	phi_M = 90 * np.pi / 180
	phi_R = 70 * np.pi / 180
	phi_L = 50 * np.pi / 180

	l = np.zeros((20,20)) # connection relationship
	l[0][1] = 7.0
	l[0][2] = 9.0
	l[0][3] = 8.5
	l[0][4] = 8.0
	l[0][5] = 8.0
	l[1][6] = 3.5
	l[6][7] = 3.0
	l[2][8] = 4.5
	l[8][9] = 2.5
	l[9][10] = 2.0
	l[3][11] = 5.0
	l[11][12] = 3.0
	l[12][13] = 2.0
	l[4][14] = 4.5
	l[14][15] = 3.0
	l[15][16] = 2.0
	l[5][17] = 4.0
	l[17][18] = 2.5
	l[18][19] = 2.0

	# # leap setting
	# phi_I = 116.93 * np.pi / 180
	# phi_M = 103.83 * np.pi / 180
	# phi_R = 73.57 * np.pi / 180
	# phi_L = 56.29 * np.pi / 180

	# l = np.zeros((20,20)) # connection relationship
	# l[0][1] = 4.5
	# l[0][2] = 7.0
	# l[0][3] = 6.4
	# l[0][4] = 5.9
	# l[0][5] = 5.9
	# l[1][6] = 3.1
	# l[6][7] = 2.4
	# l[2][8] = 3.9
	# l[8][9] = 2.2
	# l[9][10] = 1.7
	# l[3][11] = 4.4
	# l[11][12] = 2.6
	# l[12][13] = 1.9
	# l[4][14] = 4.1
	# l[14][15] = 2.5
	# l[15][16] = 1.9
	# l[5][17] = 3.2
	# l[17][18] = 1.8
	# l[18][19] = 1.7

	ind_dict_1 = {1:0, 6:2, 11:3, 16:4, 21:5}
	ind_dict_2 = {2:1, 7:8, 12:11, 17:14, 22:17}
	ind_dict_3 = {3:6, 8:9, 13:12, 18:15, 23:18}
	finger_dict = {0:7, 1:10, 2:13, 3:16, 4:19}

	e = np.zeros((20,3))
	phi = np.zeros(6)
	theta = np.zeros(20)
	tippos = np.zeros((5,3))    # x,y,z
	tipangles = np.zeros((5,3)) # pitch, yaw, roll

	hand_id = int(data[0][1]) # 0:left 1:right
	if hand_id == 0:
		phi_I = np.pi - phi_I
		phi_M = np.pi - phi_M
		phi_R = np.pi - phi_R
		phi_L = np.pi - phi_L

	touch_id = np.array([int(k) for k in HANDPOSE_DICT[int(data[0][0])].split()])

	for i in range(data.shape[0]):
		if i % 5 == 1:
			phi[ind_dict_1[i]] = data[i][0]
			theta[ind_dict_1[i]] = data[i][1]
		elif i % 5 == 2:
			theta[ind_dict_2[i]] = data[i][0]
		elif i % 5 == 3:
			theta[ind_dict_3[i]] = data[i][0]
		elif i % 5 == 4:
			for j in range(3):
				tipangles[i // 5][j] = data[i][j]
		elif i % 5 == 0 and i != 0:
			for j in range(3):
				tippos[i//5 - 1][j] = data[i][j]

	# phi[0] = phi[0] + 30.0
	# phi[1] = phi[1] - 14.06
	# phi[2] = phi[2] - 5.13
	# phi[3] = phi[3] + 5.38
	# phi[4] = phi[4] + 17.44
	
	phi = phi * np.pi / 180
	theta = theta * np.pi / 180
	tipangles = tipangles * np.pi / 180

	if hand_id == 0:
		phi = -phi

	# four digit PIPs
	e[2] = e[0] + l[0][2] * np.array([np.cos(phi_I), np.sin(phi_I), 0])
	e[3] = e[0] + l[0][3] * np.array([np.cos(phi_M), np.sin(phi_M), 0])
	e[4] = e[0] + l[0][4] * np.array([np.cos(phi_R), np.sin(phi_R), 0])
	e[5] = e[0] + l[0][5] * np.array([np.cos(phi_L), np.sin(phi_L), 0])

	# thumb
	if hand_id != 0:
		theta[1] = -theta[1]
		theta[6] = -theta[6]
	e[1] = e[0] + l[0][1] * np.array([np.cos(theta[0]) * np.cos(np.pi / 2 + phi[0]), np.cos(theta[0]) * np.sin(np.pi / 2 + phi[0]), -np.sin(theta[0])])
	t_dir1 = rot_axis(e[1] - e[0], e[2] - e[1], theta[1])
	t_dir1 = t_dir1 / np.linalg.norm(t_dir1)
	# e[6] = e[1] + l[1][6] * np.array([np.cos(theta[0] + theta[1]) * np.cos(np.pi / 2 + phi[0]), np.cos(theta[0] + theta[1]) * np.sin(np.pi / 2 + phi[0]), -np.sin(theta[0] + theta[1])])
	e[6] = e[1] + l[1][6] * t_dir1
	t_dir2 = rot_axis(e[6] - e[1], e[2] - e[1], theta[6])
	t_dir2 = t_dir2 / np.linalg.norm(t_dir2)
	# e[7] = e[6] + l[6][7] * np.array([np.cos(theta[0] + theta[1] + theta[6]) * np.cos(np.pi / 2 + phi[0]), np.cos(theta[0] + theta[1] + theta[6]) * np.sin(np.pi / 2 + phi[0]), -np.sin(theta[0] + theta[1] + theta[6])])
	e[7] = e[6] + l[6][7] * t_dir2

	# index finger
	e[8] = e[2] + l[2][8] * np.array([np.cos(theta[2]) * np.cos(phi_I + phi[2]), np.cos(theta[2]) * np.sin(phi_I + phi[2]), -np.sin(theta[2])])
	e[9] = e[8] + l[8][9] * np.array([np.cos(theta[2] + theta[8]) * np.cos(phi_I + phi[2]), np.cos(theta[2] + theta[8]) * np.sin(phi_I + phi[2]), -np.sin(theta[2] + theta[8])])
	e[10] = e[9] + l[9][10] * np.array([np.cos(theta[2] + theta[8] + theta[9]) * np.cos(phi_I + phi[2]), np.cos(theta[2] + theta[8] + theta[9]) * np.sin(phi_I + phi[2]), -np.sin(theta[2] + theta[8] + theta[9])])

	# middle finger
	e[11] = e[3] + l[3][11] * np.array([np.cos(theta[3]) * np.cos(phi_M + phi[3]), np.cos(theta[3]) * np.sin(phi_M + phi[3]), -np.sin(theta[3])])
	e[12] = e[11] + l[11][12] * np.array([np.cos(theta[3] + theta[11]) * np.cos(phi_M + phi[3]), np.cos(theta[3] + theta[11]) * np.sin(phi_M + phi[3]), -np.sin(theta[3] + theta[11])])
	e[13] = e[12] + l[12][13] * np.array([np.cos(theta[3] + theta[11] + theta[12]) * np.cos(phi_M + phi[3]), np.cos(theta[3] + theta[11] + theta[12]) * np.sin(phi_M + phi[3]), -np.sin(theta[3] + theta[11] + theta[12])])

	# ring finger
	e[14] = e[4] + l[4][14] * np.array([np.cos(theta[4]) * np.cos(phi_R + phi[4]), np.cos(theta[4]) * np.sin(phi_R + phi[4]), -np.sin(theta[4])])
	e[15] = e[14] + l[14][15] * np.array([np.cos(theta[4] + theta[14]) * np.cos(phi_R + phi[4]), np.cos(theta[4] + theta[14]) * np.sin(phi_R + phi[4]), -np.sin(theta[4] + theta[14])])
	e[16] = e[15] + l[15][16] * np.array([np.cos(theta[4] + theta[14] + theta[15]) * np.cos(phi_R + phi[4]), np.cos(theta[4] + theta[14] + theta[15]) * np.sin(phi_R + phi[4]), -np.sin(theta[4] + theta[14] + theta[15])])

	# little finger
	e[17] = e[5] + l[5][17] * np.array([np.cos(theta[5]) * np.cos(phi_L + phi[5]), np.cos(theta[5]) * np.sin(phi_L + phi[5]), -np.sin(theta[5])])
	e[18] = e[17] + l[17][18] * np.array([np.cos(theta[5] + theta[17]) * np.cos(phi_L + phi[5]), np.cos(theta[5] + theta[17]) * np.sin(phi_L + phi[5]), -np.sin(theta[5] + theta[17])])
	e[19] = e[18] + l[18][19] * np.array([np.cos(theta[5] + theta[17] + theta[18]) * np.cos(phi_L + phi[5]), np.cos(theta[5] + theta[17] + theta[18]) * np.sin(phi_L + phi[5]), -np.sin(theta[5] + theta[17] + theta[18])])
	e_local = e.copy()

	### transition from local coordinate to global coordinate: rotate and translation
	finger_idlist = np.argwhere(touch_id == 1)
	finger_id = finger_idlist[-1][0]
	shift = e[finger_dict[finger_id]]
	e = e - shift
	# print(e[finger_dict[finger_id]])
	# step1: rotate around x
	tipvec = e[finger_dict[finger_id] - 1] - e[finger_dict[finger_id]]
	# print(tipvec)
	vx = tipvec[0]
	vy = tipvec[1]
	vz = tipvec[2]
	alpha1 = in_angle(np.array([0,vy,vz]), np.array([0,0,1]), np.array([1,0,0]))
	R_1x = np.array([[1.0,0.0,0.0],[0.0, np.cos(alpha1), -np.sin(alpha1)], [0.0, np.sin(alpha1), np.cos(alpha1)]])
	e = R_1x.dot(e.T).T
	# print(e[finger_dict[finger_id]] - e[finger_dict[finger_id] - 1])

	# step2: rotate around y
	tipvec = e[finger_dict[finger_id] - 1] - e[finger_dict[finger_id]]
	vx = tipvec[0]
	vy = tipvec[1]
	vz = tipvec[2]
	beta1 = in_angle(np.array([vx,0,vz]), np.array([1,0,0]), np.array([0,1,0]))
	R_1y = np.array([[np.cos(beta1), 0.0, np.sin(beta1)], [0.0,1.0,0.0], [-np.sin(beta1), 0.0, np.cos(beta1)]])
	e = R_1y.dot(e.T).T
	# print(e[finger_dict[finger_id] - 1] - e[finger_dict[finger_id]])

	# calculate rotation angles
	pitch = tipangles[finger_id][0]
	yaw = tipangles[finger_id][1]
	# print(pitch, yaw)
	if hand_id == 0:
		yaw = -yaw
	uz = -1.0 if abs(pitch) > np.pi / 2 else 1.0
	ux = uz * np.tan(np.pi - yaw)
	uy = uz * np.tan(np.pi - pitch)
	u = np.array([ux,uy,uz])
	# print(u)
	alpha2 = in_angle(np.array([0,uy,uz]), np.array([0,0,1]), np.array([1,0,0]))
	RR_2x = np.array([[1.0,0.0,0.0],[0.0, np.cos(alpha2), -np.sin(alpha2)], [0.0, np.sin(alpha2), np.cos(alpha2)]])
	u = RR_2x.dot(u)
	# print(u)
	beta2 = in_angle(np.array([u[0],0,u[2]]), np.array([1,0,0]), np.array([0,1,0]))
	RR_2y = np.array([[np.cos(beta2), 0.0, np.sin(beta2)], [0.0,1.0,0.0], [-np.sin(beta2), 0.0, np.cos(beta2)]])
	u = RR_2y.dot(u)
	# print(u)
	
	R_2x = np.array([[1.0,0.0,0.0], [0.0, np.cos(-alpha2), -np.sin(-alpha2)], [0.0, np.sin(-alpha2), np.cos(-alpha2)]])
	R_2y = np.array([[np.cos(-beta2), 0.0, np.sin(-beta2)], [0.0,1.0,0.0], [-np.sin(-beta2), 0.0, np.cos(-beta2)]])
	e = (R_2x.dot(R_2y.dot(e.T))).T
	# print(e[finger_dict[finger_id] - 1] - e[finger_dict[finger_id]])

	# step 3: rotate around tip by roll angle
	roll = tipangles[finger_id][2]
	if hand_id != 0:
		roll = -roll
	# print(roll)
	e1 = e[finger_dict[finger_id] - 1] - e[finger_dict[finger_id] - 2]
	if finger_id == 0:
		e1 = e[finger_dict[finger_id] - 1] - e[1]
	e2 = e[finger_dict[finger_id]] - e[finger_dict[finger_id] - 1]
	ncur = np.cross(np.cross(e1,e2),e2)
	if np.linalg.norm(np.cross(e1,e2)) < 1e-6:
		ncur = np.cross(e[2] - e[0], e[3] - e[0])
		if hand_id == 0:
			ncur = -ncur
	
	ndx = 1.0 if roll > 0 else -1.0
	ndy = ndx / (np.tan(roll) + 1e-6)
	ndz = -(e2[0] * ndx + e2[1] * ndy) / (e2[2] + 1e-6)
	ndst = np.array([ndx,ndy,ndz])
	# print(e2)
	# print(ncur, ndst)
	# print(np.dot(e2,ncur), np.dot(e2,ndst))
	th = in_angle(ncur, ndst, e2)
	for i in range(e.shape[0]):
		e[i] = rot_axis(e[i], e2, th)
	# print(e[finger_dict[finger_id] - 1] - e[finger_dict[finger_id]])
	# print(np.dot(np.cross(e[finger_dict[finger_id]] - e[finger_dict[finger_id] - 1], e[finger_dict[finger_id] - 1] - e[finger_dict[finger_id] - 2]), ndst))

	tipcoord = tippos[finger_id] * 0.1
	e_global = e + tipcoord.T
	
	return e_local, e_global


def rigid_transform_3D(A, B):
	assert len(A) == len(B)
	N = A.shape[0]
	mu_A = np.mean(A, axis=0)
	mu_B = np.mean(B, axis=0)

	AA = A - np.tile(mu_A, (N, 1))
	BB = B - np.tile(mu_B, (N, 1))
	H = np.dot(np.transpose(AA), BB)

	U, S, Vt = np.linalg.svd(H)
	R = np.dot(Vt.T, U.T)

	if np.linalg.det(R) < 0:
		# print("Reflection detected")
		Vt[2, :] *= -1
		R = np.dot(Vt.T, U.T)

	t = -np.dot(R, mu_A.reshape(-1,1)) + mu_B.reshape(-1,1)

	return R, t


def transform_to_global(gt, pred, e_gt, e_pred):
	assert int(gt[0][0]) == int(pred[0][0]), "handpose unmatch"
	finger_dict = {0:7, 1:10, 2:13, 3:16, 4:19}
	l = np.array([3.0,2.0,2.0,2.0,2.0])
	pose_id = int(gt[0][0])
	hand_id = int(gt[0][1])
	touch_ind = [i for i, x in enumerate(HANDPOSE_DICT[pose_id].split()) if x == '1']
	touch_num = len(touch_ind)
	ini_points = np.zeros((2*touch_num, 3))
	if touch_num == 1:
		ini_points = np.zeros((3,3))
	tg_points = np.zeros_like(ini_points)
	for i, ti in enumerate(touch_ind):
		# tip points
		ini_points[i] = e_pred[finger_dict[ti]]
		# tg_points[i] = gt[ti*5+5] * 0.1
		tg_points[i] = e_gt[finger_dict[ti]]

		# dip points
		ini_points[i+touch_num] = e_pred[finger_dict[ti] - 1]
		gt_pitch = gt[ti*5+4][0] * np.pi / 180.0
		gt_yaw = gt[ti*5+4][1] * np.pi / 180.0
		gt_roll = gt[ti*5+4][2] * np.pi / 180.0
		pred_roll = pred[ti*5+4][2] * np.pi / 180.0
		if hand_id == 0:
			gt_yaw = -gt_yaw
			gt_roll = -gt_roll
		uz = -1.0 if abs(gt_pitch) > np.pi / 2 else 1.0
		ux = uz * np.tan(np.pi - gt_yaw)
		uy = uz * np.tan(np.pi - gt_pitch)
		u = np.array([ux,uy,uz])
		u = u / np.linalg.norm(u)
		# tg_points[i+touch_num] = tg_points[i] + l[ti] * u
		tg_points[i+touch_num] = e_gt[finger_dict[ti] - 1]

		# auxiliary point for 1-finger touch occasions
		if touch_num == 1:
			e2 = e_pred[finger_dict[ti]] - e_pred[finger_dict[ti] - 1]
			ndx = 1.0 if pred_roll > 0 else -1.0
			ndy = ndx / (np.tan(pred_roll) + 1e-6)
			ndz = -(e2[0] * ndx + e2[1] * ndy) / (e2[2] + 1e-6)
			n_pred = np.array([ndx,ndy,ndz])
			n_pred = n_pred / np.linalg.norm(n_pred)
			ini_points[2] = ini_points[1] + n_pred

			e2 = e_gt[finger_dict[ti]] - e_gt[finger_dict[ti] - 1]
			ndx = 1.0 if gt_roll > 0 else -1.0
			ndy = ndx / (np.tan(gt_roll) + 1e-6)
			ndz = -(e2[0] * ndx + e2[1] * ndy) / (e2[2] + 1e-6)
			n_gt = np.array([ndx,ndy,ndz])
			n_gt = n_gt / np.linalg.norm(n_gt)
			tg_points[2] = tg_points[1] + n_gt

	
	# fit transform matrix
	R, t = rigid_transform_3D(ini_points, tg_points)
	res_points = (np.dot(R, ini_points.T) + t).T
	# fig = plt.figure()
	# ax = fig.add_subplot(1,1,1, projection='3d')
	# ax.scatter(ini_points[:,0], ini_points[:,1], ini_points[:,2], label='变换前点集')
	# ax.scatter(tg_points[:,0], tg_points[:,1], tg_points[:,2], c='r', marker='x', label='目标点集')
	# ax.scatter(res_points[:,0], res_points[:,1], res_points[:,2], c='g', marker='o',label='变换后点集')
	# plt.show()

	# rmse = np.average(np.sqrt(np.sum(np.square(res_points - tg_points), axis=1)))
	# print(rmse)

	e_pred_trans = (np.dot(R, e_pred.T) + t).T
	return e_pred_trans


def xzy_to_xyz(e):
	e_x = e[:,0]
	e_y = e[:,1]
	e_z = e[:,2]
	e_new = np.zeros_like(e)
	e_new[:,0] = -e_x
	e_new[:,1] = -e_z
	e_new[:,2] = -e_y
	return e_new

def rot_to_mat(r, theta):
    I = np.eye(3)
    Sn = np.array([
        [0., -r[2], r[1]],
        [r[2], 0., -r[0]],
        [-r[1], r[0], 0.]
    ])

    r_T = r.reshape(3,1)
    r = r.reshape(1,3)
    R = I * np.cos(theta) + np.dot(r.T, r) * (1 - np.cos(theta)) + Sn * np.sin(theta)

    return R

def mat_to_rot(R):
    theta = np.arccos((R.trace() - 1.) * 0.5)
    Sn = (R - R.T) * 0.5 / np.sin(theta)
    r = np.array([-Sn[1][2], Sn[0][2], -Sn[0][1]])

    return r, theta