import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import argparse
import os
import os.path as osp
from tqdm import tqdm
import cv2

# HANDPOSE_DICT = ["1 0 0 0 0", "0 1 0 0 0", "0 0 1 0 0", "0 0 0 1 0", "0 0 0 0 1",
# 	"1 1 0 0 0", "0 1 1 0 0", 
# 	"1 0 1 0 0", "1 0 0 1 0", "1 0 0 0 1", "0 1 0 1 0", "0 1 0 0 1", "0 0 1 1 0", "0 0 0 1 1",
# 	"1 1 1 0 0", "0 1 1 1 0", "0 0 1 1 1", 
# 	"1 1 0 1 0", "1 0 1 1 0", "1 0 0 1 1", "1 1 0 0 1",
# 	"0 1 1 1 1", "1 0 1 1 1",
# 	"1 1 0 1 1", "1 1 1 1 0", 
# 	"1 1 1 1 1"]
HANDPOSE_DICT = ["0 1 0 0 0", "1 1 0 0 0", "0 1 1 0 0", "0 1 0 1 0", "0 1 0 0 1",
"1 1 1 0 0", "1 1 0 1 0", "1 1 0 0 1", "1 1 1 1 0", "1 1 1 1 1"]

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

def get_skeleton_from_data(data):
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


def vis_and_save_result(hand_id, pose_id, e_local_test, e_global_test, e_local_res, e_global_res, e_local_res2 = None, e_global_res2 = None, show = False, save = False, save_dir = None, save_fig = None):
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
	# axres1 = fig.add_subplot(2,2,3, projection='3d')
	# axres2 = fig.add_subplot(2,2,4, projection='3d')
	axtest1.set_title('test local')
	axtest2.set_title('test global')
	# axres1.set_title('pred local')
	# axres2.set_title('pred global')
	axtest1.set_xlim3d([-10,10])
	axtest1.set_ylim3d([0,20])
	axtest1.set_zlim3d([-10,10])
	axtest2.set_xlim3d([-10,10])
	axtest2.set_ylim3d([10,30])
	axtest2.set_zlim3d([-10,10])
	axtest2.set_xlim3d([-10,10])
	# axres1.set_xlim3d([-10,10])
	# axres1.set_ylim3d([0,20])
	# axres1.set_zlim3d([-10,10])
	# axres2.set_xlim3d([-10,10])
	# axres2.set_ylim3d([10,30])
	# axres2.set_zlim3d([-10,10])
	# axres2.set_xlim3d([-10,10])
	axtest1.set_xticklabels([])
	axtest1.set_yticklabels([])
	axtest1.set_zticklabels([])
	# axres1.set_xticklabels([])
	# axres1.set_yticklabels([])
	# axres1.set_zticklabels([])
	axtest2.set_xticklabels([])
	axtest2.set_yticklabels([])
	axtest2.set_zticklabels([])
	# axres2.set_xticklabels([])
	# axres2.set_yticklabels([])
	# axres2.set_zticklabels([])
	# ax.view_init(elev=15, azim=-30)
	axtest1.view_init(elev=45, azim=45)
	axtest2.view_init(elev=-45, azim=-90)
	# axres1.view_init(elev=45, azim=45)
	# axres2.view_init(elev=-45, azim=-90)
	
	# axtest1.scatter3D(xp_local_test, yp_local_test, zp_local_test, cmap="Greens")
	# axtest2.scatter3D(xp_global_test, yp_global_test, zp_global_test, cmap="Greens")
	# axres1.scatter3D(xp_local_res, yp_local_res, zp_local_res, cmap="Greens")
	# axres2.scatter3D(xp_global_res, yp_global_res, zp_global_res, cmap="Greens")
	
	# plot connections
	plot_kp(axtest1,xp_local_test, yp_local_test, zp_local_test, touch_ind)
	plot_kp(axtest2,xp_global_test, yp_global_test, zp_global_test, touch_ind)
	plot_kp(axtest1,xp_local_res, yp_local_res, zp_local_res, touch_ind, linestyle='--')
	plot_kp(axtest2,xp_global_res, yp_global_res, zp_global_res, touch_ind, linestyle='--')
	plot_kp(axtest1,xp_local_res2, yp_local_res2, zp_local_res2, touch_ind, linestyle=':', finger_color='green')
	plot_kp(axtest2,xp_global_res2, yp_global_res2, zp_global_res2, touch_ind, linestyle=':', finger_color='green')
	plt.axis('on')
	if show: 
		plt.show()
	if save:
		plt.savefig(osp.join(save_dir, save_fig))
	plt.close()


def save_video(img_root, output_path):
    # img_root = 'demo_test2\\'
	FPS = 30
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	size = (960, 480)
	VideoWriter = cv2.VideoWriter(output_path,fourcc, FPS, size)
	imgs = os.listdir(img_root)
	for img in tqdm(imgs):
		if img[-4:] == '.jpg':
			frame = cv2.imread(os.path.join(img_root, img))
			VideoWriter.write(frame)
	VideoWriter.release()
