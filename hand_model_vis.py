import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
import os
import os.path as osp
import cv2
# from tqdm import tqdm

def plot_kp(ax,xp,yp,zp):
	palm_color = 'blue'
	finger_color = 'violet'
	ax.plot(np.hstack((xp[0],xp[1])),
			np.hstack((yp[0],yp[1])),
			np.hstack((zp[0],zp[1])),
			ls='-', color=finger_color)
	ax.plot(np.hstack((xp[0],xp[2])),
			np.hstack((yp[0],yp[2])),
			np.hstack((zp[0],zp[2])),
			ls='-', color=palm_color)
	ax.plot(np.hstack((xp[0],xp[3])),
			np.hstack((yp[0],yp[3])),
			np.hstack((zp[0],zp[3])),
			ls='-', color=palm_color)
	ax.plot(np.hstack((xp[0],xp[4])),
			np.hstack((yp[0],yp[4])),
			np.hstack((zp[0],zp[4])),
			ls='-', color=palm_color)
	ax.plot(np.hstack((xp[0],xp[5])),
			np.hstack((yp[0],yp[5])),
			np.hstack((zp[0],zp[5])),
			ls='-', color=palm_color)
	ax.plot(np.hstack((xp[1],xp[6:8])),
			np.hstack((yp[1],yp[6:8])),
			np.hstack((zp[1],zp[6:8])),
			ls='-', color=finger_color)
	ax.plot(np.hstack((xp[2],xp[8:11])),
			np.hstack((yp[2],yp[8:11])),
			np.hstack((zp[2],zp[8:11])),
			ls='-', color=finger_color)
	ax.plot(np.hstack((xp[3],xp[11:14])),
			np.hstack((yp[3],yp[11:14])),
			np.hstack((zp[3],zp[11:14])),
			ls='-', color=finger_color)
	ax.plot(np.hstack((xp[4],xp[14:17])),
			np.hstack((yp[4],yp[14:17])),
			np.hstack((zp[4],zp[14:17])),
			ls='-', color=finger_color)
	ax.plot(np.hstack((xp[5],xp[17:20])),
			np.hstack((yp[5],yp[17:20])),
			np.hstack((zp[5],zp[17:20])),
			ls='-', color=finger_color)


# ### define Mode ###
# parser = argparse.ArgumentParser()
# parser.add_argument("--manual", help="input angles from terminal", action="store_true")
# parser.add_argument("--handtype", type=str, choices=["l", "r"], help="choose left or right hand")
# parser.add_argument("--data_dir", type=str, help="path of txt file")
# args = parser.parse_args()


def vis(samples, vis_label, save_dir):
    # samples is B, 5, 4
    B = samples.shape[0]
    for b in range(B):
        per_sample = samples[b,...]
        per_label = vis_label[b,...]
        title = ','.join(str(int(i)) for i in per_label)
        ### save the result in plt figure
        # save_dir = 'demo_test2_p3\\'
        os.makedirs(save_dir,exist_ok=True)

        ### define parameters ###
        phi_I = 110 * np.pi / 180
        phi_M = 90 * np.pi / 180
        phi_R = 70 * np.pi / 180
        phi_L = 50 * np.pi / 180
        # if args.handtype == "l": # default right hand
        # 	phi_I = np.pi - phi_I
        # 	phi_M = np.pi - phi_M
        # 	phi_R = np.pi - phi_R
        # 	phi_L = np.pi - phi_L

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

        ### input ###
        phi = np.zeros(6)
        theta = np.zeros(20)

        # ind_dict_1 = {1:0, 5:2, 9:3, 13:4, 17:5}
        # ind_dict_2 = {2:1, 6:8, 10:11, 14:14, 18:17}
        # ind_dict_3 = {3:6, 7:9, 11:12, 15:15, 19:18}
        # for i in range(len(lines)):
        #     if i % 4 == 1:
        #         line = lines[i].split()
        #         phi[ind_dict_1[i]] = float(line[0])
        #         theta[ind_dict_1[i]] = float(line[1])
        #     elif i % 4 == 2:
        #         theta[ind_dict_2[i]] = float(lines[i])
        #     elif i % 4 == 3:
        #         theta[ind_dict_3[i]] = float(lines[i])
        

        phi[[0,2,3,4,5]] = per_sample[:,0]
        theta[[0,2,3,4,5]] = per_sample[:,1]
        theta[[1,8,11,14,17]] = per_sample[:,2]
        theta[[6,9,12,15,18]] = per_sample[:,3]
        # for data in per_sample:
            


        # print(phi)
        # print(theta)

        # phi = phi * np.pi / 180
        # theta = theta * np.pi / 180

        alpha = 0
        gamma = 0

        # if args.handtype == "l":
        # 	phi = -phi

        ### calculate joint positions ###
        e = np.zeros((20,3)) # get the position, forward kinematics, set the plane as the standard

        # four digit PIPs
        e[2] = e[0] + l[0][2] * np.array([np.cos(phi_I), np.sin(phi_I), 0])
        e[3] = e[0] + l[0][3] * np.array([np.cos(phi_M), np.sin(phi_M), 0])
        e[4] = e[0] + l[0][4] * np.array([np.cos(phi_R), np.sin(phi_R), 0])
        e[5] = e[0] + l[0][5] * np.array([np.cos(phi_L), np.sin(phi_L), 0])

        # thumb
        e[1] = e[0] + l[0][1] * np.array([np.cos(theta[0]) * np.cos(np.pi / 2 + phi[0]), np.cos(theta[0]) * np.sin(np.pi / 2 + phi[0]), -np.sin(theta[0])])
        e[6] = e[1] + l[1][6] * np.array([np.cos(theta[0] + theta[1]) * np.cos(np.pi / 2 + phi[0]), np.cos(theta[0] + theta[1]) * np.sin(np.pi / 2 + phi[0]), -np.sin(theta[0] + theta[1])])
        e[7] = e[6] + l[6][7] * np.array([np.cos(theta[0] + theta[1] + theta[6]) * np.cos(np.pi / 2 + phi[0]), np.cos(theta[0] + theta[1] + theta[6]) * np.sin(np.pi / 2 + phi[0]), -np.sin(theta[0] + theta[1] + theta[6])])

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

        # rotation of palm
        R_a = np.array([[1.0,0.0,0.0],[0.0, np.cos(alpha), -np.sin(alpha)], [0.0, np.sin(alpha), np.cos(alpha)]])
        R_g = np.array([[np.cos(gamma), -np.sin(gamma), 0.0], [np.sin(gamma), np.cos(gamma), 0.0], [0.0,0.0,1.0]])
        e = (R_g.dot(R_a.dot(e.T))).T
        xp = e.T[0].T
        yp = e.T[1].T
        zp = e.T[2].T

        ### visualization: draw 3d model ###
        # ax = plt.axes(projection='3d')
        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(1,2,1, projection='3d')
        ax2 = fig.add_subplot(1,2,2, projection='3d')
        ax.set_xlim3d([-10,10])
        ax.set_ylim3d([0,20])
        ax.set_zlim3d([-10,10])
        ax2.set_xlim3d([-10,10])
        ax2.set_ylim3d([0,20])
        ax2.set_zlim3d([-10,10])
        ax.view_init(elev=45, azim=45)
        # ax.view_init(elev=90, azim=90)
        ax2.view_init(elev=0, azim=0)
        # # direct coordinations
        # xp = [8.97,0.944,2.70,4.78,6.72,-2.11,-3.78,-4.93,0.256,-1.20,-2.13,3.64,4.56,5.39,5.24,5.75,6.57]
        # yp = [13.5,12.6,12.5,12.3,11.7,11.2,10.2,9.36,10.9,9.51,8.35,8.17,8.12,9.09,8.66,8.44,9.14]
        # zp = [8.55,2.50,1.35,0.767,0.538,0.00766,-1.28,-2.12,-2.30,-4.21,-5.26,1.63,4.15,5.44,1.08,2.90,4.17]
        ax.scatter3D(xp, yp, zp, cmap="Greens")
        ax2.scatter3D(xp, yp, zp, cmap="Greens")

        # plot connections
        plot_kp(ax,xp, yp, zp)
        plot_kp(ax2,xp, yp, zp)
        save_fig = f'{b:0>3d}.jpg'
        plt.title(title,fontsize='large',fontweight='bold') 
        plt.savefig(osp.join(save_dir,save_fig))
        plt.close()

    save_file = save_dir[:-1] + '.mp4'
    save_video(save_dir, save_file)

def save_video(img_root, output_path):
    # img_root = 'demo_test2\\'
    FPS = 12
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    size = (960, 480)
    VideoWriter = cv2.VideoWriter(output_path,fourcc, FPS, size)
    imgs = os.listdir(img_root)
    for img in imgs:
        frame = cv2.imread(os.path.join(img_root, img))
        VideoWriter.write(frame)
    VideoWriter.release()