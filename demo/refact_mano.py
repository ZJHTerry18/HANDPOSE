import sys
sys.path.append('..')

from cgitb import small
import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
from PyQt5.QtWidgets import QPushButton,QApplication, QMainWindow
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.widgets.RawImageWidget import RawImageWidget
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QUrl
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
import random
from string import digits, ascii_uppercase, ascii_lowercase
chars = digits + ascii_uppercase + ascii_lowercase
chars = [chars[i] for i in range(0,len(chars))]

from PIL import Image
from scipy.spatial.transform import Rotation as R
from obj_reader import parse_obj
import os
import pickle
from convert import euler_to_axis, euler_to_4x4matrix
import multiprocessing as mp
import time
import io
import matplotlib.pyplot as plt
import mano
from mano.utils import Mesh
import torch
from loguru import logger
import glob

# local import
from config import cfg
from tool.handmodel import mat_to_rot, rigid_transform_3D, rot_to_mat

IMG_HEAD_OFFSET = 12
IMAGE_PRE_WIDTH = 600
IMAGE_PRE_HEIGHT = 480
IMAGE_PRE_SIZE = IMAGE_PRE_WIDTH * IMAGE_PRE_HEIGHT
USE_REG_ONLY = True

IMG_PATH = r'D:\Workspace\HANDPOSE\demo\images\pd2'
PKL_PATH = r'D:\Workspace\HANDPOSE\demo\output\pd\pd2\preddata.pkl'

HANDPOSE_DICT = cfg.HANDPOSE_DICT
joint_dict = {12:1, 13:2, 14:3, 0:6, 1:7, 2:8, 3:11, 4:12, 5:13, 9:16, 10:17, 11:18, 6:21, 7:22, 8:23}

class dataset():
    def __init__(self,):
        self.img_seqs = []
        self.predict_seqs = []
        self.curseq = 0
        with open(PKL_PATH, 'rb') as f:
            angledata = pickle.load(f)
        imgfiles = sorted(glob.glob(os.path.join(IMG_PATH, '*.jpg')))
        # imgfiles = imgfiles[:]
        # angledata = angledata[2000:]
        for i, dat in enumerate(angledata):
            img = np.array(Image.open(imgfiles[i])).swapaxes(0,1)
            predict = {}
            predict['head'] = dat['data'][0]
            hand_id = int(predict['head'][1])
            predict['pose'] = np.zeros((15,3))
            for j in [0,3,6,9,12]:
                predict['pose'][j][0] = dat['data'][joint_dict[j]][2]
                predict['pose'][j][1] = dat['data'][joint_dict[j]][0]
                predict['pose'][j][2] = dat['data'][joint_dict[j]][1]
            for j in [1,2,4,5,7,8,10,11]:
                predict['pose'][j][2] = dat['data'][joint_dict[j]][0]
            if hand_id == 0:
                predict['pose'][:,2] *= -1.0
                predict['pose'][:,1] *= -1.0
            predict['pose'] = predict['pose'] * np.pi / 180.0
            predict['rotmat'], predict['offset'] = rigid_transform_3D(dat['local pose'], dat['global pose'])

            self.img_seqs.append(img)
            self.predict_seqs.append(predict)
        self.hand = 'left' if hand_id == 0 else 'right'
    
    def getitem(self):
        img = self.img_seqs[self.curseq]
        pred = self.predict_seqs[self.curseq]
        self.curseq += 1
        
        return {'img':img, 'pred':pred, 'seq':self.curseq}


D = dataset()
init_img = np.ones((800, 750),dtype=np.uint8) * 255
last_angle = init_angle = [0, 0, 0]

app = QtWidgets.QApplication([])
cw = QtWidgets.QWidget()
cw.setGeometry(10, 10, 1600, 800)
cw.setWindowTitle("fingerpose demo")  # 主窗口
layout = QtWidgets.QHBoxLayout(cw)

glv = gl.GLViewWidget(cw)
glv.opts["azimuth"] = -20
glv.setCameraPosition(distance=40)

# mesh data initialize
if D.hand == 'left':
    h_model = mano.load(model_path=r'D:\Workspace\MANO\models\mano\MANO_LEFT.pkl',
                        is_right= True,
                        num_pca_comps=45,
                        batch_size=1,
                        flat_hand_mean=True)
    ini_betas = torch.rand(1, 10)*.1
    angles = torch.zeros((15,3))
    ini_pose = angles.view(1,-1)
    ini_global_orient_r = torch.tensor([[0.5773, 0.5773, 0.5773]])
    ini_global_orient_th = 2.094
    ini_global_orient = ini_global_orient_r * ini_global_orient_th
    ini_transl        = torch.tensor([[0.0959,-0.0064,-0.0061]])
else:
    h_model = mano.load(model_path=r'D:\Workspace\MANO\models\mano\MANO_RIGHT.pkl',
                        is_right= True,
                        num_pca_comps=45,
                        batch_size=1,
                        flat_hand_mean=True)
    ini_betas = torch.rand(1, 10)*.1
    angles = torch.zeros((15,3))
    ini_pose = angles.view(1,-1)
    ini_global_orient_r = torch.tensor([[0.5773, -0.5773, -0.5773]])
    ini_global_orient_th = 2.094
    ini_global_orient = ini_global_orient_r * ini_global_orient_th
    ini_transl        = torch.tensor([[-0.0956,-0.0064,-0.0062]])

output = h_model(betas=ini_betas,
                  global_orient=ini_global_orient,
                  hand_pose=ini_pose,
                  transl=ini_transl,
                  return_verts=True,
                  return_tips = True)
h_meshes = h_model.hand_meshes(output)
j_meshes = h_model.joint_meshes(output)
hj_meshes = Mesh.concatenate_meshes([h_meshes[0], j_meshes[0]])

vertices = h_meshes[0].vertices * 100
faces = h_meshes[0].faces
colors = np.zeros((vertices.shape[0], 4), dtype=float)
theta = np.arctan2(vertices[:, 2], vertices[:, 1])
theta = (theta / np.pi + 1) / 2.0  # 0 to 1
colors[:, 2] = (vertices[:, 0] - vertices[:, 0].min()) / (vertices[:, 0].max() - vertices[:, 0].min())
colors[:, 2] = (colors[:, 2] + 0.3) * 0.4
colors[:, 0] = (theta + 0.2) * 0.5
colors[:, 1] = np.linspace(0.8, 0.2, colors.shape[0])
md1 = gl.MeshData(vertexes=vertices, faces=faces, vertexColors=colors, faceColors=colors)
m1 = gl.GLMeshItem(meshdata=md1)
glv.addItem(m1)

# Grid view
m2 = gl.GLGridItem(color=(255, 255, 255, 100))
m2.scale(2, 2, 1)
glv.addItem(m2)

num = 20000
pos = np.empty((num, 3))
size = np.empty((num))
color = np.empty((num, 4))
pos[0] = (1, 0, 0)
size[0] = 0.5
color[0] = (1.0, 0.0, 0.0, 0.5)
pos[1] = (0, 1, 0)
size[1] = 0.5
color[1] = (0.0, 0.0, 1.0, 0.5)
pos[2] = (0, 0, 1)
size[2] = 0.5
color[2] = (0.0, 1.0, 0.0, 0.5)
z = 1.0
d = 6.0
step = 1.0 / num
for i in range(3, num):
    if i % 3 == 0:
        pos[i] = (0, 0, z)
        color[i] = (0.0, 1.0, 0.0, 0.5)
    elif i % 3 == 1:
        pos[i] = (0, z, 0)
        color[i] = (0.0, 0.0, 1.0, 0.5)
    else:
        pos[i] = (z, 0, 0)
        color[i] = (1.0, 0.0, 0.0, 0.5)
    size[i] = 2.0 / d
    z = z - step
    d *= 1.5

sp1 = gl.GLScatterPlotItem(pos=pos, size=size, color=color, pxMode=False)
sp1.translate(0, 0, 0)  # 调整起始obj位置
sp1.scale(15, 15, 15)
glv.addItem(sp1)

# img view
imv = RawImageWidget()
imv.setImage(init_img)
imv.sizeHint = glv.sizeHint = lambda: pg.QtCore.QSize(100, 100)
glv.setSizePolicy(imv.sizePolicy())

def update_mano(pose, global_orient, transl):
    pose = torch.tensor(pose).view(1,-1).to(torch.float32)
    global_orient = torch.tensor(global_orient).unsqueeze(0).to(torch.float32)
    transl = transl.to(torch.float32)
    output = h_model(betas=ini_betas,
                  global_orient=global_orient,
                  hand_pose=pose,
                  transl=transl,
                  return_verts=True,
                  return_tips = True)
    h_meshes = h_model.hand_meshes(output)
    j_meshes = h_model.joint_meshes(output)
    hj_meshes = Mesh.concatenate_meshes([h_meshes[0], j_meshes[0]])

    vertices = h_meshes[0].vertices * 100
    faces = h_meshes[0].faces
    colors = np.zeros((vertices.shape[0], 4), dtype=float)
    theta = np.arctan2(vertices[:, 2], vertices[:, 1])
    theta = (theta / np.pi + 1) / 2.0  # 0 to 1
    colors[:, 2] = (vertices[:, 0] - vertices[:, 0].min()) / (vertices[:, 0].max() - vertices[:, 0].min())
    colors[:, 2] = (colors[:, 2] + 0.3) * 0.4
    colors[:, 0] = (theta + 0.2) * 0.5
    colors[:, 1] = np.linspace(0.8, 0.2, colors.shape[0])
    
    return vertices, faces, colors


def format(id,roll,pitch,yaw):
    return f"{id}: roll: {roll:.2f}\t pitch: {pitch:.2f}\t yaw: {yaw:.2f}"

def update():
    # global index,last_angle,STATES,m1
    global STATES
    if STATES == "running":
        data = D.getitem()
        if data['img'] is not None:
            imv.setImage(data["img"])
            pred = data['pred']
            seq = data['seq']
            cur_pose = pred['pose']
            rmat_1 = rot_to_mat(ini_global_orient_r.squeeze(0).numpy(), ini_global_orient_th)
            rmat_2 = pred['rotmat']
            rmat = np.dot(rmat_2, rmat_1)
            global_orient_r, global_orient_th = mat_to_rot(rmat)
            cur_global_orient = global_orient_r * global_orient_th
            cur_transl = ini_transl + pred['offset'].reshape(1,-1) * 0.01
            vertices, faces, colors = update_mano(cur_pose, cur_global_orient, cur_transl)
            md1 = gl.MeshData(vertexes=vertices, faces=faces, vertexColors=colors, faceColors=colors)
            kwds = {'meshdata':md1}
            m1.setMeshData(**kwds)
            shell.addItem(str(seq))
            shell.scrollToBottom()
        else:
            shell.addItem("invalid data")
            shell.scrollToBottom()
            # print("invalid data")
        QtCore.QTimer.singleShot(20, update)  # for schedule


def control():
    global STATES, button1
    print("clicked")
    if STATES == "start":
        STATES = "running"
        button1.setText("running")
        update()
        return
    if STATES == "running":
        STATES = "stopped"
        button1.setText("stopped")
        return
    if STATES == "stopped":
        STATES = "running"
        button1.setText("running")
        update()
        return
    if STATES == "ended":
        STATES = "running"
        button1.setText("running")
        update()
        return


button1 = QPushButton(cw)
button1.setText("Start")
button1.clicked.connect(control)

STATES = "start"  # running, ended, stopped,start
shell = QtWidgets.QListWidget()
shell.setFixedSize(700, 150)
left_win = QtWidgets.QWidget()
left_layout = QtWidgets.QVBoxLayout(left_win)
left_layout.addWidget(imv)
left_layout.addWidget(shell)

layout.addWidget(left_win)
# layout.addWidget(imv2)
layout.addWidget(glv)
layout.addWidget(button1)
cw.show()

if __name__ == "__main__":
    app.exec_()
