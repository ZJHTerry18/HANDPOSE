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

import torch
# local import
import data

from sklearn.linear_model import LinearRegression
# from train.interface import CapativePosePredictor 

IMG_HEAD_OFFSET = 12
IMAGE_PRE_WIDTH = 600
IMAGE_PRE_HEIGHT = 480
IMAGE_PRE_SIZE = IMAGE_PRE_WIDTH * IMAGE_PRE_HEIGHT
USE_REG_ONLY = True

IMG_PATH = r'D:\Workspace\LeapMotion\leapHandpose\leapHandpose\dataset_fptype\fingerprint_single\p1'
PKL_PATH = r'D:\Workspace\LeapMotion\leapHandpose\leapHandpose\dataset_fptype\fingerprint_single\p1\result.pkl'

# M = CapativePosePredictor()


class average_history():
    def __init__(self):
        self._meta = []
        self.average_length = 10
        self.filter_length = 10

    def filter_noise(self,seq):
        # estimate the mean and sigma
        N = seq.shape[0]
        M = np.mean(seq,0,keepdims=True)
        M = np.tile(M,(N,1))
        S = np.std(seq,axis=0,keepdims=True)
        S = np.tile(S,(N,1))
        condition = (seq - M) < S
        return np.where(condition,seq,M)

    def update(self,x):
        self._meta.append(x)
        if len(self._meta)> 100:
            self._meta = self._meta[-100:]
        _len = min(len(self._meta),self.average_length)
        seqs = np.array(self._meta[-_len:])
        lr = LinearRegression()
        x = np.arange(0,_len)[:,np.newaxis] # N,c_in
        lr.fit(x,seqs) # seqs: N,c_out, coef_: c_out,c_in
        result = lr.predict(x)
        if _len>=3:
            return (result[-1,:]*0.4 + result[-2,:]*0.4 + result[-3,:]*0.2)
        else:
            return result[-1]
        # weights = np.ones_like(seqs)
        # weights[-1,:] = 3
        # weights[0,:] = 0.3
        # if abs(x[1])>50:
        #     seqs[-self.filter_length:,2:3] = self.filter_noise(seqs[-self.filter_length:,2:3])
        #     weights[:,1] = np.exp(0.5*np.arange(0,_len))
        #     weights[:,2] = np.exp(0.1*np.arange(0,_len))
        # else:
        #     seqs[-self.filter_length:,1:2] = self.filter_noise(seqs[-self.filter_length:,1:2])
        #     weights[:,1] = np.exp(0.1*np.arange(0,_len))
        #     weights[:,2] = np.exp(0.5*np.arange(0,_len))
        # weights = weights/weights.sum(0,keepdims=True)
        # return (weights*seqs).sum(0)

class dataset():
    def __init__(self,):
        self.img_seqs = []
        self.predict_seqs = []
        self.curseq = 0
        with open(PKL_PATH, 'rb') as f:
            angledata = pickle.load(f)
        for dat in angledata:
            path = dat['path'].split('-')
            finger = dat['finger']
            hand = path[0]
            path = '/'.join(path)
            path = os.path.join(IMG_PATH, path) + '.png'
            img = np.array(Image.open(path)).swapaxes(0,1)
            roll = dat['roll']
            pitch = dat['pitch']
            yaw = dat['yaw']
            if hand == 'right':
                roll = -roll
                yaw = -yaw
            predict = np.array([roll,pitch,yaw])
            self.img_seqs.append(img)
            self.predict_seqs.append(predict)
        # self.predict_seqs = average_history()
        # self.count = 0
        # self.yaw_mids,self.pitch_mids,_ = self.get_interval_mids()

    # def request_fingerprint(self,):
    #     data.FetchData()
    #     raw_arr = data.GetMatrixData()
    #     cropped,display = get_contoured_img(raw_arr)
    #     return cropped,display

    # def plot2array(self,fig,index):
    #     buf = io.BytesIO()
    #     fig.savefig(buf)
    #     buf.seek(0)
    #     img = Image.open(buf)
    #     img.save(f"../r22_flag/{index}_crop.png")
    #     print("index: ",index," saved")
    #     arr = np.array(img)
    #     arr = np.transpose(arr,(1,0,2))
    #     return arr

    # def spec2toStr(self):
    #     prefix = "Specs:\n"
    #     for k,v in self.specs.items():
    #         prefix += f"{k}:{v}\n"
    #     prefix = prefix[:-1]
    #     return prefix

    # def check_fingerprint(self,img):
    #     '''
    #     check if it's None, save for debug
    #     '''
    #     if img is None:
    #         return None
    #     h,w = img.shape
    #     if (img<220).sum()>5:
    #         self.count += 1
    #         # if self.count % 20 == 1:
    #         #     print(f"DEBUG: save img to capture/{self.count}.npy")
    #         #     name = ''.join([random.sample(chars,k=1)[0] for _ in range(6)])
    #         #     np.save(f"./capture/{name}_{self.count}.npy",img)
    #         return img
    #     else:
    #         return None

    # def get_interval_mids(self,):
    #     yaw_interval = (90+90)*1.0/60
    #     yaw_mids = torch.tensor([(i + 0.5 )* yaw_interval + (-90) for i in range(60)])

    #     pitch_interval = (0+90)*1.0/30
    #     pitch_mids = torch.tensor([(i + 0.5 )* pitch_interval + (-90) for i in range(30)])

    #     roll_interval = (90+90)*1.0/60
    #     roll_mids = torch.tensor([(i + 0.5 )* roll_interval + (-90)  for i in range(60)])

    #     return yaw_mids[None,:],pitch_mids[None,:],roll_mids[None,:]

    # def process_img(self,img):
    #     now_h, now_w = img.shape # this is the cropped img
    #     img = (img-img.min())/(img.max()-img.min()) # 0-1
    #     pad_h, pad_w = 32-now_h, 32-now_w
    #     img = np.pad(img,((pad_h//2,32-pad_h//2-now_h),(pad_w//2,32-pad_w//2-now_w)),mode='constant',constant_values = 1)
    #     img = img[np.newaxis,np.newaxis,:,:]
    #     return img

    # def make_predict(self,img):
    #     global M
    #     with torch.no_grad():
    #         yaw_pred,pitch_pred = M(img)
    #         # yaw_pred = torch.sum(self.yaw_mids*torch.softmax(yaw_pred_cls,-1),-1)
    #         # pitch_pred = torch.sum(self.pitch_mids*torch.softmax(pitch_pred_cls,-1),-1)
    #         # print("yaw pred cls: ",yaw_pred_cls[0].argmax().item())
    #         # print("pitch pred cls: ",pitch_pred_cls[0].argmax().item())
    #     return [0,pitch_pred[0].item(),yaw_pred[0].item()] # roll pitch yaw
    
    def getitem(self):
        # cropped, display = self.request_fingerprint()
        # cropped = self.check_fingerprint(cropped)
        # if cropped is None:
        #     return {"img":None,"angle":None}
        # else:
        #     processed = self.process_img(cropped)
        #     predict = self.make_predict(torch.tensor(processed))
        #     # print("before: ",predict)
        #     predict = self.predict_seqs.update(predict)
        #     # print("after: ",predict)
        #     return {"img":display.astype(np.uint8),"angle":predict} # roll pitch yaw
        #     # smoothed_predict = self.make_smooth()
        #     # return {"img":display,"angle":smoothed_predict}
        img = self.img_seqs[self.curseq]
        angle = self.predict_seqs[self.curseq]
        self.curseq += 1
        
        return {'img':img, 'angle':angle}


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

# mesh data
init_vertices, faces = parse_obj('./finger.obj')
masspoint_x = np.mean(init_vertices[:, 0])
masspoint_y = np.min(init_vertices[:, 1])
masspoint_z = np.min(init_vertices[:, 2])
r = R.from_euler("zyx", [-90, 0, 0], degrees=True)
vertices = r.apply(init_vertices - np.array([[masspoint_x, masspoint_y, np.mean(init_vertices[:, 2])]]))
colors = np.zeros((vertices.shape[0], 4), dtype=float)
theta = np.arctan2(vertices[:, 2], vertices[:, 1])
theta = (theta / np.pi + 1) / 2.0  # 0 to 1
colors[:, 2] = (vertices[:, 0] - vertices[:, 0].min()) / (vertices[:, 0].max() - vertices[:, 0].min())
colors[:, 2] = (colors[:, 2] + 0.3) * 0.4
colors[:, 0] = (theta + 0.2) * 0.5
colors[:, 1] = np.linspace(0.8, 0.2, colors.shape[0])

vertices = r.apply(init_vertices - np.array([[masspoint_x, masspoint_y, masspoint_z]]))
md1 = gl.MeshData(vertexes=vertices, faces=faces, vertexColors=colors)
m1 = gl.GLMeshItem(meshdata=md1)

results = euler_to_axis(-init_angle[0], init_angle[1], -init_angle[2])
m1.rotate(*results)
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

# imv2 = RawImageWidget()
# simulate = np.load("./blur_simulate.npy")
# simulate = (simulate*255).astype(np.uint8)
# imv2.setImage(simulate)
# imv2.sizeHint = glv.sizeHint = lambda: pg.QtCore.QSize(100, 100)
# glv.setSizePolicy(imv2.sizePolicy())

def format(id,roll,pitch,yaw):
    return f"{id}: roll: {roll:.2f}\t pitch: {pitch:.2f}\t yaw: {yaw:.2f}"

def update():
    # global index,last_angle,STATES,m1
    global last_angle, STATES
    if STATES == "running":
        data = D.getitem()
        if data['img'] is not None:
            imv.setImage(data["img"])
            now_angle = data["angle"]
            roll,pitch,yaw = now_angle
            text = format("left",roll,pitch,yaw)
            m1.resetTransform()
            results = euler_to_axis(-roll, pitch + 20, -yaw)
            m1.rotate(*results)
            last_angle = now_angle
            shell.addItem(text)
            shell.scrollToBottom()
        else:
            shell.addItem("invalid data")
            shell.scrollToBottom()
            # print("invalid data")
        QtCore.QTimer.singleShot(500, update)  # for schedule


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
