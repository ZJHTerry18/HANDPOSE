import sys
sys.path.append('..')
import numpy as np
import os
from tqdm import tqdm
import copy
import pickle
from loguru import logger
import os.path as osp
from config import cfg
from tool.handmodel import get_skeleton_from_data
from tool.visualize import vis
from tool.dataload import load_dataset
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from loss import eloss, loss_stat

HANDPOSE_DICT = cfg.HANDPOSE_DICT
NUM_POSE = cfg.NUM_POSE
TRAIN_PATH = '../dataset/train_type'
TEST_PATH = '../dataset/test_type_new'
LOSS_TYPE = 1
SAVE_FIGURE = True
WRITE_RESULT = True
SAVE_DIR = osp.join('..', 'results', '_'.join(['quadnew','1','type']))
TEST_ONLY = False
MODEL_PATH = '../pkls/quadratic_model.pkl'


def prepare_polyinput(dataset):
    a_ind = [4,9,14,19,24]
    p_ind = [5,10,15,20,25]
    num = len(dataset)
    u = np.zeros((num, 20)) # (p1,p2,...p5,s1,...,s5) pi=(pitch_i, yaw_i, roll_i)
    # u = np.zeros((num, 30)) # (p1,p2,...p5,s1,...,s5) pi=(pitch_i, yaw_i, roll_i, x_i, y_i)
    y = np.zeros((num, 20)) # (t1,t2,...t5) t_i=(pip_theta,pip_phi,iip_theta,dip_theta)
    for i in range(num):
        dat = dataset[i].copy()
        pose_id = int(dat[0][0])
        touch_ind = [i for i, x in enumerate(HANDPOSE_DICT[pose_id].split()) if x == '1']

        # standarization
        yaw_bx = dat[a_ind][touch_ind][0,1]
        x_bx = dat[p_ind][touch_ind][0,0]
        y_bx = dat[p_ind][touch_ind][0,2]
        for ti in touch_ind:
            dat[a_ind[ti], 1] -= yaw_bx
            dat[p_ind[ti], 0] -= x_bx
            dat[p_ind[ti], 2] -= y_bx

        for ti in touch_ind:
            u[i][ti*3:(ti+1)*3] = dat[a_ind[ti]] * np.pi / 180.0 # p_i = (0.,0.,0.) if no touch
            # u[i][ti*5:ti*5+3] = dat[a_ind[ti]] * np.pi / 180.0
            # u[i][ti*5+3] = dat[a_ind[ti] + 1][0] * 0.01
            # u[i][ti*5+4] = dat[a_ind[ti] + 1][2] * 0.01
        for j in range(15,20):
            u[i][j] = 1.0 if (j - 15) in touch_ind else 0.0
        
        for j in range(5):
            y[i][j*4] = dat[j*5+1][0] * np.pi / 180.0
            y[i][j*4+1] = dat[j*5+1][1] * np.pi / 180.0
            y[i][j*4+2] = dat[j*5+2][0] * np.pi / 180.0
            y[i][j*4+3] = dat[j*5+3][0] * np.pi / 180.0
        
    return u, y


def fit_quadratic_model(u, y):
    model = [LinearRegression() for _ in range(20)]
    loss = 0.0
    for i in range(20):
        logger.info('fitting model %d/20...' % (i+1))
        y_rav = y[:,i].ravel()
        model[i].fit(u, y_rav)
        y_pred = model[i].predict(u)
        loss += np.linalg.norm(y_rav - y_pred, ord=1)
    logger.info('model fit finished. Model loss: %.4f' % (loss * 180.0 / np.pi / y.size))
    
    return model


if __name__ == "__main__":
    if not osp.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    if not TEST_ONLY:
        train_set = load_dataset(TRAIN_PATH)
        u_train, y_train = prepare_polyinput(train_set)
        
        po = PolynomialFeatures(degree=2, interaction_only=False, include_bias=True)
        u_train_poly = po.fit_transform(u_train)

        quadmodel = fit_quadratic_model(u_train_poly, y_train)
        with open('quadratic_model.pkl', 'wb') as f:
            pickle.dump(quadmodel, f)
    else:
        with open(MODEL_PATH, 'rb') as f:
            quadmodel = pickle.load(f)
    
    test_set = load_dataset(TEST_PATH)
    u_test, y_test = prepare_polyinput(test_set)

    po = PolynomialFeatures(degree=2, interaction_only=False, include_bias=True)
    u_test_poly = po.fit_transform(u_test)
    y_pred = np.zeros_like(y_test)
    for i in range(20):
        y_pred[:,i] = quadmodel[i].predict(u_test_poly)
    logger.info('prediction finished. Model avg angle prediction loss: %.4f' % 
        (np.average(np.abs(y_pred - y_test)) * 180.0 / np.pi))

    pred_set = copy.deepcopy(test_set)

    losses = dict([(k, []) for k in range(2 * NUM_POSE)])
    all_losses = []
    logger.info('calculating loss and saving results...')
    for i in tqdm(range(len(pred_set))):
        dat = pred_set[i]
        for j in range(5):
            dat[j*5+1][0] = y_pred[i][j*4] * 180.0 / np.pi
            dat[j*5+1][1] = y_pred[i][j*4+1] * 180.0 / np.pi
            dat[j*5+2][0] = y_pred[i][j*4+2] * 180.0 / np.pi
            dat[j*5+3][0] = y_pred[i][j*4+3] * 180.0 / np.pi
        
        hand_id = int(dat[0][1])
        pose_id = int(dat[0][0])
        seq = int(dat[0][2])
        touch_ind = [i for i, x in enumerate(HANDPOSE_DICT[pose_id].split()) if x == '1']
        
        e_test_local, e_test_global = get_skeleton_from_data(test_set[i])
        e_pred_local, e_pred_global = get_skeleton_from_data(pred_set[i])

        lossval = eloss(e_test_local, e_pred_local, touch_ind, all=(LOSS_TYPE == 0))
        losses[hand_id * NUM_POSE + pose_id].append(lossval)
        all_losses.append(lossval)

        if SAVE_FIGURE:    
            save_figname = '_'.join([str(pose_id).zfill(2), str(hand_id), str(seq).zfill(3)]) + '.jpg'
            vis(dat, e_test_local, e_test_global, e_pred_local, e_pred_global, 
                None, None, show=False, save=True, save_dir=SAVE_DIR, save_fig=save_figname)
    
    loss_stat(NUM_POSE, all_losses, losses, None, None, WRITE_RESULT, SAVE_DIR)
    