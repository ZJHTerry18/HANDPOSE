import os
import os.path as osp
import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from dataload import load_dataset
from loguru import logger
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from handmodel import get_skeleton_from_data
from visualize import vis

DATA_PATH = '../dataset/train/txts'
VAE_DATA_PATH = 'D:\Workspace\LeapMotion\leapHandpose\leapHandpose\\nvae_results\\nvae_samples.pkl'
GAT_DATA_PATH = 'D:\Workspace\LeapMotion\leapHandpose\leapHandpose\\nvae_results\\gat_samples.pkl'
SAVE_PKL_PATH = '../results/pkls/tsne'
METHOD = 'tSNE'
DIM = 2

with open(VAE_DATA_PATH, 'rb') as f:
    vae_data = pickle.load(f)
with open(GAT_DATA_PATH, 'rb') as f:
    gat_data = pickle.load(f)

def read_pkldata(datapath):
    ind = [0,1,2,3,6,7,8,11,12,13,16,17,18,21,22,23]
    with open(datapath, 'rb') as f:
        data = pickle.load(f)
    dat_set = []
    seq = 0
    for x in data:
        x_new = np.zeros((26,3))
        for i in range(5):
            x_new[i*5+1][0] = x[i][0] * 180.0 / np.pi
            x_new[i*5+1][1] = x[i][1] * 180.0 / np.pi
            x_new[i*5+2][0] = x[i][2] * 180.0 / np.pi
            x_new[i*5+3][0] = x[i][3] * 180.0 / np.pi
        dat_set.append(x_new)
        # e_local, e_global = get_skeleton_from_data(x_new)
        # savefig = str(seq).zfill(5) + '.jpg'
        # vis(x_new, e_local, e_global, e_local, e_global, show=False, save=True, save_dir='../results/nvae_result', save_fig=savefig)
        # seq += 1
    
    x = np.array([k[ind][1:].ravel() for k in dat_set])
    y_curl = np.array([curl_type(x) for x in dat_set])

    return dat_set, x, y_curl

def curl_type(x):
    ind = [(2,3),(7,8),(12,13),(17,18),(22,23)]
    curl_num = 0
    for i1,i2 in ind:
        angles = x[i1-1][1] + x[i1][0] + x[i2][0]
        if angles > 120:
            curl_num += 1

    if curl_num == 0:
        return 0
    else:
        return 1


def decomposition(x, method = 'tSNE', dim = 2):
    if method == 'PCA':
        pca = PCA(n_components=dim)
        return pca, pca.fit_transform(x)
    elif method == 'tSNE':
        tsne = TSNE(n_components=dim, init='pca')
        return tsne, tsne.fit_transform(x)

def vis_decomp(x, y, hand, method):
    dim = x.shape[1]
    assert dim == 2 or dim == 3
    color = ['r', 'b', 'g', 'y', 'k', 'c']
    label = ['no-curl', 'curl', 'vae no-curl', 'vae curl', 'gat no-curl', 'gat curl']
    alpha = [1.0, 1.0, 0.05, 0.05]

    if dim == 2:
        plt.figure()
        plt.xlim(-125,250)
        plt.ylim(-100,125)
        tag_list = np.unique(y).tolist()
        for i in tag_list:
            plt.scatter(x[y == i,0], x[y == i,1], c=color[i], label=label[i])
        plt.legend()
        # plt.title('%s of %s hand (2-dimension)' % (method, hand))
        plt.show()
    elif dim == 3:
        fig = plt.figure()
        ax = Axes3D(fig)
        for i in range(5):
            ax.scatter(x[y == i, 0], x[y == i,1], x[y == i,2], c=color[i], label=label[i])
        # plt.legend()
        plt.title('%s of %s hand (3-dimension)' % (method, hand))
        plt.show()

def do_tsne():
    ### tsne for train dataset ###
    dataset = load_dataset(DATA_PATH)
    dataset_left = []
    dataset_right = []

    ind = [0,1,2,3,6,7,8,11,12,13,16,17,18,21,22,23]
    y_dict = dict([(k,0) for k in range(5)] + [(k,1) for k in range(5,14)] +
        [(k,2) for k in range(14,21)] + [(k,3) for k in range(21,25)] + [(25,4)])
    # print(y_dict)
    # print(dataset[0][ind])
    for x in dataset:
        if int(x[0][1]) == 0:
            dataset_left.append(x)
        else:
            dataset_right.append(x)

    x_gt = np.array([k[ind][1:].ravel() for k in dataset])
    y_gt_curl = np.array([curl_type(x) for x in dataset])

    logger.info('decompositing training data')

    # model_x_left, new_x_left = decomposition(x_left, method=METHOD, dim=DIM)
    # model_x_right, new_x_right = decomposition(x_right, method=METHOD, dim=DIM)
    model_x_gt, new_x_gt = decomposition(x_gt, method=METHOD, dim=DIM)
    with open(osp.join(SAVE_PKL_PATH, 'x_tsne_gt.pkl'), 'wb') as f:
        pickle.dump([new_x_gt, y_gt_curl], f)

    ## visualize some points
    vis_decomp(new_x_gt, y_gt_curl, hand='right', method=METHOD)

    ### tsne for nvae results
    vae_set, x_vae, y_vae_curl = read_pkldata(VAE_DATA_PATH)
    gat_set, x_gat, y_gat_curl = read_pkldata(GAT_DATA_PATH)

    
    logger.info('decompositing only vae results')
    # only nvae result
    model_vaeonly, new_x_vaeonly = decomposition(x_vae, method=METHOD, dim=DIM)
    with open(osp.join(SAVE_PKL_PATH, 'x_tsne_vae.pkl'), 'wb') as f:
        pickle.dump([new_x_vaeonly, y_vae_curl], f)
    # vis_decomp(new_x_vaeonly, y_vae_curl + 2, hand='right', method=METHOD)

    # only gat result
    logger.info('decompositing only gat results')
    model_gatonly, new_x_gatonly = decomposition(x_gat, method=METHOD, dim=DIM)
    with open(osp.join(SAVE_PKL_PATH, 'x_tsne_gat.pkl'), 'wb') as f:
        pickle.dump([new_x_gatonly, y_gat_curl], f)
    # vis_decomp(new_x_gatonly, y_gat_curl + 4, hand='right', method=METHOD)

    logger.info('decompositing all results altogether')
    # plot vae tsne result in train dataset tsne result
    x_all = np.concatenate((x_gt, x_vae, x_gat), axis=0)
    y_all_curl = np.concatenate((y_gt_curl, y_vae_curl + 2, y_gat_curl + 4), axis=0)
    __ , new_x_all = decomposition(x_all, method=METHOD, dim=DIM)
    with open(osp.join(SAVE_PKL_PATH, 'x_tsne_all.pkl'), 'wb') as f:
        pickle.dump([new_x_all, y_all_curl], f)
    # vis_decomp(new_x_vaeandall[:x_all.shape[0]], y_vaeandall_curl[:x_all.shape[0]], hand='right', method=METHOD)
    # vis_decomp(new_x_vaeandall[x_all.shape[0]:], y_vaeandall_curl[x_all.shape[0]:], hand='right', method=METHOD)

def tsne_vis():
    gt_set = load_dataset(DATA_PATH)
    vae_set, __, __ = read_pkldata(VAE_DATA_PATH)
    gat_set, __, __ = read_pkldata(GAT_DATA_PATH) 
    with open(osp.join(SAVE_PKL_PATH, 'x_tsne_gt.pkl'), 'rb') as f:
        new_x_gt, y_gt_curl = pickle.load(f)
    with open(osp.join(SAVE_PKL_PATH, 'x_tsne_vae.pkl'), 'rb') as f:
        new_x_vae, y_vae_curl = pickle.load(f)
    with open(osp.join(SAVE_PKL_PATH, 'x_tsne_gat.pkl'), 'rb') as f:
        new_x_gat, y_gat_curl = pickle.load(f)
    with open(osp.join(SAVE_PKL_PATH, 'x_tsne_all.pkl'), 'rb') as f:
        new_x_all, y_all_curl = pickle.load(f)
    
    # vis_decomp(new_x_all, y_all_curl, hand='right', method=METHOD)
    # vis_decomp(new_x_vae, y_vae_curl, hand='right', method=METHOD)
    # vis_decomp(new_x_vaeandall, y_vaeandall_curl, hand='right', method=METHOD)

    # vis_decomp(new_x_all[:new_x_gt.shape[0]], y_all_curl[:new_x_gt.shape[0]], hand='right', method=METHOD)
    # vis_decomp(new_x_all[new_x_gt.shape[0]:new_x_gt.shape[0]+new_x_vae.shape[0]], y_all_curl[new_x_gt.shape[0]:new_x_gt.shape[0]+new_x_vae.shape[0]], hand='right', method=METHOD)
    # vis_decomp(new_x_all[new_x_gt.shape[0]+new_x_vae.shape[0]:], y_all_curl[new_x_gt.shape[0]+new_x_vae.shape[0]:], hand='right', method=METHOD)

    for i in range(new_x_all.shape[0]):
        xi = new_x_all[i][0]
        yi = new_x_all[i][1]

        if i in range(new_x_gt.shape[0]):
            j = i
            if int(gt_set[j][0][1]) == 0:
                if 80 > xi > 70 and 100 > yi > 80:
                    e_local, e_global = get_skeleton_from_data(gt_set[j])
                    vis(gt_set[j], e_local, e_global, e_local, e_global, show=True)
            
        elif i in range(new_x_gt.shape[0], new_x_gt.shape[0] + new_x_vae.shape[0]):
            # j = i - new_x_gt.shape[0]
            # if 80 > xi > 70 and 100 > yi > 80:
            #     e_local, e_global = get_skeleton_from_data(vae_set[j])
            #     vis(vae_set[j], e_local, e_global, e_local, e_global, show=True)
            pass
            
        else:
            # j = i - new_x_gt.shape[0] - new_x_vae.shape[0]
            # if 0 > xi > -5 and 0 > yi > -5:
            #     e_local, e_global = get_skeleton_from_data(gat_set[j])
            #     vis(gat_set[j], e_local, e_global, e_local, e_global, show=True)
            pass

def tsne_search(query_list):
    gt_set = load_dataset(DATA_PATH)
    vae_set, __, __ = read_pkldata(VAE_DATA_PATH)
    gat_set, __, __ = read_pkldata(GAT_DATA_PATH) 
    with open(osp.join(SAVE_PKL_PATH, 'x_tsne_gt.pkl'), 'rb') as f:
        new_x_gt, y_gt_curl = pickle.load(f)
    with open(osp.join(SAVE_PKL_PATH, 'x_tsne_vae.pkl'), 'rb') as f:
        new_x_vae, y_vae_curl = pickle.load(f)
    with open(osp.join(SAVE_PKL_PATH, 'x_tsne_gat.pkl'), 'rb') as f:
        new_x_gat, y_gat_curl = pickle.load(f)
    with open(osp.join(SAVE_PKL_PATH, 'x_tsne_all.pkl'), 'rb') as f:
        new_x_all, y_all_curl = pickle.load(f)

    save_dir = '../results/graphs/gt_samples'
    for queryfile in query_list:
        query = [int(x) for x in queryfile.split('_')]
        for i, dat in enumerate(gt_set):
            if query == [int(x) for x in dat[0]]:
                dat[0][1] = 0.0
                x_gt_i = new_x_all[i][0]
                y_gt_i = new_x_all[i][1]
                curl_gt_i = y_all_curl[i]
                print(x_gt_i, y_gt_i, curl_gt_i)
                e_local, e_global = get_skeleton_from_data(dat)
                savefig = 'x' + str(int(x_gt_i)) + 'y' + str(int(y_gt_i)) +  'c' + str(curl_gt_i) + '_' + queryfile + '.jpg'
                vis(dat, e_local, e_global, e_local, e_global, show=False, save=True, save_dir=save_dir, save_fig=savefig)
    

if __name__ == "__main__":
    # do_tsne()
    tsne_vis()
    query_list = [
        '00_0_0209', '01_0_0061', '01_0_0737', '02_0_0865', '02_0_0897', '02_1_0119', '03_0_0580', '04_0_0675'
        '05_1_0824', '07_0_0880', '07_0_0884', '08_0_0261', '08_0_0734', '09_0_0813', '09_0_0814', '09_0_0817'
        '09_1_0052', '09_1_0148', '09_1_0163', '15_0_0642', '15_0_0734', '16_1_0620', '20_0_0139', '21_0_0432'
        '25_0_0109', '25_0_0233'
    ]
    # tsne_search(query_list)
