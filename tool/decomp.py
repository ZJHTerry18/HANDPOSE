import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from dataload import load_dataset
from loguru import logger
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from handmodel import get_skeleton_from_data, vis_and_save_result

DATA_PATH = '../dataset/train'
METHOD = 'tSNE'
DIM = 2

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


def decomposition(x, method = PCA, dim = 2):
    if method == 'PCA':
        pca = PCA(n_components=dim)
        return pca.fit_transform(x)
    elif method == 'tSNE':
        tsne = TSNE(n_components=dim, init='pca')
        return tsne.fit_transform(x)

def vis_decomp(x, y, hand, method):
    dim = x.shape[1]
    assert dim == 2 or dim == 3
    color = ['r', 'b', 'g', 'y', 'k']
    label = ['0', '1', '2', '3', '4']

    if dim == 2:
        plt.figure()
        for i in range(5):
            plt.scatter(x[y == i,0], x[y == i,1], c=color[i], label=label[i])
        # plt.legend()
        plt.title('%s of %s hand (2-dimension)' % (method, hand))
        plt.show()
    elif dim == 3:
        fig = plt.figure()
        ax = Axes3D(fig)
        for i in range(5):
            ax.scatter(x[y == i, 0], x[y == i,1], x[y == i,2], c=color[i], label=label[i])
        # plt.legend()
        plt.title('%s of %s hand (3-dimension)' % (method, hand))
        plt.show()

    

if __name__ == "__main__":
    dataset = load_dataset(DATA_PATH)
    dataset_left = []
    dataset_right = []

    ind = [0,1,2,3,4,6,7,8,9,11,12,13,14,16,17,18,19,21,22,23,24]
    y_dict = dict([(k,0) for k in range(5)] + [(k,1) for k in range(5,14)] +
        [(k,2) for k in range(14,21)] + [(k,3) for k in range(21,25)] + [(25,4)])
    # print(y_dict)
    # print(dataset[0][ind])
    for x in dataset:
        if int(x[0][1]) == 0:
            dataset_left.append(x)
        else:
            dataset_right.append(x)

    x_left = np.array([k[ind][1:].ravel() for k in dataset_left])
    y_left = np.array([y_dict[int(k[0][0])] for k in dataset_left])
    y_left_curl = np.array([curl_type(x) for x in dataset_left])
    x_right = np.array([k[ind][1:].ravel() for k in dataset_right])
    y_right = np.array([y_dict[int(k[0][0])] for k in dataset_right])
    y_right_curl = np.array([curl_type(x) for x in dataset_right])

    logger.info('decompositing...')

    new_x_left = decomposition(x_left, method=METHOD, dim=DIM)
    new_x_right = decomposition(x_right, method=METHOD, dim=DIM)

    # visualize some points
    # print(x_left.shape, new_x_left.shape)
    numc = 0
    num = 0
    for i in range(new_x_left.shape[0]):
        x1 = new_x_left[i][0]
        x2 = new_x_left[i][1]
        data = dataset_left[i]
        e_local, e_global = get_skeleton_from_data(data)
        hand_id = int(data[0][1])
        pose_id = int(data[0][0])
        if pose_id in list([0,1,5,6,14,15,16,21,22]):
            if x1 < -150 or x1 > 50 or x2 > 180:
                numc += 1
        # if num % 10 == 0:
        #     vis_and_save_result(hand_id, pose_id, e_local, e_global, e_global, e_local, show=True)
            num += 1
    print(numc, num)

    print(np.sum(y_left_curl == 0), np.sum(y_left_curl == 1), np.sum(y_right_curl == 0), np.sum(y_right_curl == 1))

    ## visualize by touch type
    # vis_decomp(new_x_left, y_left, hand='left', method=METHOD)
    # vis_decomp(new_x_right, y_right, hand='right', method=METHOD)

    # visualize by curl type
    vis_decomp(new_x_left, y_left_curl, hand='left', method=METHOD)
    vis_decomp(new_x_right, y_right_curl, hand='right', method=METHOD)

    # print(x_left.shape, x_right.shape)