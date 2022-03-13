import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from dataload import load_dataset
from loguru import logger
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

DATA_PATH = '../dataset/train'
METHOD = 'tSNE'
DIM = 3

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
        plt.legend()
        plt.title('%s of %s hand (2-dimension)' % (method, hand))
        plt.show()
    elif dim == 3:
        fig = plt.figure()
        ax = Axes3D(fig)
        for i in range(5):
            ax.scatter(x[y == i, 0], x[y == i,1], x[y == i,2], c=color[i], label=label[i])
        plt.legend()
        plt.title('%s of %s hand (3-dimension)' % (method, hand))
        plt.show()

    

if __name__ == "__main__":
    dataset = load_dataset(DATA_PATH)
    dataset_left = []
    dataset_right = []

    ind = [0,1,2,3,4,6,7,8,9,11,12,13,14,16,17,18,19,21,22,23,24]
    y_dict = dict([(k,0) for k in range(5)] + [(k,1) for k in range(5,14)] +
        [(k,2) for k in range(14,21)] + [(k,3) for k in range(21,25)] + [(25,4)])
    print(y_dict)
    # print(dataset[0][ind])
    for x in dataset:
        if int(x[0][1]) == 0:
            dataset_left.append(x[ind].ravel())
        else:
            dataset_right.append(x[ind].ravel())

    x_left = np.array([k[3:] for k in dataset_left])
    y_left = np.array([y_dict[int(k[0])] for k in dataset_left])
    x_right = np.array([k[3:] for k in dataset_right])
    y_right = np.array([y_dict[int(k[0])] for k in dataset_right])

    logger.info('decompositing...')

    new_x_left = decomposition(x_left, method=METHOD, dim=DIM)
    new_x_right = decomposition(x_right, method=METHOD, dim=DIM)

    vis_decomp(new_x_left, y_left, hand='left', method=METHOD)
    vis_decomp(new_x_right, y_right, hand='right', method=METHOD)

    # print(x_left.shape, x_right.shape)