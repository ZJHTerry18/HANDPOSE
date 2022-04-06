import numpy as np
import torch
import pickle
import torch.nn as nn
import torch.utils.data as data
import glob
import os
import sys
import cv2
sys.path.append('/home/panzhiyu/project/HandPose/Prior/')
from hand_model_vis import vis

namespace = ['t','i','m','r','p']
namespace_mapping = {'t':0,'i':1,'m':2,'r':3,'p':4}

class FingerPrint(data.Dataset):
    def __init__(self, folder_images, folder_motion, annotation_file, is_training):
        super(FingerPrint, self).__init__()
        # fusing different types
        self.image_size = np.array([800,750]) # just test the fix size
        self.is_training = is_training
        self.folder_image = folder_images
        self.persons = os.listdir(folder_images)  #TODO: fix one data
        self.hand_type = ['left','right']
        self.touchtype = os.listdir(os.path.join(folder_images, self.persons[0], self.hand_type[0]))
        self.touchtype.sort(key = lambda x: int(x)) # in-place operation
        self.folder_motion = folder_motion
        with open(annotation_file, 'rb' ) as f:
            self.annotations = pickle.load(f)
        # fix length
        self.persons_files = len(self.hand_type) * len(self.touchtype) * 150
        self.hands_files = len(self.touchtype) * 150
        self.type_files = 150
        total_len = len(self.persons) * len(self.hand_type) * len(self.touchtype) * 150 # 150 samples per type
        self.gap = 0
        train_len = int(total_len * 0.7)
        if is_training:
            self.len = train_len
            self.gap = 0
        else:
            self.len = total_len - train_len
            self.gap = train_len

    def __len__(self):
        return self.len
        
    def __getitem__(self, index):
        # processing the idx
        index = index + self.gap
        p_idx = index // self.persons_files
        person = self.persons[p_idx]
        hand_idx = (index - p_idx * self.persons_files) // self.hands_files
        hand_type = self.hand_type[hand_idx]
        type_idx = (index - p_idx * self.persons_files - hand_idx * self.hands_files) // self.type_files
        touchtype = self.touchtype[type_idx]
        fig_idx = (index - p_idx * self.persons_files - hand_idx * self.hands_files - type_idx * self.type_files)
        # get the label info and other pics
        image_idx = touchtype + '_' + f'{fig_idx:0>3d}_*.png'
        images_files = glob.glob(os.path.join(self.folder_image, person, hand_type, touchtype, image_idx))
        image_name = []
        # label_collection = []
        center_collection = []
        type_collection = []
        images_collection = []
        for im in images_files:
            image_name.append(im.split('/')[-1][:-4])
            label = self.annotations[person][hand_type][touchtype][image_name[-1]] # get the position, center and finger type
            center_collection.append(torch.tensor(label['center']).reshape(1,2) / self.image_size)
            type_collection.append(torch.tensor(namespace_mapping[label['type']]).reshape(1,1)) # get the finger type           
            # read the gray image
            image = cv2.imread(im, cv2.IMREAD_GRAYSCALE)
            image = image.astype(np.float32) / 255.0   # normalize
            image = torch.from_numpy(image).unsqueeze(0)
            images_collection.append(image)
        type_collection = torch.cat(type_collection, dim=0)
        center_collection = torch.cat(center_collection, dim=0)
        images_collection = torch.cat(images_collection, dim=0)
        # padding data to 5
        if len(images_collection) < 5:
            pad_num = 5 - len(images_collection)
            type_collection = torch.cat([type_collection, -1 * torch.ones(pad_num, 1)], dim=0)
            center_collection = torch.cat([center_collection, -1 * torch.ones(pad_num, 2)], dim=0)
            images_collection = torch.cat([images_collection, torch.zeros(pad_num, 256, 256)], dim=0)
        # get the motion
        motion_folder = os.path.join(self.folder_motion, person, hand_type + '_pickle')
        motion_file = os.path.join(motion_folder, touchtype + '_' + f'{fig_idx:0>3d}.pkl')
        with open(motion_file, 'rb') as f:
            motion = pickle.load(f)
        # split the data
        cond = torch.from_numpy(motion['cond']).to(torch.float32)
        # curl = torch.from_numpy(readin_data['curl']) # add the curl paras
        finger_ang = torch.zeros(len(namespace), 4) # 4 is the angle dim per finger
        for idx, name in enumerate(namespace):
            finger_ang[idx:idx+1,:] = torch.from_numpy(motion[name] * np.pi / 180)

        # sort other data according to the type_collection value
        images = -1 * torch.ones_like(images_collection, dtype=torch.float32)
        centers = -1 * torch.ones_like(center_collection, dtype=torch.float32)
        types = -1 * torch.ones_like(type_collection, dtype=torch.float32)
        for s in range(5):
            current_idx = int(type_collection[s])
            if current_idx == -1:
                break
            images[current_idx,...] = images_collection[s,...]
            centers[current_idx,...] = center_collection[s,...]
            types[current_idx,...] = type_collection[s,...]

        return images, finger_ang, centers, types, cond

if __name__ == '__main__':
    folder_images = '/Extra/panzhiyu/finger/datasetfp_v0.4/fingerprint_single'
    folder_motion = '/Extra/panzhiyu/finger/datasetfp_v0.4/leap' 
    annotation_file = '/Extra/panzhiyu/finger/datasetfp_v0.4/position_info_type.pkl'
    test_dataset = FingerPrint(folder_images, folder_motion, annotation_file, True)
    images, finger_ang, center, types, cond = test_dataset[1201]