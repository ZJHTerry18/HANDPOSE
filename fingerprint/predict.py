import sys
sys.path.append("..")

from tqdm import tqdm
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
import pickle
import pandas as pd
# from dataset.normalize import normalize_img
# from dataset.resize import resize_img
# from dataset.loader import load_img
# from dataset import CustomDatasetDataLoader

from hopenet import Hopenet_multi_frame
from loguru import logger

N = 5

model = Hopenet_multi_frame([2, 2, 2, 2], 60, 5)
model.eval()
ckp = torch.load("best.pth.tar", lambda storage, _: storage)
model.load_state_dict(ckp["model"])
model.cuda()

def readseq(seqpath):
    seq = [None] * N
    i = 0
    for path in seqpath:
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (224,224), interpolation=cv2.INTER_CUBIC)
        img = cv2.normalize(img, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # cv2.imshow('image', img)
        # cv2.waitKey()
        seq[i] = img
        i += 1
    return torch.tensor(np.array(seq))

class JiaheSet(Dataset):
    def __init__(self, data):
        self._data = data

    def __getitem__(self,index):
        _item = self._data[index]
        path = _item['path']
        seqpath = _item['seqpath']
        seq = readseq(seqpath)
        n, h, w = seq.shape
        seq = seq.view(n, 1, h, w)
        # seq = list(map(partial(load_img,dim=3),seq))
        # seq = np.stack(seq)
        return {"path":path,"imgs":seq}

    def __len__(self,):
        return len(self._data)

# class JiaheLoader(CustomDatasetDataLoader):
#     def __init__(self,*args,**kwargs):
#         super().__init__(*args,**kwargs)
#     def __iter__(self,):
#         for _, item in enumerate(self.dataloader):
#             imgs = item['imgs']
#             B,N,*remain = imgs.shape
#             assert N==5
#             imgs = imgs.view(B*N,*remain)
#             imgs = resize_img(imgs,224,'bilinear','kornia','img',True)
#             imgs = normalize_img(imgs,'min-max')
#             imgs = imgs.view(B,N,1,224,224) # type: ignore
#             yield {"path":item["path"],"imgs":imgs}


if __name__ == "__main__":
    with open('fingerprint.pkl', 'rb') as f:
        data = pickle.load(f)
    logger.info('dataset preparing...')
    dataset = JiaheSet(data)
    # data_loader = JiaheLoader(JiaheSet(data),is_training=False,num_workers=4,batch_size=32,target_size=224)
    logger.info('dataloader preparing...')
    data_loader = DataLoader(dataset, batch_size=32)
    result = []
    logger.info('predicting...')
    with torch.no_grad():
        for i,item in tqdm(enumerate(data_loader)):
            path = item['path']
            imgs = item['imgs'].cuda()
            yaw_prob, pitch_prob, roll_prob = model(imgs)

            yaw_predict = torch.sum(torch.softmax(yaw_prob, -1) * model.backbone.idx_tensor_yaw.unsqueeze(0).unsqueeze(0), dim=-1)
            pitch_predict = torch.sum(torch.softmax(pitch_prob, -1) * model.backbone.idx_tensor_pitch.unsqueeze(0).unsqueeze(0), dim=-1)
            roll_predict = torch.sum(torch.softmax(roll_prob, -1) * model.backbone.idx_tensor_roll.unsqueeze(0).unsqueeze(0), dim=-1)

            yaw_predict = yaw_predict.mean(-1).cpu().numpy().tolist()
            pitch_predict = pitch_predict.mean(-1).cpu().numpy().tolist()
            roll_predict = roll_predict.mean(-1).cpu().numpy().tolist()
            _df = pd.DataFrame.from_dict({
                "yaw":yaw_predict,
                "pitch":pitch_predict,
                "roll":roll_predict,
                "path":path,
            })
            result.append(_df)

    result = pd.concat(result)
    print(len(result))
    pd.to_pickle(result,'./result.pkl')
