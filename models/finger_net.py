# resnet_50_gat_finger_net.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.gnn_vae import GAT_VAE
import sys
import argparse
sys.path.append('/home/panzhiyu/project/HandPose/Prior/')
from thirdparty.swish import Swish
from torch.nn import TransformerEncoder, TransformerEncoderLayer

BN_MOMENTUM = 0.1

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

# create a resnet block
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class resnet50(nn.Module):
    def __init__(self, latent_feature_num=1000):
        super(resnet50, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)  # in_channels, out_channels, kernel_size, stride, padding; the image is in gray scale
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        self.globalpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * Bottleneck.expansion, latent_feature_num)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        
        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.globalpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

# class resnet1d(nn.Module):
#     def __init__(self, latent_feature_num=1000):
#         super(resnet1d, self).__init__()
#         self.inplanes = 64
#         self.conv1 = nn.Conv1d(2, 64, kernel_size=7, stride=2, padding=3,
#                                bias=False)  # in_channels, out_channels, kernel_size, stride, padding; the image is in gray scale
#         self.bn1 = nn.BatchNorm1d(64, momentum=BN_MOMENTUM)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)


class finger_gat_embeding_model(nn.Module):
    def __init__(self, args, resnet_file = None, gat_file = None, init_feature_dims = 128, device = torch.device('cpu')):
        super().__init__()
        self.device = device
        self.init_feature_dims = init_feature_dims
        self.args = args
        self.resnet = resnet50(init_feature_dims)
        self.gat_vae = GAT_VAE(args.input_dims, args.latent_dims, 8, args.num_headers, device = device) # input length = 2, latent dims = 8, hidden_dims = 8 ,num of heads = 4, device
        if resnet_file is not None:
            self.resnet.load_state_dict(torch.load(resnet_file)) # load pretrained resnet50, only contains the model state_dict
        if gat_file is not None:
            self.gat_vae.load_state_dict(torch.load(gat_file, map_location='cpu')['state_dict']) # load pretrained gat_vae, only contains the model state_dict
        # layers for positional embedding, not embedding, direct add into the feature
        # pos_dim_list = [2, 8, 16, init_feature_dims]
        # self.pos_embedding = self.make_pos_embedding(pos_dim_list)
        self.feature_fusing = nn.Sequential(nn.Linear(init_feature_dims + 2, init_feature_dims, bias=True), Swish())
        # gat embedding initialization
        ## set the placeholder for the input of gat embedding
        self.dropout = nn.Dropout(0.2)
        self.finger_repre = nn.Parameter(torch.randn(1,5,init_feature_dims).to(device), requires_grad=True) # 5 finger t, i, m, r, p
        encoder_layer = TransformerEncoderLayer(d_model=init_feature_dims, nhead=4, batch_first=True) # batch, seq, feature_dims
        self.feature_encoder = TransformerEncoder(encoder_layer, num_layers=2) # TODO: just set 2 layers
        self.embed_mu = nn.Sequential(nn.Linear(5 * init_feature_dims, init_feature_dims, bias=True), self.dropout, Swish(),\
            nn.Linear(init_feature_dims, init_feature_dims // 2, bias=True), self.dropout, Swish(),\
                nn.Linear(init_feature_dims // 2, args.latent_dims, bias=True),  Swish()) # squish and embed
        # self.embed_logl = nn.Sequential(nn.Linear(5 * init_feature_dims, args.latent_dims, bias=True), Swish())
      
        
    # def _get_covnet(self):
    #     return nn.Sequential(nn.Conv1d(2, 2, kernel_size=1, bias=True),  Swish(),\
    #                     nn.Conv1d(2, 2, kernel_size=3, padding=1, bias=True), Swish(),\
    #                     nn.Conv1d(2, 1, kernel_size=3, padding=1, bias=True),Swish()) # fusing all the channels

    # def make_pos_embedding(self, dim_l):
    #     layers = []
    #     length = len(dim_l)
    #     # layers.append(nn.BatchNorm1d(dim_l[0]))
    #     for idx in range(length - 1):
    #         layers.append(nn.Sequential(nn.Linear(dim_l[idx], dim_l[idx + 1]), Swish())) # no batch_norm  nn.BatchNorm1d(dim_l[idx+1]), 
    #     return nn.Sequential(*layers)
        
    def forward(self, images, centers, types, motions):
        # image is B x 5 x 1 x W x H
        # center is B x 5 x 2
        # types is B x 5 x 1 # TODO: add one dimention
        # motion is B x 5 x 4
        batch_size = images.shape[0]
        image_size = images.shape[-1]
        images_input = images.reshape(-1,1,image_size, image_size)
        images_features = self.resnet(images_input)
        images_features = images_features.reshape(batch_size, 5, self.init_feature_dims) # image
        # position_embeding = self.pos_embedding(centers)
        cat_feature = torch.cat([images_features, centers],dim=-1)
        fusing_feature = self.feature_fusing(cat_feature)
        # There are two choices, first, using the default input; second, uisng the placeholder
        input_feature = self.finger_repre.repeat(batch_size, 1, 1) # B X 5 X F
        org_mask = (types == -1)
        input_feature = input_feature * org_mask + fusing_feature * (~org_mask)
        cross_feature = self.feature_encoder(input_feature)
        cross_feature_c = [cross_feature[:,k, :] for k in range(5)]
        cross_feature_c = torch.cat(cross_feature_c, dim=-1)
        finger_mu = self.embed_mu(cross_feature_c)
        # finger_logl = self.embed_logl(cross_feature_c)
        # finger_sigma = torch.exp(0.5* finger_logl)
        # seed = torch.randn_like(finger_sigma).to(self.device)
        # sample_finger = finger_mu + seed * finger_sigma
        finger_reconstruct = self.gat_vae._decode_part(finger_mu)
        # run the gat net
        gat_reconstruct, kl_loss, gat_mu, gat_logl = self.gat_vae(motions)
        # gat_sigma = torch.exp(0.5* gat_logl)
        # calculate the KL distance between finger and gat gaussian distribution
        # kl_loss = 0.5 * torch.mean(torch.sum((finger_mu - gat_mu) ** 2 + (finger_sigma/gat_sigma) ** 2 - torch.log((finger_sigma/gat_sigma) ** 2) - 1, dim=-1))
        feature_dist = 0.5 * torch.mean(torch.norm(finger_mu - gat_mu, dim=-1, p=2))

        return finger_reconstruct, feature_dist, gat_reconstruct, kl_loss


if __name__ == '__main__': 
    parser = argparse.ArgumentParser('VAE')
    parser.add_argument('--latent_dims', type=int, default=8,
                        help='latent_dims')
    parser.add_argument('--input_dims', type=int, default=2,
                        help='input_dim')
    parser.add_argument('--num_headers', type=int, default=4,
                        help='headers')
    args = parser.parse_args()
    gat_file = 'gat_curl/best_model.pt'

    net = finger_gat_embeding_model(args, gat_file = gat_file)
    images = torch.rand(50,5,1,256,256)
    centers = torch.rand(50,5,2)
    types = torch.rand(50,5,1)
    motions = torch.rand(50,5,6)
    re, loss = net(images, centers, types, motions)
