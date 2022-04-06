from tkinter import SW
import torch
import numpy as np
import torch.nn as nn
import collections
from thirdparty.swish import Swish
# create a mini resnet1d
BN_MOMENTUM = 0.1

class Bottleneck(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes, momentum=BN_MOMENTUM) #
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv1d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = Swish()#nn.ReLU(inplace=True)
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


class CVAE(nn.Module):
    def __init__(self, input_channels, sequence_length, latent_dims, class_dims, device):
        super().__init__()
        # set up the models
        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.class_dims = class_dims
        self.latent_dims = latent_dims

        data_list = [self.sequence_length, self.sequence_length * 2, self.sequence_length * 3, latent_dims]
        label_list = [self.class_dims, self.sequence_length * 2, self.sequence_length * 3, latent_dims]
        self.embed_data = self._get_encoder(data_list)
        self.embed_label = self._get_encoder(label_list)
        self.encode = self._get_encB()

        dec_list = [self.latent_dims + self.class_dims, (self.latent_dims + self.class_dims)*2, (self.latent_dims + self.class_dims)*3,\
                            (self.latent_dims + self.class_dims)*2, self.sequence_length]
        self.decoder = self._get_decoder(dec_list)
        self.device = device
        self.all_bn_layers = []
        for n, layer in self.named_modules():
            if isinstance(layer, nn.BatchNorm1d):
                self.all_bn_layers.append(layer)
        
    def forward(self, x, cond):
        # x is B x C x L and cond is B x 1 x L' --> they map into the B x C x F
        # input value is in the mode B x C x L
        batch_size = x.shape[0]
        # init_feature = self.backbone_encode(x)s
        # init_feature = init_feature.reshape(batch_size,-1) 
        init_data_feature = x.reshape(batch_size,-1) 
        init_label_feature = cond.reshape(batch_size,-1)
        init_data_feature = self.embed_data(init_data_feature)
        init_label_feature = self.embed_label(init_label_feature)

        fusing_feat = torch.cat([init_data_feature[:,None,...], init_label_feature[:,None,...]], dim=1)

        mulogl = self.encode(fusing_feat)
        mu = mulogl[:,0,...]
        logl = mulogl[:,1,...]

        sigma = torch.exp(0.5* logl) 
        seed = torch.randn_like(sigma).to(self.device)
        decode_feature = mu + sigma * seed
        
        decode_feature = torch.cat([decode_feature, cond.reshape(batch_size,-1)], dim=-1)
        recovered_data = self.decoder(decode_feature)

        return recovered_data, mu, logl


    def sample_and_decode(self, cond):
        number = cond.shape[0]
        # seed = torch.empty(number, self.latent_dims).uniform_(0, 1).to(self.device)
        # label = torch.bernoulli(seed)
        latent_feature = torch.randn(number, self.latent_dims).to(self.device) * 5
        # add the representation 
        decode_feature = torch.cat([latent_feature, cond.reshape(number,-1)], dim=-1)
        recovered_data = self.decoder(decode_feature)

        return recovered_data
    
    def _get_encB(self):
        # set the Encoder network
        layers = []
        hidden_dims = [2,4,8,4,2]
        layer_num = len(hidden_dims)
        for l in range(layer_num - 1):
            downsample = nn.Sequential(
                nn.Conv1d(hidden_dims[l], hidden_dims[l+1],
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm1d(hidden_dims[l+1], momentum=BN_MOMENTUM),
            )
            layers.append(Bottleneck(hidden_dims[l], hidden_dims[l+1], downsample=downsample))
        return nn.Sequential(*layers)

    def _get_encoder(self, dim_l, is_class=False):
        # set the mlp network # set it as the fix
        layers = []
        length = len(dim_l)
        layers.append(nn.BatchNorm1d(dim_l[0]))
        for idx in range(length - 1):
            layers.append(nn.Sequential(nn.Linear(dim_l[idx], dim_l[idx + 1]), nn.BatchNorm1d(dim_l[idx+1]), Swish()))

        return nn.Sequential(*layers)
    

    def _get_decoder(self, dim_l):
        layers = []
        length = len(dim_l)
        layers.append(nn.BatchNorm1d(dim_l[0]))
        for idx in range(length -2):
            layers.append(nn.Sequential(nn.Linear(dim_l[idx], dim_l[idx + 1]), nn.BatchNorm1d(dim_l[idx+1]), Swish()))

        layers.append(nn.Sequential(nn.Linear(dim_l[-2], dim_l[-1]), nn.Tanh()))
        return nn.Sequential(*layers)

    def batchnorm_loss(self):
        loss = 0
        for l in self.all_bn_layers:
            if l.affine:
                loss += torch.max(torch.abs(l.weight))
        return loss


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    test_data = torch.randn(10,1,20).to(device)
    cond = torch.randn(10,1,5).to(device)
    VAE_n = CVAE(1, 20, 16, 5,device)
    VAE_n = VAE_n.cuda()
    output, mu, sigma = VAE_n(test_data, cond)
        

