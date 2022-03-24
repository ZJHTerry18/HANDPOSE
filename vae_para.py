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
        self.relu = Swish()  # nn.ReLU(inplace=True)
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


class PVAE(nn.Module):
    def __init__(self, input_channels, sequence_length, latent_dims, class_num, device):
        super().__init__()
        # set up the models
        self.input_channels = input_channels
        self.sequence_length = sequence_length
        # self.backbone_encode = self._get_BackBone(layer_num=3) 
        # Test No backbone version
        # self.backbone_channels = self.input_channels * 3
        # self.mlp_input = self.backbone_channels * self.sequence_length
        self.class_num = class_num
        self.mlp_input = self.sequence_length
        self.latent_dims = latent_dims
        self.mlp_b = [self.mlp_input, self.mlp_input * 3, self.mlp_input * 2, self.mlp_input * 2]
        self.mlp_c = [self.mlp_input * 2, class_num]
        self.mlp_u = [self.mlp_input * 2, latent_dims]
        self.mlp_s = [self.mlp_input * 2, latent_dims]


        self.mlp_d = [latent_dims, latent_dims * 3, latent_dims*2, self.sequence_length] # decorder part

        # self.mlp_c = [self.mlp_input, self.mlp_input * 3, self.mlp_input * 2, self.class_num]

        self.bac = self._get_backbone()
        self.get_mu = self._get_encoder(self.mlp_u)
        self.get_logl = self._get_encoder(self.mlp_s)
        self.curl_enc = self._get_encoder(self.mlp_c,True)
        self.decoder = self._get_decoder()
        self.dec_back = self._get_decB(layer_num=2) 

        self.curl_para = nn.Parameter(torch.randn(1,latent_dims), requires_grad=True).to(device)
        self.noncurl_para = nn.Parameter(torch.randn(1,latent_dims), requires_grad=True).to(device)

        self.device = device
        self.all_bn_layers = []
        for n, layer in self.named_modules():
            if isinstance(layer, nn.BatchNorm1d):
                self.all_bn_layers.append(layer)
        
    def forward(self,x):
        # input value is in the mode B x C x L
        batch_size = x.shape[0]
        # init_feature = self.backbone_encode(x)s
        # init_feature = init_feature.reshape(batch_size,-1) 
        init_feature = x.reshape(batch_size,-1) 
        init_feature = self.bac(init_feature)
        mu = self.get_mu(init_feature)
        logl = self.get_logl(init_feature) # normal label
        curl_c = self.curl_enc(init_feature)
        curl_feature = curl_c[:,0:1] * self.noncurl_para + curl_c[:,1:2] * self.curl_para # B * 3
        # sample
        sigma = torch.exp(0.5* logl) 
        seed = torch.randn_like(sigma).to(self.device)
        latent_feature = mu + sigma * seed
        # for representing the fusing feature

        latent_feature = torch.cat([latent_feature[:,None,:], curl_feature[:,None,:]], dim=1)
        dc_f = self.dec_back(latent_feature)
        # decode
        dc_f = dc_f.reshape(batch_size, -1)
        out = self.decoder(dc_f)
        recovered_data = out.reshape(batch_size, self.sequence_length)

        return recovered_data, mu, logl, curl_c

    def sample_and_decode(self, is_curl=True, number=10, seed_fix=False):
        # number of generated_samples
        if seed_fix:
            seed = torch.empty(1, self.latent_dims).uniform_(0, 1).to(self.device)
            label = torch.bernoulli(seed)
            label = label.repeat(number,1)
        else:
            seed = torch.empty(number, self.latent_dims).uniform_(0, 1).to(self.device)
            label = torch.bernoulli(seed)
        latent_feature = label + torch.randn(number, self.latent_dims).to(self.device)
        # add the representation 
        if is_curl:
            curl_feature = self.curl_para.repeat(number,1)
        else:
            curl_feature = self.noncurl_para.repeat(number,1)
        latent_feature = torch.cat([latent_feature[:,None,:], curl_feature[:,None,:]], dim=1)
        dc_f = self.dec_back(latent_feature)
        # decode
        dc_f = dc_f.reshape(number, -1)
        out = self.decoder(dc_f)
        recovered_data = out.reshape(number, self.sequence_length)
        return recovered_data, label

    # def _get_BackBone(self,layer_num = 3):
    #     # set the Encoder network
    #     layers = []
    #     for l in range(layer_num - 1):
    #         layers.append(Bottleneck(self.input_channels, self.input_channels))
    #     downsample = nn.Sequential(
    #             nn.Conv1d(self.input_channels, self.input_channels * 3,
    #                       kernel_size=1, stride=1, bias=False),
    #             nn.BatchNorm1d(self.input_channels * 3, momentum=BN_MOMENTUM),
    #         )
    #     layers.append(Bottleneck(self.input_channels, self.input_channels * 3,downsample=downsample))
    #     return nn.Sequential(*layers)

    def _get_decB(self,layer_num = 3):
        # set the Encoder network
        layers = []
        for l in range(layer_num - 1):
            layers.append(Bottleneck(2, 2))
        downsample = nn.Sequential(
                nn.Conv1d(2, 1,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm1d(1, momentum=BN_MOMENTUM),
            )
        layers.append(Bottleneck(2, 1,downsample=downsample))
        return nn.Sequential(*layers)
    
    def _get_backbone(self):
        layers = []
        length = len(self.mlp_b)
        layers.append(nn.BatchNorm1d(self.mlp_b[0]))
        for idx in range(length - 1):
            layers.append(nn.Sequential(nn.Linear(self.mlp_b[idx], self.mlp_b[idx + 1]), nn.BatchNorm1d(self.mlp_b[idx+1]), Swish()))
        # layers.append(nn.BatchNorm1d(self.mlp_s[-1]))

        return nn.Sequential(*layers)


    # def _get_encoder(self, is_class = False):
    #     # set the mlp network # set it as the fix
    #     layers = []
    #     length = len(self.mlp_s)
    #     layers.append(nn.BatchNorm1d(self.mlp_s[0]))
    #     for idx in range(length - 1):
    #         layers.append(nn.Sequential(nn.Linear(self.mlp_s[idx], self.mlp_s[idx + 1]), nn.BatchNorm1d(self.mlp_s[idx+1]), Swish()))
        
    #     layers.append(nn.BatchNorm1d(self.mlp_s[-1]))
    #     if is_class:
    #         layers.append(nn.Sequential(nn.Linear(self.mlp_s[-1], self.class_num), nn.Softmax(dim=-1))) # softmax activation
    #     else:
    #         layers.append(nn.Sequential(nn.Linear(self.mlp_s[-1], self.mlp_s[-1]), Swish()))

    #     return nn.Sequential(*layers)

    def _get_encoder(self, dim_l, is_class=False):
        # set the mlp network # set it as the fix
        layers = []
        length = len(dim_l)
        layers.append(nn.BatchNorm1d(dim_l[0]))
        for idx in range(length - 2):
            layers.append(nn.Sequential(nn.Linear(dim_l[idx], dim_l[idx + 1]), nn.BatchNorm1d(dim_l[idx+1]), Swish()))
        
        layers.append(nn.BatchNorm1d(dim_l[-2]))
        if is_class:
            layers.append(nn.Sequential(nn.Linear(dim_l[-2], dim_l[-1]), nn.Softmax(dim=-1))) # softmax activation
        else:
            layers.append(nn.Sequential(nn.Linear(dim_l[-2], dim_l[-1]), Swish()))

        return nn.Sequential(*layers)
    

    def _get_decoder(self):
        layers = []
        length = len(self.mlp_d)
        layers.append(nn.BatchNorm1d(self.mlp_d[0]))
        for idx in range(length -2):
            layers.append(nn.Sequential(nn.Linear(self.mlp_d[idx], self.mlp_d[idx + 1]), nn.BatchNorm1d(self.mlp_d[idx+1]), Swish()))

        layers.append(nn.Sequential(nn.Linear(self.mlp_d[-2], self.mlp_d[-1]), nn.Tanh()))

        return nn.Sequential(*layers)

    def batchnorm_loss(self):
        loss = 0
        for l in self.all_bn_layers:
            if l.affine:
                loss += torch.max(torch.abs(l.weight))
        return loss


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    test_data = torch.randn(10,2,20).to(device)

    VAE_n = P_VAE(2, 20, 16, device)
    VAE_n = VAE_n.cuda()
    output, mu, sigma = VAE_n(test_data)
        

