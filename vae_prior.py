import torch
import numpy as np
import torch.nn as nn
import collections

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


class NAIVE_VAE(nn.Module):
    def __init__(self, input_channels, sequence_length, latent_dims,device):
        super().__init__()
        # set up the models
        self.input_channels = input_channels
        self.sequence_length = sequence_length
        # self.backbone_encode = self._get_BackBone(layer_num=3) 
        # Test No backbone version
        # self.backbone_channels = self.input_channels * 3
        # self.mlp_input = self.backbone_channels * self.sequence_length
        self.mlp_input = self.sequence_length
        self.latent_dims = latent_dims
        self.mlp_s = [self.mlp_input, self.mlp_input * 3, self.mlp_input * 2, latent_dims]
        self.mlp_d = [latent_dims, latent_dims * 3, latent_dims*2, self.sequence_length] # decorder part
        self.get_mu = self._get_encoder()
        self.get_logl = self._get_encoder()
        self.decoder = self._get_decoder()

        self.device = device
        
    def forward(self,x):
        # input value is in the mode B x C x L
        batch_size = x.shape[0]
        # init_feature = self.backbone_encode(x)s
        # init_feature = init_feature.reshape(batch_size,-1) 
        init_feature = x.reshape(batch_size,-1) 
        mu = self.get_mu(init_feature)
        logl = self.get_logl(init_feature)
        # sample
        sigma = torch.exp(0.5* logl) 
        seed = torch.randn_like(sigma).to(self.device)
        latent_feature = mu + sigma * seed
        # decode
        out = self.decoder(latent_feature)
        recovered_data = out.reshape(batch_size, self.sequence_length)

        return recovered_data, mu, logl # this is logl

    def sample_and_decode(self, number=10, seed_fix=False):
        # number of generated_samples
        if seed_fix:
            seed = torch.empty(1, self.latent_dims).uniform_(0, 1).to(self.device)
            label = torch.bernoulli(seed)
            label = label.repeat(number,1)
        else:
            seed = torch.empty(number, self.latent_dims).uniform_(0, 1).to(self.device)
            label = torch.bernoulli(seed)
        latent_feature = label + torch.randn(number, self.latent_dims).to(self.device) * 5
        out = self.decoder(latent_feature)
        generated_data = out.reshape(number, self.sequence_length)
        return generated_data, label

    def _get_BackBone(self,layer_num = 3):
        # set the Encoder network
        layers = []
        for l in range(layer_num - 1):
            layers.append(Bottleneck(self.input_channels, self.input_channels))
        downsample = nn.Sequential(
                nn.Conv1d(self.input_channels, self.input_channels * 3,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm1d(self.input_channels * 3, momentum=BN_MOMENTUM),
            )
        layers.append(Bottleneck(self.input_channels, self.input_channels * 3,downsample=downsample))
        return nn.Sequential(*layers)
    
    def _get_encoder(self):
        # set the mlp network # set it as the fix
        layers = []
        length = len(self.mlp_s)
        for idx in range(length - 1):
            layers.append(nn.Sequential(nn.Linear(self.mlp_s[idx], self.mlp_s[idx + 1]), nn.LeakyReLU()))
        

        # layers.append(nn.BatchNorm1d(self.mlp_s[-1]))
        # layers.append(nn.Linear(self.mlp_s[-1], 1))

        return nn.Sequential(*layers)
    

    def _get_decoder(self):
        layers = []
        length = len(self.mlp_d)
        for idx in range(length -2):
            layers.append(nn.Sequential(nn.Linear(self.mlp_d[idx], self.mlp_d[idx + 1]), nn.LeakyReLU()))

        layers.append(nn.Sequential(nn.Linear(self.mlp_d[-2], self.mlp_d[-1]), nn.Tanh()))

        return nn.Sequential(*layers)


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    test_data = torch.randn(10,2,20).to(device)

    VAE_n = NAIVE_VAE(2, 20, 16, device)
    VAE_n = VAE_n.cuda()
    output, mu, sigma = VAE_n(test_data)
        

