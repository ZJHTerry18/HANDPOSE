import torch
import numpy as np
import torch.nn as nn
import collections
import torch.nn.functional as F

from thirdparty.swish import Swish
# create a mini resnet1d
BN_MOMENTUM = 0.1

# class Conv2D(nn.Conv2d):
#     """Allows for weights as input."""

#     def __init__(self, C_in, C_out, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, data_init=False,
#                  weight_norm=True):
#         """
#         Args:
#             use_shared (bool): Use weights for this layer or not?
#         """
#         super(Conv2D, self).__init__(C_in, C_out, kernel_size, stride, padding, dilation, groups, bias)

#         self.log_weight_norm = None
#         if weight_norm:
#             init = norm(self.weight, dim=[1, 2, 3]).view(-1, 1, 1, 1)
#             self.log_weight_norm = nn.Parameter(torch.log(init + 1e-2), requires_grad=True)

#         self.data_init = data_init
#         self.init_done = False
#         self.weight_normalized = self.normalize_weight()

#     def forward(self, x):
#         """
#         Args:
#             x (torch.Tensor): of size (B, C_in, H, W).
#             params (ConvParam): containing `weight` and `bias` (optional) of conv operation.
#         """
#         # do data based initialization
#         if self.data_init and not self.init_done:
#             with torch.no_grad():
#                 weight = self.weight / (norm(self.weight, dim=[1, 2, 3]).view(-1, 1, 1, 1) + 1e-5)
#                 bias = None
#                 out = F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)
#                 mn = torch.mean(out, dim=[0, 2, 3])
#                 st = 5 * torch.std(out, dim=[0, 2, 3])

#                 # get mn and st from other GPUs
#                 average_tensor(mn, is_distributed=True)
#                 average_tensor(st, is_distributed=True)

#                 if self.bias is not None:
#                     self.bias.data = - mn / (st + 1e-5)
#                 self.log_weight_norm.data = -torch.log((st.view(-1, 1, 1, 1) + 1e-5))
#                 self.init_done = True

#         self.weight_normalized = self.normalize_weight()

#         bias = self.bias
#         return F.conv2d(x, self.weight_normalized, bias, self.stride,
#                         self.padding, self.dilation, self.groups)

#     def normalize_weight(self):
#         """ applies weight normalization """
#         if self.log_weight_norm is not None:
#             weight = normalize_weight_jit(self.log_weight_norm, self.weight)
#         else:
#             weight = self.weight

#         return weight


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
        self.relu = Swish() #nn.ReLU(inplace=True)
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


class BVAE(nn.Module):
    def __init__(self, input_channels, sequence_length, latent_dims, device):
        super().__init__()
        # set up the models
        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.backbone_encode = self._get_BackBone(layer_num=3) 
        # Test No backbone version
        self.backbone_channels = self.input_channels * 3
        self.mlp_input = self.backbone_channels * self.sequence_length

        # self.mlp_input = self.sequence_length

        self.latent_dims = latent_dims
        self.mlp_s = [self.mlp_input, self.mlp_input * 3, self.mlp_input * 2, latent_dims]
        self.mlp_d = [latent_dims, latent_dims * 3, latent_dims * 2, self.sequence_length] # decorder part
        self.get_mu = self._get_encoder()
        self.get_logl = self._get_encoder()
        self.decoder = self._get_decoder()
        self.device = device

        # calculate the spectral and bn loss
        self.all_bn_layers = []
        # self.all_log_norm = []
        self.all_conv_layers = []
        for n, layer in self.named_modules():
            if isinstance(layer, nn.Conv1d): #or isinstance(layer, nn.Linear): # TODO the linear layer?
                # self.all_log_norm.append(layer.log_weight_norm)
                self.all_conv_layers.append(layer)
            if isinstance(layer, nn.BatchNorm1d):
                self.all_bn_layers.append(layer)
        
        # left/right singular vectors used for SR
        self.sr_u = {}
        self.sr_v = {}
        self.num_power_iter = 4

        
    def forward(self,x):
        # input value is in the mode B x C x L
        batch_size = x.shape[0]
        init_feature = self.backbone_encode(x)
        init_feature = init_feature.reshape(batch_size,-1) 
        # init_feature = x.reshape(batch_size,-1) 
        # import pdb;pdb.set_trace()
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
        latent_feature = label + torch.randn(number, self.latent_dims).to(self.device)
        out = self.decoder(latent_feature)
        generated_data = out.reshape(number, self.sequence_length)
        return generated_data, label

    def _get_BackBone(self,layer_num = 3):
        # set the Encoder network
        layers = []
        downsample = nn.Sequential(
                nn.Conv1d(self.input_channels, self.input_channels * 3,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm1d(self.input_channels * 3, momentum=BN_MOMENTUM),
            )
        layers.append(Bottleneck(self.input_channels, self.input_channels * 3, downsample = downsample))
        for _ in range(layer_num - 1):
            layers.append(Bottleneck(self.input_channels * 3, self.input_channels * 3))
        
        # layers.append(Bottleneck(self.input_channels, self.input_channels * 3,downsample=downsample))
        return nn.Sequential(*layers)
    
    def _get_encoder(self):
        # set the mlp network # set it as the fix
        layers = []
        length = len(self.mlp_s)
        for idx in range(length - 1):
            layers.append(nn.Sequential(nn.Linear(self.mlp_s[idx], self.mlp_s[idx + 1]), Swish())) #nn.LeakyReLU()
        
        # layers.append(nn.BatchNorm1d(self.mlp_s[-1])) # Using the BN
        # layers.append(nn.Linear(self.mlp_s[-1], self.mlp_s[-1]))

        return nn.Sequential(*layers)
    

    def _get_decoder(self):
        layers = []
        length = len(self.mlp_d)
        for idx in range(length -2):
            layers.append(nn.Sequential(nn.Linear(self.mlp_d[idx], self.mlp_d[idx + 1]), Swish()))

        layers.append(nn.Sequential(nn.Linear(self.mlp_d[-2], self.mlp_d[-1]), nn.Tanh()))

        return nn.Sequential(*layers)

    # def spectral_norm_parallel(self):
    #     """ This method computes spectral normalization for all conv layers in parallel. This method should be called
    #      after calling the forward method of all the conv layers in each iteration. """

    #     weights = {}   # a dictionary indexed by the shape of weights
    #     for l in self.all_conv_layers:
    #         weight = l.weight_normalized
    #         weight_mat = weight.view(weight.size(0), -1)
    #         if weight_mat.shape not in weights:
    #             weights[weight_mat.shape] = []

    #         weights[weight_mat.shape].append(weight_mat)

    #     loss = 0
    #     for i in weights:
    #         weights[i] = torch.stack(weights[i], dim=0)
    #         with torch.no_grad():
    #             num_iter = self.num_power_iter
    #             if i not in self.sr_u:
    #                 num_w, row, col = weights[i].shape
    #                 self.sr_u[i] = F.normalize(torch.ones(num_w, row).normal_(0, 1).cuda(), dim=1, eps=1e-3)
    #                 self.sr_v[i] = F.normalize(torch.ones(num_w, col).normal_(0, 1).cuda(), dim=1, eps=1e-3)
    #                 # increase the number of iterations for the first time
    #                 num_iter = 10 * self.num_power_iter

    #             for j in range(num_iter):
    #                 # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
    #                 # are the first left and right singular vectors.
    #                 # This power iteration produces approximations of `u` and `v`.
    #                 self.sr_v[i] = F.normalize(torch.matmul(self.sr_u[i].unsqueeze(1), weights[i]).squeeze(1),
    #                                            dim=1, eps=1e-3)  # bx1xr * bxrxc --> bx1xc --> bxc
    #                 self.sr_u[i] = F.normalize(torch.matmul(weights[i], self.sr_v[i].unsqueeze(2)).squeeze(2),
    #                                            dim=1, eps=1e-3)  # bxrxc * bxcx1 --> bxrx1  --> bxr

    #         sigma = torch.matmul(self.sr_u[i].unsqueeze(1), torch.matmul(weights[i], self.sr_v[i].unsqueeze(2)))
    #         loss += torch.sum(sigma)
    #     return loss

    def batchnorm_loss(self):
        loss = 0
        for l in self.all_bn_layers:
            if l.affine:
                loss += torch.max(torch.abs(l.weight))
        return loss




if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    test_data = torch.randn(10,1,20).to(device)
    VAE_n = NVAE(1, 20, 5, device)
    VAE_n = VAE_n.cuda()
    output, mu, sigma = VAE_n(test_data)
    bn_loss = VAE_n.batchnorm_loss()

        

