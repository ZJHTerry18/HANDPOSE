# resnet_50_gat_finger_net.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import argparse
sys.path.append('/home/panzhiyu/project/HandPose/Prior/')
from models.distributions import Normal
from models.mlp_nvae import NVAE_mlp # read in the NVAE
from thirdparty.swish import Swish
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class Cellenc(nn.Module):
    def __init__(self, dim_in, mult = 2):
        super().__init__()
        self.ctype = 'enc'
        self.bn1 = nn.BatchNorm1d(dim_in, momentum=0.05)
        self.lift = nn.Linear(dim_in, dim_in * mult)
        self.bnswish1 = nn.Sequential(nn.BatchNorm1d(dim_in * mult, momentum=0.05), Swish())
        self.enc = nn.Linear(dim_in * mult, dim_in * mult)
        self.bnswish2 = nn.Sequential(nn.BatchNorm1d(dim_in * mult, momentum=0.05), Swish())
        self.down = nn.Linear(dim_in * mult, dim_in)

        self.skip = nn.Linear(dim_in, dim_in)
    def forward(self, x):
        s = self.bn1(x)
        s = self.lift(s)
        s = self.bnswish1(s)
        s = self.enc(s)
        s = self.bnswish2(s)
        s = self.down(s)
        return s + self.skip(x)
  
class CellDown(nn.Module):
    def __init__(self, dim_in, dim_out, mult = 2):
        super().__init__()
        self.ctype = 'down'
        self.bn1 = nn.BatchNorm1d(dim_in, momentum=0.05)
        self.lift = nn.Linear(dim_in, dim_in * mult)
        self.bnswish1 = nn.Sequential(nn.BatchNorm1d(dim_in * mult, momentum=0.05), Swish())
        self.enc = nn.Linear(dim_in * mult, dim_in * mult)
        self.bnswish2 = nn.Sequential(nn.BatchNorm1d(dim_in * mult, momentum=0.05), Swish())
        self.down = nn.Linear(dim_in * mult, dim_out)

        self.skip = nn.Linear(dim_in, dim_out)
    def forward(self, x):

        s = self.bn1(x)
        s = self.lift(s)
        s = self.bnswish1(s)
        s = self.enc(s)
        s = self.bnswish2(s)
        s = self.down(s)
        return s + self.skip(x)

class EncCombinerCell(nn.Module):
    def __init__(self, x1_dim, x2_dim):
        super().__init__()
        self.ctype = 'enc_combiner'
        self.process = nn.Linear(x2_dim, x2_dim, bias=True)
    def forward(self, x1, x2):
        x2 = self.process(x2)
        out = x1 + x2
        return out

# sampler
class sampler(nn.Module): # create by adding,
    def __init__(self, dim):
        super().__init__()
        self.mu = nn.Linear(dim, dim, bias=True)
        self.logsigma = nn.Linear(dim, dim, bias=True)
    def forward(self, x):
        mu = self.mu(x)
        logsigma = self.logsigma(x)
        return mu, logsigma   

BN_MOMENTUM = 0.05

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

class finger_nvae_embeding_model(nn.Module):
    def __init__(self, args, resnet_file = None, nvae_file = None, init_feature_dims = 128, device = torch.device('cpu')):
        super().__init__()
        self.device = device
        self.init_feature_dims = init_feature_dims
        self.args = args
        self.resnet = resnet50(init_feature_dims)
        self.input_length = 20
        self.mlp_nvae = NVAE_mlp(args, self.input_length) # input length = 2, latent dims = 8, hidden_dims = 8 ,num of heads = 4, device
        if resnet_file is not None:
            self.resnet.load_state_dict(torch.load(resnet_file)) # load pretrained resnet50, only contains the model state_dict
        if nvae_file is not None:
            self.mlp_nvae.load_state_dict(torch.load(nvae_file, map_location='cpu')['state_dict']) # load pretrained gat_vae, only contains the model state_dict
        # layers for positional embedding, not embedding, direct add into the feature
        # pos_dim_list = [2, 8, 16, init_feature_dims]
        # self.pos_embedding = self.make_pos_embedding(pos_dim_list)
        self.feature_fusing = nn.Sequential(nn.Linear(init_feature_dims + 2, init_feature_dims, bias=True), Swish())
        # gat embedding initialization
        ## set the placeholder for the input of gat embedding
        # self.dropout = nn.Dropout(0.2)
        self.finger_repre = nn.Parameter(torch.randn(1,5,init_feature_dims).to(device), requires_grad=True) # 5 finger t, i, m, r, p
        encoder_layer = TransformerEncoderLayer(d_model=init_feature_dims, nhead=args.num_headers, batch_first=True) # batch, seq, feature_dims
        self.feature_encoder = TransformerEncoder(encoder_layer, num_layers=2) # TODO: just set 2 layers
        ## set the hireachical forms (psi- residual block) TODO
        # preprocess the feature_encoder compress to the 20 latent dims
        self.preprocess = self.init_process(init_feature_dims * 5,self.input_length, args.num_per_group)
        # add one cellenc to process
        self.enc = self.process(self.input_length, args.num_per_group)

        # self.dim_sets = self.mlp_nvae.set_dims(self.input_length, args.latent_dims, 2)
        # self.encf_samplers = []
        # self.enc_cells_tow = self.mlp_nvae.init_enc_tow(self.dim_sets, args.num_per_group, self.encf_samplers)
        # self.encf_samplers.reverse()
        # self.encf_samplers = nn.ModuleList(self.encf_samplers)
        # self.enc0 = nn.Sequential(nn.ELU(), nn.Linear(args.latent_dims, args.latent_dims), nn.ELU())
        
    def init_process(self, input_dims, output_dims, num_groups):
        preprocess_term = nn.ModuleList()
        for _ in range(num_groups):
            preprocess_term.append(Cellenc(input_dims))
        preprocess_term.append(CellDown(input_dims, output_dims))
        return preprocess_term
    
    def process(self, input_dims, num_groups):
        process_term = nn.ModuleList()
        for _ in range(num_groups):
            process_term.append(Cellenc(input_dims))
        return process_term

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
        cross_feature_c = torch.cat(cross_feature_c, dim=-1) # B, 5*F
        # using the cross_feature_c as the init part
        so = cross_feature_c.clone()
        for cell in self.preprocess:
            so = cell(so)

        for cell in self.enc:
            so = cell(so)

        # process the orig motion data
        motions = motions.reshape(-1,20)
        m_rec, _, kl_v_r, all_rq, _, mapping_feature = self.mlp_nvae(motions)

        # feature dist
        feature_loss = torch.mean(torch.norm((so - mapping_feature), p = 2, dim=-1))
        
        # combiner_cells_enc = []
        # combiner_cells_s = []
        # for cell in self.enc_cells_tow:
        #     if cell.ctype == 'enc_combiner':
        #         combiner_cells_enc.append(cell)
        #         combiner_cells_s.append(so)
        #     else:
        #         so = cell(so)
        
        # combiner_cells_enc.reverse()
        # combiner_cells_s.reverse()
        # idx_dec = 0
        # ftr = self.enc0(so)
        # mu_q, log_sig_q = self.encf_samplers[idx_dec](ftr)
        # dist = Normal(mu_q, log_sig_q)
        # z, _ = dist.sample()
        # log_q_conv = dist.log_p(z)
        # all_q = [dist]
        # all_log_q = [log_q_conv]

        # so = 0
        # dist = Normal(mu = torch.zeros_like(z), log_sigma=torch.zeros_like(z))
        # log_p_conv = dist.log_p(z)
        # all_p = [dist]
        # all_log_p = [log_p_conv]

        # batch_size = z.size(0)
        # so = self.mlp_nvae.h0.repeat(batch_size,1)
        # for cell in self.mlp_nvae.dec_tow_cell:
        #     if cell.ctype == 'dec_combiner':
        #         if idx_dec > 0:
        #             mu_p, log_sig_p = self.mlp_nvae.dec_samplers[idx_dec -1](so)
        #             ftr = combiner_cells_enc[idx_dec - 1](combiner_cells_s[idx_dec - 1], so)
        #             mu_q, log_sig_q = self.encf_samplers[idx_dec](ftr)
        #             dist = Normal(mu_p + mu_q, log_sig_p + log_sig_q)
        #             z, _ = dist.sample()
        #             log_q_conv = dist.log_p(z)
                    
        #             all_log_q.append(log_q_conv)
        #             all_q.append(dist)

        #             dist = Normal(mu_p, log_sig_p)
        #             log_p_conv = dist.log_p(z)
        #             all_p.append(dist)
        #             all_log_p.append(dist)
                
        #         so = cell(so,z)
        #         idx_dec += 1
        #     else:
        #         so = cell(so)

        for cell in self.mlp_nvae.postprocess:
            so = cell(so)
        
        final_preds = self.mlp_nvae.final_pred(so)
        final_preds = final_preds * np.pi

        # compute the loss between the finger and nvae of q
        # kl_all_vf = []
        # for fq, nq in zip(all_q, all_rq):
        #     kl_per_var = fq.kl(nq) # basically copy the routine of AE part
        #     kl_all_vf.append(torch.mean(kl_per_var, dim=-1))

        # kl_all_vf = torch.stack(kl_all_vf, dim=1)
        # kl_v_f = torch.mean(kl_all_vf)

        return final_preds, m_rec, feature_loss


if __name__ == '__main__': 
    parser = argparse.ArgumentParser('VAE')
    parser.add_argument('--latent_dims', type=int, default=4,
                        help='latent_dims')
    parser.add_argument('--num_per_group', type=int, default=2,
                        help='num_per_group')
    parser.add_argument('--num_headers', type=int, default=4,
                        help='headers')
    args = parser.parse_args()
    gat_file = 'output_nvae_vae/best_model.pt'
    device = torch.device('cuda')
    net = finger_nvae_embeding_model(args, nvae_file = gat_file)
    net = net.cuda()
    images = torch.rand(25,5,1,256,256).to(device)
    centers = torch.rand(25,5,2).to(device)
    types = torch.rand(25,5,1).to(device)
    motions = torch.rand(25,5,4).to(device)
    re, m_rec, kl_loss = net(images, centers, types, motions)
