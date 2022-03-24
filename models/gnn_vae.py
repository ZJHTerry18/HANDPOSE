import torch
import numpy as np
# from torch_geometric.nn.conv import GATConv
# from torch_geometric.nn import GCNConv
import torch.nn as nn
import sys
sys.path.append('/home/panzhiyu/project/HandPose/Prior/')
from thirdparty.swish import Swish


class GAT_VAE(nn.Module):
    def __init__(self, input_length, latent_dims, hidden_dims, headers=4,device=torch.device('cpu')):
        super().__init__()
        self.latent_dims = latent_dims
        self.device = device
        self.input_dims = input_length
        self.hidden_dims = hidden_dims
        self.headers = headers
        # three classes different embedding layers
        self.init_embeding = nn.ModuleList()
        dim_embed = [input_length, 2**2, 2**3, hidden_dims]
        for _ in range(3):
             # input dim's 
            temp_layers = self._get_encoder(dim_embed)
            self.init_embeding.append(temp_layers)
        # set the hidden feature

        # self.hidden_layers = nn.Parameter(torch.randn(1, 1, hidden_dims), requires_grad=True).to(device) # TODO: be the others
        self.hidden_representation = nn.Parameter(torch.randn(1, 1, latent_dims).to(device), requires_grad=True)

        # setting the GAT layers
        # self.gat_layerers_num = gat_num
        # set the graph
        nodes_num = 15
        self.nodes_num = nodes_num
        self.affinity_cond = torch.zeros(nodes_num, nodes_num).to(device)
        affinity_cond = torch.tensor([[1,6],[2,7],[3,8],[4,9],[5,10],[6,11],[7,12],[8,13],[9,14],[10,15],\
                        [1,2],[2,3],[3,4],[4,5]]) - 1 # cut one node
        edge_num = affinity_cond.shape[0]
        for i in range(edge_num):
            self.affinity_cond[affinity_cond[i,:][0], affinity_cond[i,:][1]] = 1
        self.affinity_cond = self.affinity_cond.T + self.affinity_cond + torch.eye(nodes_num).to(device)
        # self.degree_matrix = torch.(torch.sum(self.affinity_cond, dim=-1, keepdim=True))
        # self.affinity_cond_norm = self.affinity_cond @ torch.pinverse(self.degree_matrix)
        # # totally 16 nodes
        self.gat_paras = nn.ParameterList()
        self.weight_core = nn.ParameterList()
        gat_features_dims = [hidden_dims, hidden_dims * 2, hidden_dims * 2] # multi-head attention with 6 times within residual and one 
        gat_features_dims.append(self.latent_dims)
        gat_num = len(gat_features_dims)
        self.gat_num = gat_num
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.2)
        self.swish = Swish()
        for i in range(gat_num - 1): # It can add the residual block
            if i == 0:
                self.weight_core.append(nn.Parameter(torch.randn(headers, gat_features_dims[i], gat_features_dims[i+1]).to(device), requires_grad=True))
            else:
                self.weight_core.append(nn.Parameter(torch.randn(headers, gat_features_dims[i] * headers, gat_features_dims[i+1]).to(device), requires_grad=True))
            self.gat_paras.append(nn.Parameter(torch.randn(headers, 2 * gat_features_dims[i+1]).to(device), requires_grad=True))
        self.softmax = nn.Softmax(dim=-1) 
        # vae and decoder part
        # decode the hidden para into the contact form
        self.pcp_embed = nn.Sequential(self._get_covnet(), nn.Linear(self.latent_dims * 5, self.latent_dims, bias=False), Swish())

        self.hidden_decode = nn.Sequential(nn.Linear(self.latent_dims, self.latent_dims), Swish(),\
                            nn.Linear(self.latent_dims, 5), nn.Sigmoid()) # TODO: it may be the other forms
        # self.middle_devode = nn.Sequential(nn.Linear(self.latent_dims, 1), nn.Sigmoid()) # useful or not , this is for decoded PCP joint
        
        self.vae_PCP = nn.ModuleList()
        for _ in range(5):
            self.vae_PCP.append(self._get_covnet())
 
        # mu logl 

        self.vae_dp = nn.ModuleList()
        self.node_decode = nn.ModuleList()
        for i in range(3):
            if i == 0:
                self.node_decode.append(nn.Sequential(nn.Linear(self.latent_dims, 2, bias=False), nn.Tanh()))
            else:
                self.node_decode.append(nn.Sequential(nn.Linear(self.latent_dims, 1, bias=False), nn.Tanh()))

            if i > 0:
                self.vae_dp.append(self._get_covnet()) # only for 


    def forward(self, x):
        # x is the input dims B x 5 x 6
        batch_size = x.shape[0]
        init_PCP = x[:,:,:2]; init_PIP = x[:,:,2:4]; init_DIP = x[:,:,4:6]
        init_info = [init_PCP, init_PIP, init_DIP]
        # feature embeding
        embeding_feature = []
        for i in range(3):
            embeding_feature.append(self.init_embeding[i](init_info[i]))
        # then it get the B X 5 X hidden_feature
        # GAT
        # embeding_feature.insert(0, self.hidden_layers.repeat(batch_size, 1, 1))
        embeding_feature = torch.cat(embeding_feature, dim = -2) # B x 16 x hidden
        gat_layers_num = len(self.weight_core)
        for j in range(gat_layers_num):
            embeding_feature = torch.einsum('bnf, kfj -> bknj', embeding_feature, self.weight_core[j]) # B x headers x 16 x j
            embeding_f_copy_r = embeding_feature[:,:,None,:,:].repeat(1, 1, self.nodes_num, 1, 1)
            embeding_f_copy_l = embeding_feature[:,:,:,None,:].repeat(1, 1, 1, self.nodes_num, 1) # B x headers x 16 x 16 x j 
            embeding_f_matrix = torch.cat([embeding_f_copy_l, embeding_f_copy_r], dim=-1) # B X H X 16 X 16 X 2H
            pre_attention_coef = self.leakyrelu(torch.einsum('bhijk, hk -> bhij', embeding_f_matrix, self.gat_paras[j])) # B X H X 16 X 16
            # times the original affinity
            pre_attention_coef = pre_attention_coef * (self.affinity_cond > 0)
            # calculate the attention 
            # B x H x 16 x 16
            attention_coef = self.softmax(pre_attention_coef)
            weighted_feature = attention_coef.unsqueeze(-1) * embeding_f_copy_r # B x H x 16 x 16 x F
            if j < gat_layers_num - 1:
                embeding_feature = self.swish(torch.sum(weighted_feature, dim = -2)) # B x H x 16 X f
                split_feature = [embeding_feature[:,k,...] for k in range(self.headers)]
                embeding_feature = torch.cat(split_feature, dim = -1) # B x H x 16 x KF
            else:
                embeding_feature = self.swish(torch.mean(torch.sum(weighted_feature, dim = -2), dim=1))
        
        # PCP with H generate the whole generation
        kl_loss = 0
        fusing_feature = torch.cat([embeding_feature[:,0:5,...].unsqueeze(-2), self.hidden_representation.repeat(batch_size,5,1).unsqueeze(-2)], dim = -2) # 
        fusing_feature_c = [fusing_feature[:,k, :,:] for k in range(5)]
        fusing_feature_c = torch.cat(fusing_feature_c, dim=-1)
        # root_representation = self.pcp_embed(fusing_feature.reshape(-1,2,self.latent_dims))
        root_representation = self.pcp_embed(fusing_feature_c)
        # root_representation = root_representation.reshape(batch_size,5)
        root_mu = root_representation[...,0,:]; root_logl = root_representation[...,1,:]
        # decode the feature
        root_sigma = torch.exp(0.5* root_logl)
        kl_loss += 0.5 * torch.mean(torch.sum(root_mu ** 2 + root_sigma ** 2 - root_logl - 1, dim=-1))
        seed = torch.randn_like(root_sigma).to(self.device)
        sample_root = root_mu + seed * root_sigma
        contact_info = self.hidden_decode(sample_root) # output --------
        # VAE part modification  
        root_feature = torch.cat([sample_root.unsqueeze(-2), self.hidden_representation.repeat(batch_size,1,1)], dim = -2)
        reconstruct_features = []
        PCP_vae_c = []
        # vae for 5 PCPs
        for k in range(5):
            temp_r = self.vae_PCP[k](root_feature)
            temp_mu = temp_r[...,0,:]; temp_logl = temp_r[...,1,:]
            temp_sigma = torch.exp(0.5* temp_logl)
            seed = torch.randn_like(temp_sigma).to(self.device)
            sample_pcp = temp_mu + seed * temp_sigma  # can be the bottom up form
            kl_loss += 0.5 * torch.mean(torch.sum(temp_mu ** 2 + temp_sigma ** 2 - temp_logl - 1, dim=-1)) # TODO: weight of kl_loss
            PCP_vae_c.append(sample_pcp.unsqueeze(-2))
        PCP_vae_c = torch.cat(PCP_vae_c, dim=-2)
        reconstruct_features.append(PCP_vae_c)
        PCP_f = torch.cat([PCP_vae_c.unsqueeze(-2), sample_root.unsqueeze(-2).unsqueeze(-2).repeat(1,5,1,1)], dim = -2)
        # vae for 5 PIPs and DIPs
        VAE_F = PCP_f.clone()
        for k in range(2):
            temp_r = self.vae_dp[k](VAE_F.reshape(-1,2,self.latent_dims))
            temp_mu = temp_r[...,0,:].reshape(batch_size,5,self.latent_dims); temp_logl = temp_r[...,1,:].reshape(batch_size,5,self.latent_dims)
            temp_sigma = torch.exp(0.5* temp_logl)
            seed = torch.randn_like(temp_sigma).to(self.device)
            sample_f = temp_mu + seed * temp_sigma  # can be the bottom up form
            kl_loss += 0.5 * torch.mean(torch.sum(temp_mu ** 2 + temp_sigma ** 2 - temp_logl - 1, dim=[-1,-2])) # TODO: weight of kl_loss
            reconstruct_features.append(sample_f)#.reshape(batch_size,5,self.latent_dims)
            VAE_F = torch.cat([sample_f.unsqueeze(-2), PCP_vae_c.unsqueeze(-2)], dim = -2)
        reconstruction_angle = []
        # reconstruct the result
        for l in range(3):
            reconstruction_angle.append(self.node_decode[l](reconstruct_features[l]))
        reconstruction_angle = torch.cat(reconstruction_angle, dim=-1)

        contact_info = torch.cat([1-contact_info.unsqueeze(-2), contact_info.unsqueeze(-2)], dim=-2)
        return reconstruction_angle, contact_info, kl_loss
        

    def _get_encoder(self, dim_l):
        # set the mlp network # set it as the fix
        layers = []
        length = len(dim_l)
        # layers.append(nn.BatchNorm1d(dim_l[0]))
        for idx in range(length - 1):
            layers.append(nn.Sequential(nn.Linear(dim_l[idx], dim_l[idx + 1]), Swish())) # no batch_norm  nn.BatchNorm1d(dim_l[idx+1]), 

        return nn.Sequential(*layers)

    def _get_covnet(self):
        return nn.Sequential(nn.Conv1d(2, 2, kernel_size=1, bias=False), Swish(),\
                        nn.Conv1d(2, 2, kernel_size=3, padding=1, bias=False), Swish(),\
                        nn.Conv1d(2, 2, kernel_size=3, padding=1, bias=False), Swish())
    

if __name__ == '__main__':
    device = torch.device('cpu')
    test = GAT_VAE(2,4,8,4,device)
    x = torch.randn(10,5,6)
    re_angle, contact_info, kl_loss = test(x)
    pass
