import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('/home/panzhiyu/project/HandPose/Prior/')
from models.distributions import Normal
from thirdparty.swish import Swish
import argparse 
# cells for mlp process

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


class Celldec(nn.Module):
    def __init__(self, dim_in, mult = 2):
        super().__init__()
        self.ctype = 'dec'
        self.bnswish1 = nn.Sequential(nn.BatchNorm1d(dim_in , momentum=0.05), Swish())
        self.lift = nn.Linear(dim_in, dim_in * mult)
        self.bnswish2 = nn.Sequential(nn.BatchNorm1d(dim_in * mult, momentum=0.05), Swish())
        self.down = nn.Linear(dim_in * mult, dim_in)
        self.skip = nn.Linear(dim_in, dim_in)
    
    def forward(self, x):
        s = self.bnswish1(x)
        s = self.lift(s)
        s = self.bnswish2(s)
        s = self.down(s)

        return s + self.skip(x)


class CellUp(nn.Module):
    def __init__(self, dim_in, dim_out, mult = 2):
        super().__init__()
        self.ctype = 'up'
        self.bnswish1 = nn.Sequential(nn.BatchNorm1d(dim_in , momentum=0.05), Swish())
        self.lift = nn.Linear(dim_in, dim_in * mult)
        self.bnswish2 = nn.Sequential(nn.BatchNorm1d(dim_in * mult, momentum=0.05), Swish())
        self.down = nn.Linear(dim_in * mult, dim_out)
        self.skip = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        s = self.bnswish1(x)
        s = self.lift(s)
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

class DecCombinerCell(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ctype = 'dec_combiner'
        self.channel_dims = 2
        self.conv1d = nn.Conv1d(self.channel_dims, self.channel_dims, kernel_size=3, padding=1, bias=True)
        self.process = nn.Linear(dim *2, dim, bias=True)
    def forward(self, x1, x2):
        # x1, x2  B, C
        B = x1.shape[0]
        comb = torch.cat([x1.unsqueeze(1), x2.unsqueeze(1)], dim=1)
        comb = comb.reshape(B, -1)
        comb = self.process(comb)
        return comb

# for sampler
class sampler(nn.Module): # create by adding,
    def __init__(self, dim):
        super().__init__()
        self.mu = nn.Linear(dim, dim, bias=True)
        self.logsigma = nn.Linear(dim, dim, bias=True)
    def forward(self, x):
        mu = self.mu(x)
        logsigma = self.logsigma(x)
        return mu, logsigma              

class NVAE_mlp(nn.Module):
    def __init__(self, args, input_length, device=torch.device('cuda')):
        super().__init__()
        self.input_length = input_length
        self.device = device
        self.latent_dims = args.latent_dims
        # give a soft excess from input length to the latent dims 
        gap = 2       
        self.dims_set = self.set_dims(input_length, self.latent_dims, gap)
        self.num_per_group = args.num_per_group
        self.preprocess = nn.ModuleList()
        for _ in range(3): # 
            self.preprocess.append(Cellenc(self.dims_set[0]))
        # create the enc tower
        self.enc_samplers = []
        self.enc_tow_cell = self.init_enc_tow(self.dims_set, args.num_per_group, self.enc_samplers)
        self.enc_samplers.reverse()
        self.enc_samplers = nn.ModuleList(self.enc_samplers)
        # add one enc0
        self.enc0 = nn.Sequential(nn.ELU(), nn.Linear(self.latent_dims, self.latent_dims), nn.ELU())
        # q for the z0 and for hidden variable
        self.h0 = nn.Parameter(torch.rand(1,self.latent_dims), requires_grad=True)
        # create the dec tower
        self.dec_samplers = nn.ModuleList()
        self.dims_set.reverse()
        self.dec_tow_cell = self.init_dec_tow(self.dims_set, args.num_per_group, self.dec_samplers)
        # post process
        self.postprocess = nn.ModuleList()
        for _ in range(3):
            self.postprocess.append(Celldec(self.dims_set[-1]))
        # final_pred
        self.final_pred = nn.Sequential(nn.ELU(), nn.Linear(self.dims_set[-1], self.dims_set[-1]))
        self.all_bn_layers = []
        for n, layer in self.named_modules():
            if isinstance(layer, nn.BatchNorm1d):
                self.all_bn_layers.append(layer)
    def init_enc_tow(self, dims_set, num_per_group, samplers):
        enc_init = nn.ModuleList()
        total_length = len(dims_set)

        # samplers.append(sampler(dims_set[0]))
        # enc_init.append(EncCombinerCell(dims_set[0], dims_set[0]))
        for i in range(total_length - 1):
            for g in range(num_per_group):
                enc_init.append(Cellenc(dims_set[i]))
            # add a downsampling net
            # if i+1 != total_length - 1:
            enc_init.append(EncCombinerCell(dims_set[i], dims_set[i])) # do not add the enc fusion sampling for the lowest layer
            enc_init.append(CellDown(dims_set[i], dims_set[i+1]))
            samplers.append(sampler(dims_set[i+1]))
            

        # end with one down sampling net
        return enc_init

    def init_dec_tow(self, dims_set, num_per_group, samplers):
        dec_init = nn.ModuleList()
        total_length = len(dims_set)
        
        for i in range(total_length -1):
            for g in range(num_per_group):
                dec_init.append(Celldec(dims_set[i]))
            # add one upsampling set
            dec_init.append(DecCombinerCell(dims_set[i]))
            dec_init.append(CellUp(dims_set[i], dims_set[i+1]))
            if i+1 != total_length -1:
                samplers.append(sampler(dims_set[i+1])) # do not set the sampler for the orig dims

        return dec_init

    def set_dims(self, input_length, latent_dims, gap=2):
        dims_set = []
        dims_set.append(input_length)
        length = (input_length - latent_dims) // gap + 1
        current = input_length
        for i in range(length):
            current -= gap
            if current < latent_dims:
                if latent_dims - current == gap:
                    break
                dims_set.append(latent_dims)
                break
            else:
                dims_set.append(current)
        return dims_set

    def forward(self, x):
        # input is B, L
        s = x / np.pi
        for cell in self.preprocess:
            s = cell(s)

        # run the main encoder tower
        combiner_cells_enc = []
        combiner_cells_s = []
        for cell in self.enc_tow_cell:
            if cell.ctype == 'enc_combiner':
                combiner_cells_enc.append(cell)
                combiner_cells_s.append(s)
            else:
                s = cell(s)

        combiner_cells_enc.reverse()
        combiner_cells_s.reverse()

        idx_dec = 0
        ftr = self.enc0(s)
        mu_q, log_sig_q = self.enc_samplers[idx_dec](ftr)
        dist = Normal(mu_q, log_sig_q)
        z, _ = dist.sample()
        log_q_conv = dist.log_p(z)
        all_q = [dist]
        all_log_q = [log_q_conv]

        s = 0
        dist = Normal(mu = torch.zeros_like(z), log_sigma=torch.zeros_like(z))
        log_p_conv = dist.log_p(z)
        all_p = [dist]
        all_log_p = [log_p_conv]

        batch_size = z.size(0)
        s = self.h0.repeat(batch_size,1)
        for cell in self.dec_tow_cell:
            if cell.ctype == 'dec_combiner':
                if idx_dec > 0:
                    mu_p, log_sig_p = self.dec_samplers[idx_dec -1](s)
                    ftr = combiner_cells_enc[idx_dec - 1](combiner_cells_s[idx_dec - 1], s)

                    mu_q, log_sig_q = self.enc_samplers[idx_dec](ftr)
                    dist = Normal(mu_p + mu_q, log_sig_p + log_sig_q)
                    z, _ = dist.sample()
                    log_q_conv = dist.log_p(z)
                    
                    all_log_q.append(log_q_conv)
                    all_q.append(dist)

                    dist = Normal(mu_p, log_sig_p)
                    log_p_conv = dist.log_p(z)
                    all_p.append(dist)
                    all_log_p.append(dist)
                
                s = cell(s,z)
                idx_dec += 1
            else:
                s = cell(s)

        for cell in self.postprocess:
            s = cell(s)
        
        final_preds = self.final_pred(s)
        final_preds = final_preds * np.pi
        # compute kl
        kl_all = []
        # log_p, log_q = 0., 0.
        for q, p, log_q_conv, log_p_conv in zip(all_q, all_p, all_log_q, all_log_p):
            kl_per_var = q.kl(p)
            kl_all.append(torch.mean(kl_per_var, dim=-1))
            # log_q += torch.sum(log_q_conv, dim=-1)
            # log_p += torch.sum(log_p_conv, dim=-1)
        
        kl_all = torch.stack(kl_all, dim=1)
        kl_v = torch.mean(kl_all)

        return final_preds, kl_all, kl_v

    def batchnorm_loss(self):
        loss = 0
        for l in self.all_bn_layers:
            if l.affine:
                loss += torch.max(torch.abs(l.weight))

        return loss 
        
    def sample(self,num_samples, t=1):
        s = self.h0.repeat(num_samples,1)
        z0_size = s.shape
        dist = Normal(mu=torch.zeros(z0_size).cuda(), log_sigma=torch.zeros(z0_size).cuda(), temp=t) # Normal distribution
        z, _ = dist.sample()

        idx_dec = 0
        for cell in self.dec_tow_cell:
            if cell.ctype == 'dec_combiner':
                if idx_dec > 0:
                    # form prior
                    mu, log_sigma = self.dec_samplers[idx_dec -1](s)
                    dist = Normal(mu, log_sigma, t)
                    z, _ = dist.sample()

                s = cell(s, z)
                idx_dec += 1
            else:
                s = cell(s)

        for cell in self.postprocess:
            s = cell(s)
        final_preds = self.final_pred(s)
        final_preds = final_preds * np.pi
        
        return final_preds

if __name__ == '__main__':
    parser = argparse.ArgumentParser('VAE')
    parser.add_argument('--num_per_group', type=int, default=2,
                        help='num groups')
    parser.add_argument('--latent_dims', type=int, default=4,
                        help='latent_dims')
    args = parser.parse_args()

    device = torch.device('cuda')
    x = torch.rand(10,20).to(device)
    test = NVAE_mlp(args, 20)
    test = test.to(device)
    fi, kl = test(x)


    

