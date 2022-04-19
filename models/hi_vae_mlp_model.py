import torch
import numpy as np
import torch.nn as nn
import collections
import torch.nn.functional as F
import sys
sys.path.append('/home/panzhiyu/project/HandPose/Prior/')
from thirdparty.swish import Swish
from transfer_cordinate import angle2cord

# A encoder for q(z|x), decoder for p(x|z);
## for the prior: P(z|k) p(k) and q(k|z), and q(k|z) is the sampling IW method
class hi_vae(nn.Module): # Test 2 hierachical priors
    def __init__(self, sequence_length, latent_dims, MC_samples, IW_samples, recons_err, tao=3, slope=1, beta=1e-1, alpha = 0.7, device = torch.device('cuda')):
        super().__init__()
        # set up the models hyper_parameters
        self.sequence_length = sequence_length
        self.latent_dims = latent_dims
        self.IW_samples = IW_samples
        self.MC_samples = MC_samples
        self.device = device
        # set up the model layers
        self.outerloop = nn.ModuleList()
        self.innerloop = nn.ModuleList()
        ## q(z|x)
        self.mlp_enc = [self.sequence_length, 16, 16, self.latent_dims]
        self.mlp_dec = [self.latent_dims, 16, 16, self.sequence_length]
        self.enc_mu = self._get_encoder(self.mlp_enc)
        self.enc_logl = self._get_encoder(self.mlp_enc)
        self.outerloop.append(self.enc_mu)
        self.outerloop.append(self.enc_logl)
        ## p(x|z)
        self.dec_ = self._get_decoder(self.mlp_dec)
        self.outerloop.append(self.dec_)
        # set up the prior layer
        ## q(k|z)
        self.mlp_vi = [self.latent_dims, 16, 16, self.latent_dims]
        self.vi_mu = self._get_encoder(self.mlp_vi)
        self.vi_logl = self._get_encoder(self.mlp_vi)
        self.innerloop.append(self.vi_mu)
        self.innerloop.append(self.vi_logl)
        ## p(z|k)
        self.mlp_infer = [self.latent_dims, 16, 16, self.latent_dims]
        self.infer_mu = self._get_encoder(self.mlp_infer)
        self.innerloop.append(self.infer_mu)
        self.infer_logl = self._get_encoder(self.mlp_infer)
        self.innerloop.append(self.infer_logl)
        # loss set
        self.ka = recons_err
        self.tao = tao
        self.slope = slope
        self.beta = torch.tensor(beta).to(device)
        self.low_bound = torch.tensor(beta).to(device) / 100
        self.alpha = alpha
        self.C = -1
        self.initialphase = True # debug
        # collection
        self.eps = 1e-8


    def forward(self, x, gt):
        # x is B x 20
        B = x.shape[0]
        z_mu = self.enc_mu(x)
        z_logl = self.enc_logl(x)
        z_sigma2 = torch.exp(z_logl)
        z_sigma = torch.exp(0.5 * z_logl)
        # reconstrunction
        # seed_samples = torch.randn_like(z_sigma).to(self.device)
        # dec_f = z_mu + seed_samples * z_sigma
        # recons = self.dec_(dec_f)
        # recons = recons.reshape(B, 5, 4) * np.pi
        # rec_cor = angle2cord(recons.to(torch.float), self.device)
        # org_cor = angle2cord(gt, self.device)
        # cor_loss = torch.sum(torch.norm(rec_cor - org_cor, dim=-1, p=2)) / (15 * B)
        # if self.C < 0:
        #     self.C = cor_loss
        # if self.C > self.ka:
        #     loss = self.C
        #     return loss
        # joint optimization phase
        ## MCMC
        seed_mcmc = torch.randn(B, self.MC_samples, self.latent_dims).to(self.device)
        seed_z = z_mu.unsqueeze(1).repeat(1, self.MC_samples, 1) + z_sigma.unsqueeze(1).repeat(1, self.MC_samples, 1) * seed_mcmc
        # get the reconstruction err
        recons_mc = self.dec_(seed_z.reshape(-1, self.latent_dims))
        recons_mc = recons_mc.reshape(B, self.MC_samples ,5, 4) * np.pi
        gt_un = gt.unsqueeze(1).repeat(1,self.MC_samples, 1,1)
        rec_cor = angle2cord(recons_mc.to(torch.float).reshape(-1,5,4), self.device)
        org_cor = angle2cord(gt_un.reshape(-1,5,4), self.device)
        cor_loss = torch.sum(torch.norm(rec_cor - org_cor, dim=-1, p=2)) / (15 * B * self.MC_samples)
        
        if self.C < 0:
            self.C = cor_loss.detach().clone()
            return cor_loss
        else:
            old_C = self.C.detach().clone()
            self.C = (1 - self.alpha) * cor_loss.detach().clone() + self.alpha * self.C
        
        if self.C < self.ka:
            self.initialphase = False

        if self.initialphase: ## for testing
            loss = (1/self.beta) * ((1 - self.alpha) * cor_loss + self.alpha * old_C - self.ka) # add a weight for better learning
            return loss
        else:
            self.beta = self.beta * torch.exp(self.slope * self._coef(self.beta, self.C - self.ka, self.tao) * (self.C - self.ka))
            # constrain the value of beta
            self.beta = torch.clamp(self.beta, self.low_bound)
            loss = (1/self.beta) * ((1 - self.alpha) * cor_loss + self.alpha * old_C - self.ka)
            
        log_prob = torch.log(self._get_prob_normal(z_mu, z_sigma2, seed_z) + self.eps) # seed_z is B_2, F
        F1 = torch.mean(log_prob, dim=-1) # E_q(log(q))
        # get the epis
        epis_mu = self.vi_mu(seed_z.reshape(-1, self.latent_dims))
        epis_logl = self.vi_logl(seed_z.reshape(-1, self.latent_dims))
        epis_sigma2 = torch.exp(epis_logl)
        epis_sigma = torch.exp(0.5 * epis_logl)
        B_2 = epis_sigma2.shape[0]
        # IW k
        seed_iwk = torch.randn(B_2, self.IW_samples, self.latent_dims).to(self.device)
        seed_epis = epis_mu.unsqueeze(1).repeat(1, self.IW_samples, 1) + epis_sigma.unsqueeze(1).repeat(1, self.IW_samples, 1) * seed_iwk
        prob_q = self._get_prob_normal(epis_mu, epis_sigma2, seed_epis) # B_2, IW

        normal_zeros = torch.zeros(B_2, self.latent_dims).to(self.device)
        normal_ones = torch.ones(B_2, self.latent_dims).to(self.device)
        prob_prior = self._get_prob_normal(normal_zeros, normal_ones, seed_epis) # B_2, IW

        # get the infer z distribution
        infer_zmu = self.infer_mu(seed_epis.reshape(-1, self.latent_dims)) # B_3, F
        infer_zlogl= self.infer_logl(seed_epis.reshape(-1, self.latent_dims))
        infer_sigma2 = torch.exp(infer_zlogl)
        B_3 = infer_zmu.shape[0]
        eq_z = seed_z.reshape(-1, self.latent_dims).unsqueeze(-1).repeat(1, self.IW_samples, 1).reshape(-1, self.latent_dims)
        prob_infer = self._get_prob_normal2(infer_zmu, infer_sigma2, eq_z)
        prob_infer = prob_infer.reshape(B_2, self.IW_samples)
        F2 = torch.log(torch.mean(prob_prior * prob_infer / prob_q, dim=-1) + self.eps) # avoid the 
        F2 = F2.reshape(B, -1)
        F2 = torch.mean(F2, dim=-1)
        F = F1 - F2
        psi_loss = torch.mean(F)
        loss += psi_loss
        if torch.isinf(loss):
            import pdb;pdb.set_trace()
        return loss

    def validate(self, x, gt):
        B = x.shape[0]
        z_mu = self.enc_mu(x)
        z_logl = self.enc_logl(x)
        z_sigma2 = torch.exp(z_logl)
        z_sigma = torch.exp(0.5 * z_logl)
        # reconstrunction
        seed_samples = torch.randn_like(z_sigma).to(self.device)
        dec_f = z_mu + seed_samples * z_sigma
        recons = self.dec_(dec_f)
        recons = recons.reshape(B, 5, 4) * np.pi
        rec_cor = angle2cord(recons.to(torch.float), self.device)
        org_cor = angle2cord(gt, self.device)
        cor_loss = torch.sum(torch.norm(rec_cor - org_cor, dim=-1, p=2)) / (15 * B)
        return cor_loss
    
    def _get_prob_normal(self, mu, sigma, samples):
        # mu is B, latent_dims
        # sigma is B, latent_dims
        # samples is B x N x latent_dims
        # return is B x N x 1
        pi = torch.tensor(np.pi).to(self.device)
        latent_dims = samples.shape[-1]
        samples_num = samples.shape[1]
        mu = mu.unsqueeze(1).repeat(1, samples_num, 1)
        sigma = sigma.unsqueeze(1).repeat(1, samples_num, 1)
        prob = torch.pow(2 * pi, -latent_dims/2) * torch.pow(torch.prod(sigma, dim=-1), -1/2) * \
            torch.exp(-0.5 * torch.sum((samples - mu) * (1/sigma) * (samples - mu), dim=-1))

        return prob

    def _get_prob_normal2(self, mu, sigma, samples):
        # mu is B, latent_dims
        # sigma is B, latent_dims
        # samples is B x latent_dims
        # return is B x 1
        pi = torch.tensor(np.pi).to(self.device)
        latent_dims = samples.shape[-1]
        prob = torch.pow(2 * pi, -latent_dims/2) * torch.pow(torch.prod(sigma, dim=-1), -1/2) * \
            torch.exp(-0.5 * torch.sum((samples - mu) * (1/sigma) * (samples - mu), dim=-1))

        return prob

    def _coef(self, beta, delta, tao):
        if delta <=0:
            return torch.tanh(tao * (beta-1))
        else:
            return -1


    def _get_encoder(self, mlp):
        # set the mlp network # set it as the fix
        layers = []
        length = len(mlp)
        for idx in range(length - 1):
            layers.append(nn.Sequential(nn.Linear(mlp[idx], mlp[idx + 1]), Swish())) #nn.LeakyReLU()

        return nn.Sequential(*layers)

    def _get_decoder(self, mlp):
        layers = []
        length = len(mlp)
        for idx in range(length -2):
            layers.append(nn.Sequential(nn.Linear(mlp[idx], mlp[idx + 1]), Swish()))

        layers.append(nn.Sequential(nn.Linear(mlp[-2], mlp[-1]), nn.Tanh()))

        return nn.Sequential(*layers)

    def sample_and_decode(self, num):
        epis = torch.randn(num, self.latent_dims).to(self.device) 
        z_mu = self.infer_mu(epis)
        z_logl = self.infer_logl(epis)
        z_sigma = torch.exp(0.5 * z_logl)
        # import pdb;pdb.set_trace()
        feature = z_mu + torch.randn_like(z_sigma).to(self.device) * z_sigma
        rec = self.dec_(feature)
        rec_angle = rec.reshape(num, 5, 4) * np.pi
        return rec_angle


if __name__ == '__main__':
    device = torch.device('cuda')
    x = torch.rand(10,20).to(device)
    gt = torch.rand(10,5,4).to(device)
    test = hi_vae(20,8,50,50,1)
    test = test.to(device)
    loss = test(x, gt)
    import pdb;pdb.set_trace()