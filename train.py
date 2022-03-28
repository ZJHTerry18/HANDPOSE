import torch
import torch.nn as nn
import numpy as np
import os
import argparse # prepare for the arg config

import torch.distributed as dist
from torch.multiprocessing import Process

from vae_prior import NAIVE_VAE
from backbone_n_vae_prior import BVAE
from vae_cond import CVAE

from data.prior_dataset import ivg_HD_naive
import utils
from thirdparty.adamax import Adamax
from torch.cuda.amp import autocast, GradScaler

from hand_model_vis import vis
import warnings
warnings.filterwarnings("ignore")

def init_processes(rank, size, fn, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = args.port#'6020' # set the port 
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size)
    fn(args)
    cleanup()

def cleanup():
    dist.destroy_process_group()

def frange_cycle_linear(n_iter, start=0.0, stop=1.0,  n_cycle=1, ratio=0.7): # iter as epochs? 
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L 

def main(args):
    # ensures that weight initializations are all the same
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    logging = utils.Logger(args.global_rank, args.save)
    writer = utils.Writer(args.global_rank, args.save)

    # get the train data
    # train_dataset = ivg_HD_naive('/Extra/panzhiyu/IVG_HAND/train/curl', True)
    train_dataset = ivg_HD_naive('/Extra/panzhiyu/IVG_HAND/train_pkl', True)
    if args.data_vis:
        train_dataset.vis_dataset()
        exit() 

    # valid_dataset = ivg_HD_naive('/Extra/panzhiyu/IVG_HAND/test/curl', False) # is_training is meaningless
    valid_dataset = ivg_HD_naive('/Extra/panzhiyu/IVG_HAND/test_pkl', False) # is_training is meaningless
    train_sampler, valid_sampler = None, None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
    
    train_queue = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler, pin_memory=False, num_workers=8, drop_last=True)
    
    valid_queue = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler, pin_memory=False, num_workers=1, drop_last=False)

    # get the train_queue and valid queue
    args.num_total_iter = len(train_queue) * args.epochs
    warmup_iters = len(train_queue) * args.warmup_epochs
    swa_start = len(train_queue) * (args.epochs - 1)

    # set the model paras
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # model = NAIVE_VAE(1,20,5,device) # TODO: using the fixed number, latent related with the cond vector
    model = eval(args.model)(1,20,args.latent_dims,device) # 5 is the class dims
    model = model.cuda()
    logging.info('param size = %fM ', utils.count_parameters_in_M(model)) # counting the size

    if args.fast_adamax:
        cnn_optimizer = Adamax(model.parameters(), args.learning_rate,
                               weight_decay=args.weight_decay, eps=1e-3)
    else:
        cnn_optimizer = torch.optim.Adamax(model.parameters(), args.learning_rate,
                                           weight_decay=args.weight_decay, eps=1e-3)
    
    cnn_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        cnn_optimizer, float(args.epochs - args.warmup_epochs - 1), eta_min=args.learning_rate_min)
    
    grad_scalar = GradScaler(2**10)

    # loading checkpoint or not 
    checkpoint_file = os.path.join(args.save, 'checkpoint.pt')
    best_file = os.path.join(args.save, 'best_model.pt')
    if args.cont_training:
        logging.info('loading the model.')
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        init_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        model = model.cuda()
        cnn_optimizer.load_state_dict(checkpoint['optimizer'])
        grad_scalar.load_state_dict(checkpoint['grad_scalar'])
        cnn_scheduler.load_state_dict(checkpoint['scheduler'])
        global_step = checkpoint['global_step']
    else:
        global_step, init_epoch = 0, 0

    # generate the beta paras via epochs
    VPP = args.batch_size / len(train_dataset)
    beta_collects = frange_cycle_linear(args.epochs, stop= VPP, n_cycle=1, ratio=0.7) # set the cyclic rounds
    if args.sample:
        number_s = 50
        seed = torch.empty(number_s, 5).uniform_(0, 1).to(device)
        label = torch.bernoulli(seed)
        get_samples = sample_vis(model,best_file,label,number_s,False) # best or checkpoint
        get_samples = get_samples.reshape(number_s, 5,4)
        get_samples = get_samples.cpu().numpy()
        get_labels = label.cpu().numpy()
        # import pdb;pdb.set_trace()
        vis(get_samples, get_labels,'vis_curl_50/')

    else:
        for epoch in range(init_epoch, args.epochs): # start training
            # update lrs.
            if args.distributed:
                train_queue.sampler.set_epoch(global_step + args.seed) # random _ sampler
                valid_queue.sampler.set_epoch(0)

            if epoch > args.warmup_epochs:
                cnn_scheduler.step() # 
            
            # Logging.
            logging.info('epoch %d', epoch)
            beta = beta_collects[epoch]
            # set the constant beta
            # beta = 1
            global_step = train(train_queue, model, beta, cnn_optimizer, global_step, grad_scalar, warmup_iters, writer, logging)
            # prepare the eval
            model.eval()
            eval_freq = 15
            min_loss = torch.tensor(np.inf).to(device)
            if epoch % eval_freq == 0 or epoch == (args.epochs - 1):
                torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(),
                                'optimizer': cnn_optimizer.state_dict(), 'global_step': global_step,
                                'args': args, 'scheduler': cnn_scheduler.state_dict(),
                                'grad_scalar': grad_scalar.state_dict()}, checkpoint_file)
                output_re_loss = test(valid_queue, model, args, logging)
                if output_re_loss < min_loss:
                    torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(),
                                'optimizer': cnn_optimizer.state_dict(), 'global_step': global_step,
                                'args': args, 'scheduler': cnn_scheduler.state_dict(),
                                'grad_scalar': grad_scalar.state_dict()}, best_file)
            writer.close()


def sample_vis(model, checkpoint_file, cond, num=10, fix_cond=False):
    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    # for paras in model.decoder.parameters():
    #     print(paras)
    model = model.cuda()
    model.eval()
    with torch.no_grad():
        generate_sampling = model.sample_and_decode(num)
    # generate_sampling = torch.permute(generate_sampling,(0,2,1))
    generate_angle = generate_sampling * np.pi
    # generate_angle = torch.atan2(generate_sampling[...,0], generate_sampling[...,1])
    return generate_angle


def train(train_queue, model, beta, cnn_optimizer, global_step, grad_scalar, warmup_iters, writer, logging): # temporally not using the grad scaling
    # nelbo = utils.AvgrageMeter()
    KL_loss = utils.AvgrageMeter()
    reconstruction_loss = utils.AvgrageMeter()
    Total_loss = utils.AvgrageMeter()
    model.train()
    
    for step, data in enumerate(train_queue):
        cond, curl, x, ang = data # TODO: cond can be used as the vae condition
        # import pdb;pdb.set_trace()
        x = x.cuda()
        cond = cond.cuda().to(x.dtype)
        curl = curl.cuda()
        # TODO: add one dimension to the cond
        # cond = torch.cat([cond, curl], dim=-1)
        curl_weight = curl * 6.7 + (curl == 0)
        # change bit length
        # x = utils.pre_process(x, args.num_x_bits)
        # warm-up lr
        if global_step < warmup_iters:
            lr = args.learning_rate * float(global_step) / warmup_iters
            for param_group in cnn_optimizer.param_groups:
                param_group['lr'] = lr
        # cnn_optimizer.zero_grad()
        with autocast():
            # x_input = torch.permute(x, (0,2,1)) # B,C,L
            org_angle = torch.atan2(x[...,0], x[...,1])
            # input the angle
            # org_angle = org_angle.reshape(-1,20)# 20 is the seq length
            org_angle = org_angle.reshape(-1,1,20)
            # norm [-pi, pi] to [-1,1]
            input_angle = org_angle / np.pi
            # recovered_data, latent_mu, latent_sigma = model(x_input)
            recovered_data, latent_mu, latent_logv = model(input_angle)
            # calulate the KL and re loss
            loss = 0
            # mu_2 = torch.pow((latent_mu - cond),2) # TODO ： one form
            mu_2 = torch.pow(latent_mu,2)
            sigma_2 = torch.exp(latent_logv)          
            kl_loss = 0.5 * torch.mean(torch.sum(mu_2 + sigma_2 - latent_logv - 1, dim=-1))
            # recovered_data = torch.permute(recovered_data,(0,2,1))
            # recovered_angle = torch.atan2(recovered_data[...,0], recovered_data[...,1])
            recovered_data = recovered_data * np.pi # scale [-1,1] to [-pi, pi]
            # angle_loss = 2 * (1 - torch.cos(recovered_data - org_angle)) # decoder needs to output probs, 
            # # add loss weight for the curl samples
            # # angle_loss = angle_loss * curl_weight # does not matter for only curls
            # angle_loss = torch.mean(angle_loss)
            # import pdb;pdb.set_trace()
            angle_loss = torch.mean(torch.norm(recovered_data - org_angle.squeeze(), dim=-1, p=1)) # naive L1 loss
            if args.model == 'BVAE' or args.model == 'CVAE' or args.model == 'PVAE':
                loss += model.batchnorm_loss() * 0.1
            loss = loss + beta * kl_loss + angle_loss # add a batchnorm loss
            KL_loss.update(kl_loss.item())
            reconstruction_loss.update(angle_loss.item())
            Total_loss.update(loss.item())

        grad_scalar.scale(loss).backward()
        # utils.average_gradients(model.parameters(), args.distributed) # for distributed training
        grad_scalar.step(cnn_optimizer)
        grad_scalar.update()
        # cnn_optimizer.zero_grad()
        # loss.backward()
        # # torch.nn.utils.clip_grad_norm(model.parameters(),5)
        # cnn_optimizer.step()
        # nelbo.update(loss.data, 1)

        if (global_step + 1) % 100 == 0:
            # norm
            writer.add_scalar('train/loss', loss, global_step)
            writer.add_scalar('train/re_loss', angle_loss, global_step)
            writer.add_scalar('train/kl_loss', kl_loss, global_step)
            writer.add_scalar('train/beta', beta, global_step)
            logging.info('train %d: the total loss is %f', global_step, Total_loss.avg)
            logging.info('The beta is %f', beta)
            logging.info('The kl loss is %f', KL_loss.avg)
            logging.info('The re loss is %f', reconstruction_loss.avg)
        
        global_step += 1

    return global_step


def test(valid_queue, model, args, logging):
    if args.distributed:
        dist.barrier()
    vali_rel = utils.AvgrageMeter()
    vali_total = utils.AvgrageMeter()
    model.eval()
    for step, data in enumerate(valid_queue):
        cond, curl, x, ang = data
        cond = cond.cuda().to(x.dtype)
        curl = curl.cuda()
        # cond = torch.cat([cond, curl],dim=-1)
        curl_weight = curl * 6.7 + (curl == 0)
        x = x.cuda()
        with torch.no_grad():
            # x_input = torch.permute(x, (0,2,1)) # B,C,L
            org_angle = torch.atan2(x[...,0], x[...,1])
            org_angle = org_angle.reshape(-1,1,20) # Add the backbone
            input_angle = org_angle / np.pi
            recovered_data, latent_mu, latent_logv= model(input_angle)
            # calulate the KL and re loss
            # mu_2 = torch.pow(latent_mu - cond,2)     
            mu_2 = torch.pow(latent_mu,2)
            sigma_2 = torch.exp(latent_logv)       
            kl_loss = 0.5 * torch.mean(torch.sum(mu_2 + sigma_2 - latent_logv - 1, dim=-1))
            # recovered_data = torch.permute(recovered_data,(0,2,1))
            # recovered_angle = torch.atan2(recovered_data[...,0], recovered_data[...,1])
            recovered_data = recovered_data * np.pi
            # angle_loss = 2 * (1 - torch.cos(recovered_data - org_angle))
            # angle_loss = torch.mean(angle_loss) # do not times the curl weight
            # angle_loss = torch.mean(torch.norm(recovered_data - org_angle, dim=(-1,-2), p=1)) # naive L1 loss
            angle_loss = torch.mean(torch.norm(recovered_data - org_angle.squeeze(), dim=-1, p=1))
            loss = kl_loss + angle_loss
            vali_rel.update(angle_loss.item())
            vali_total.update(loss.item())
            

    # utils.average_tensor(nelbo_avg.avg, args.distributed)
    # utils.average_tensor(neg_log_p_avg.avg, args.distributed)
    if args.distributed:
        # block to sync
        dist.barrier()
    logging.info('val, step: %d, Total: %f, Reconstraction %f', step, vali_total.avg, vali_rel.avg)
    return vali_rel.avg

def print_para(model):
    for name, paras in model.named_parameters():
        print(name, '        ', paras)
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser('VAE')
    parser.add_argument('--save', type=str, default='output_0309_v2',
                        help='id used for storing intermediate results')
    # parser.add_argument('--dataset', type=str, default='mnist',
    #                     choices=['cifar10', 'mnist', 'omniglot', 'celeba_64', 'celeba_256',
    #                              'imagenet_32', 'ffhq', 'lsun_bedroom_128', 'stacked_mnist',
    #                              'lsun_church_128', 'lsun_church_64'],
    #                     help='which dataset to use') # TODO: for multi-datasets
     # optimization
    parser.add_argument('--batch_size', type=int, default=100,
                        help='batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=1e-5,
                        help='min learning rate')
    parser.add_argument('--weight_decay', type=float, default=3e-4,
                        help='weight decay')
    parser.add_argument('--weight_decay_norm', type=float, default=0.,
                        help='The lambda parameter for spectral regularization.')
    parser.add_argument('--weight_decay_norm_init', type=float, default=10.,
                        help='The initial lambda parameter')
    parser.add_argument('--weight_decay_norm_anneal', action='store_true', default=False,
                        help='This flag enables annealing the lambda coefficient from '
                             '--weight_decay_norm_init to --weight_decay_norm.')
    parser.add_argument('--epochs', type=int, default=500,
                        help='num of training epochs')
    parser.add_argument('--warmup_epochs', type=int, default=3,
                        help='num of training epochs in which lr is warmed up')
    parser.add_argument('--fast_adamax', action='store_true', default=False,
                        help='This flag enables using our optimized adamax.')

    parser.add_argument('--num_latent_scales', type=int, default=1,
                        help='the number of latent scales')

    parser.add_argument('--latent_dims', type=int, default=5,
                        help='the dimensions of latent scales')

    parser.add_argument('--cont_training', action='store_true', default=False,
                        help='This flag enables training from an existing checkpoint.')

    parser.add_argument('--sample', action='store_true', default=False,
                        help='This flag enables sampling from an existing bestmodel.')

    parser.add_argument('--data_vis', action='store_true', default=False,
                        help='This flag enables visualization from collected datasets.')

    parser.add_argument('--model', type=str, default='NAIVE_VAE',
                        help='choices from NAIVE_VAE and BVAE, PVAE, CVAE')

    # DDP.
    parser.add_argument('--num_proc_node', type=int, default=1,
                        help='The number of nodes in multi node env.')
    parser.add_argument('--node_rank', type=int, default=0,
                        help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process in the node')
    parser.add_argument('--global_rank', type=int, default=0,
                        help='rank of process among all the processes')
    parser.add_argument('--num_process_per_node', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='address for master')
    parser.add_argument('--port', type=str, default='10000',
                        help='address for master')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed used for initialization')
    args = parser.parse_args()
    utils.create_exp_dir(args.save) # create folder

    size = args.num_process_per_node # equal the GPU number
    if size > 1:
        args.distributed = True
        processes = []
        for rank in range(size):
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_process_per_node
            global_size = args.num_proc_node * args.num_process_per_node
            args.global_rank = global_rank
            print('Node rank %d, local proc %d, global proc %d' % (args.node_rank, rank, global_rank))
            p = Process(target=init_processes, args=(global_rank, global_size, main, args)) # changed into ours
            p.start()
            processes.append(p)
    else:
        # for debugging
        print('starting in debug mode')
        args.distributed = True
        init_processes(0, size, main, args) 

