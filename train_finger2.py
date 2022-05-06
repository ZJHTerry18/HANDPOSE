import torch
import torch.nn as nn
import numpy as np
import os
import argparse # prepare for the arg config

import torch.distributed as dist
from torch.multiprocessing import Process
import torch.nn.functional as F

# from models.gnn_vae import GAT_VAE
from model_finger.finger_net import finger_embeding_model

from data.finger_dataset import FingerPrint
import utils
from thirdparty.adamax import Adamax
from torch.cuda.amp import autocast, GradScaler

from hand_model_vis import vis
from transfer_cordinate import angle2cord
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

def get_optimizer(args, model):
    vae_paras_list = []
    if args.fast_adamax:
        cnn_optimizer = Adamax(filter(lambda p: p.requires_grad, model.parameters()), args.learning_rate,
                               weight_decay=args.weight_decay, eps=1e-3)
    else:
        cnn_optimizer = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()), args.learning_rate,
                                           weight_decay=args.weight_decay, eps=1e-3)

    return model, cnn_optimizer, vae_paras_list

def main(args):
    # ensures that weight initializations are all the same
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    logging = utils.Logger(args.global_rank, args.save)
    writer = utils.Writer(args.global_rank, args.save)

    # get the train data
    folder_images = '/Extra/panzhiyu/finger/dataset_fptype_v0.3/fingerprint_single'
    folder_motion = '/Extra/panzhiyu/finger/dataset_fptype_v0.3/leap' 
    annotation_file = '/Extra/panzhiyu/finger/dataset_fptype_v0.3/position_info_type.pkl'
    refer_folder = '/Extra/panzhiyu/finger/dataset_fptype_v0.3/fingerprint/'
    train_dataset = FingerPrint(refer_folder, folder_images,folder_motion, annotation_file, True)
    valid_dataset = FingerPrint(refer_folder,folder_images,folder_motion, annotation_file, False) # is_training is meaningless
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
    nvae_file = 'output_nvae_vae/best_model.pt' # flat or curl
    model = finger_embeding_model(args, nvae_file = nvae_file, device=device) # input length, latent, hidden, headers
    model = model.cuda()
    logging.info('param size = %fM ', utils.count_parameters_in_M(model)) # counting the size
    # load the pretrained decoder
    model, cnn_optimizer, vae_paralist = get_optimizer(args, model)
    cnn_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        cnn_optimizer, float(args.epochs - args.warmup_epochs - 1), eta_min=args.learning_rate_min)
    
    grad_scalar = GradScaler(2**10)

    # loading checkpoint or not 
    checkpoint_file = os.path.join(args.save, 'checkpoint.pt')
    best_file = os.path.join(args.save, 'best_model.pt')
    if args.cont_training:
        logging.info('loading the model.')
        checkpoint = torch.load(best_file, map_location='cpu') # TODO: use the best file
        # checkpoint = torch.load(best_file, map_location='cpu')
        init_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        model = model.cuda()
        cnn_optimizer.load_state_dict(checkpoint['optimizer'])
        grad_scalar.load_state_dict(checkpoint['grad_scalar'])
        cnn_scheduler.load_state_dict(checkpoint['scheduler'])
        global_step = checkpoint['global_step']
        # do an initial test
        logging.info('Testing !!!')
        output_re_loss = test(valid_queue, model, args, logging, writer, global_step)
    else:
        global_step, init_epoch = 0, 0

    # generate the beta paras via epochs
    VPP = 0.01
    beta_collects = frange_cycle_linear(args.epochs, stop= VPP, n_cycle=4, ratio=0.7) # set the cyclic rounds
    if args.sample:
        validate_vis(valid_queue, best_file, model)
        exit()
    else:
        min_loss = torch.tensor(np.inf).to(device)
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
            global_step = train(train_queue, vae_paralist, beta, model, cnn_optimizer, global_step, grad_scalar, warmup_iters, writer, logging, device)
            # prepare the eval
            model.eval() 
            eval_freq = 15
            if epoch % eval_freq == 0 or epoch == (args.epochs - 1):
                torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(),
                                'optimizer': cnn_optimizer.state_dict(), 'global_step': global_step,
                                'args': args, 'scheduler': cnn_scheduler.state_dict(),
                                'grad_scalar': grad_scalar.state_dict()}, checkpoint_file)
                output_re_loss = test(valid_queue, model, args, logging, writer, global_step)
                if output_re_loss < min_loss:
                    min_loss = output_re_loss
                    torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(),
                                'optimizer': cnn_optimizer.state_dict(), 'global_step': global_step,
                                'args': args, 'scheduler': cnn_scheduler.state_dict(),
                                'grad_scalar': grad_scalar.state_dict()}, best_file)
            writer.close()


def sample_vis(model, checkpoint_file, num=10, var_ = 1):
    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    # for paras in model.decoder.parameters():
    #     print(paras)
    model = model.cuda()
    model.eval()
    with torch.no_grad():
        generate_sampling, contact_info = model.sample_generation(num, var_)
    # generate_sampling = torch.permute(generate_sampling,(0,2,1))
    generate_angle = generate_sampling * np.pi
    # generate_angle = torch.atan2(generate_sampling[...,0], generate_sampling[...,1])
    return generate_angle, contact_info

# validation : using the validate images for 1. generate the posible poses and sample from the vae latent space; 2. visualization by the MANO or others


def train(train_queue, vae_paralist, beta, model, cnn_optimizer, global_step, grad_scalar, warmup_iters, writer, logging, device): # temporally not using the grad scaling
    # nelbo = utils.AvgrageMeter()
    KL_loss = utils.AvgrageMeter()
    reconstruction_loss = utils.AvgrageMeter()
    touch_loss = utils.AvgrageMeter()
    gat_loss = utils.AvgrageMeter() 
    feature_dist_loss = utils.AvgrageMeter()
    para_dist_loss = utils.AvgrageMeter()
    Total_loss = utils.AvgrageMeter()
    model.train()
    
    for step, data in enumerate(train_queue):
        images, finger_ang, centers, types = data
        # extent the dims B x 5 x 6
        orig_ang = finger_ang.clone()
        orig_ang = orig_ang.cuda()
        # zeros = torch.zeros(finger_ang.shape[0], finger_ang.shape[1], 1)
        # ang = torch.cat([finger_ang[:,:,:3], zeros, finger_ang[:,:,3:4], zeros], dim=-1)
        # cond = cond.cuda().to(ang.dtype)                                                     
        # ang = ang.cuda()
        images = images.cuda()
        centers = centers.cuda()
        types = types.cuda()
        if global_step < warmup_iters:
            lr = args.learning_rate * float(global_step) / warmup_iters
            for param_group in cnn_optimizer.param_groups:
                param_group['lr'] = lr
        cnn_optimizer.zero_grad()
        with autocast():
            # input the angle
            # input_angle = ang / np.pi # norm [-pi, pi] to [-1,1]
            recovered_data = model(images, centers, types, orig_ang)
            # calulate the KL and re loss
            loss = 0
            # recovered_data = recovered_data * np.pi # scale [-1,1] to [-pi, pi]
            # gat_recov = gat_recov * np.pi
            # l2 loss
            recovered_data = recovered_data.reshape(-1,5,4)
            rec_cor = angle2cord(recovered_data.to(torch.float), device)
            org_cor = angle2cord(orig_ang, device)
            cor_loss = torch.sum(torch.norm(rec_cor - org_cor, dim=-1, p=2)) / (15 * rec_cor.shape[0]) # do not limit the others 
            ## process the mask for touched finger
            refer_indx = torch.tensor([[0,0,0,0],[0,1,6,7],[2,8,9,10],[3,11,12,13],[4,14,15,16],[5,17,18,19]]).to(device).unsqueeze(0).repeat(orig_ang.shape[0],1,1)
            org_idx = (types + 1).repeat(1,1,refer_indx.shape[-1])
            new_idx = torch.gather(refer_indx, dim=1, index=org_idx.to(torch.int64))
            mask = torch.zeros(orig_ang.shape[0], 5, 20).to(device)
            ones_fill = torch.ones(orig_ang.shape[0], 5, 20).to(device)
            new_mask = torch.scatter(input = mask, dim=-1, index = new_idx, src=ones_fill)
            new_mask = torch.sum(new_mask, dim=1) > 0
            new_mask[:,0] = False
            total_cal_num = torch.sum(new_mask)
            cor_touched_loss = torch.sum(torch.norm(rec_cor - org_cor, dim=-1, p=2) * new_mask) / total_cal_num
            loss += (cor_touched_loss + cor_loss) # gat loss represent the prior loss

            # # para dist
            # terms_num = len(vae_paralist)
            # para_loss = 0
            # for idx,params in enumerate(model.mlp_nvae.parameters()):
            #     # params.requires_grad = False
            #     target = vae_paralist[idx]
            #     para_loss += torch.norm(params - target, p=1)
            # para_loss = para_loss / terms_num # add one regularization term
            # loss += para_loss # 

            reconstruction_loss.update(cor_loss.item())
            touch_loss.update(cor_touched_loss.item())
            # feature_dist_loss.update(feature_dist.item())
            # para_dist_loss.update(para_loss.item())
            Total_loss.update(loss.item())

        grad_scalar.scale(loss).backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        # utils.average_gradients(model.parameters(), args.distributed) # for distributed training # TODO: some params has no name and grad
        grad_scalar.step(cnn_optimizer)
        grad_scalar.update()

        # nelbo.update(loss.data, 1)

        if (global_step + 1) % 100 == 0:
            # norm
            writer.add_scalar('train/loss', loss, global_step)
            writer.add_scalar('train/re_loss', cor_loss, global_step)
            writer.add_scalar('train/touch_loss', cor_touched_loss, global_step)
            # writer.add_scalar('train/feature_loss', feature_dist, global_step)
            # writer.add_scalar('train/para_loss', para_loss, global_step)
            # writer.add_scalar('train/beta', beta, global_step)
            logging.info('train %d: the total loss is %f', global_step, Total_loss.avg)
            # logging.info('The beta is %f', beta)
            logging.info('The kl loss is %f', KL_loss.avg)
            logging.info('The re loss is %f', reconstruction_loss.avg)
            logging.info('The touch loss is %f', touch_loss.avg)
            # logging.info('The feature loss is %f', feature_dist_loss.avg)
            logging.info('The para loss is %f', para_dist_loss.avg)
        
        global_step += 1

    return global_step


def test(valid_queue, model, args, logging, writer, global_step, device=torch.device('cuda')):
    if args.distributed:
        dist.barrier()
    vali_rel = utils.AvgrageMeter()
    vali_total = utils.AvgrageMeter()
    vali_angle = utils.AvgrageMeter()
    vali_touch = utils.AvgrageMeter()
    vali_nvae = utils.AvgrageMeter()
    model.eval()
    for step, data in enumerate(valid_queue):
        images, finger_ang, centers, types = data
        orig_ang = finger_ang.clone()
        # zeros = torch.zeros(finger_ang.shape[0], finger_ang.shape[1], 1)
        # ang = torch.cat([finger_ang[:,:,:3], zeros, finger_ang[:,:,3:4], zeros], dim=-1)
        # cond = cond.cuda().to(ang.dtype)
        # ang = ang.cuda()
        orig_ang = orig_ang.cuda()
        images = images.cuda()
        centers = centers.cuda()
        types = types.cuda()
        with torch.no_grad():
            # input_angle = ang / np.pi
            recovered_data = model(images, centers, types, orig_ang)
            # calulate the KL and re loss
            loss = 0
            # recovered_data = recovered_data * np.pi
            # angle_loss = 2 * (1 - torch.cos(recovered_data - orig_ang))
            # angle_loss = torch.mean(angle_loss) # do not times the curl weight
            recovered_data = recovered_data.reshape(-1,5,4)
            rec_cor = angle2cord(recovered_data.to(torch.float), device)
            org_cor = angle2cord(orig_ang, device)
            cor_loss = torch.sum(torch.norm(rec_cor - org_cor, dim=-1, p=2)) / (15 * rec_cor.shape[0])

            refer_indx = torch.tensor([[0,0,0,0],[0,1,6,7],[2,8,9,10],[3,11,12,13],[4,14,15,16],[5,17,18,19]]).to(device).unsqueeze(0).repeat(orig_ang.shape[0],1,1)
            org_idx = (types + 1).repeat(1,1,refer_indx.shape[-1])
            new_idx = torch.gather(refer_indx, dim=1, index=org_idx.to(torch.int64))
            mask = torch.zeros(orig_ang.shape[0], 5, 20).to(device)
            ones_fill = torch.ones(orig_ang.shape[0], 5, 20).to(device)
            new_mask = torch.scatter(input = mask, dim=-1, index = new_idx, src=ones_fill)
            new_mask = torch.sum(new_mask, dim=1) > 0
            new_mask[:,0] = False
            total_cal_num = torch.sum(new_mask)
            cor_touched_loss = torch.sum(torch.norm(rec_cor - org_cor, dim=-1, p=2) * new_mask) / total_cal_num

            angle_loss = torch.mean(torch.norm(recovered_data - orig_ang, dim=(-1,-2), p=1))
            loss += cor_loss
            vali_rel.update(cor_loss.item())
            vali_total.update(loss.item())
            vali_angle.update(angle_loss.item())
            vali_touch.update(cor_touched_loss.item())

    if args.distributed:
        # block to sync
        dist.barrier()
    logging.info('val, step: %d, Total: %f, Reconstraction %f, touch %f, angle %f, nvae %f', step, vali_total.avg, vali_rel.avg, vali_touch.avg, vali_angle.avg, vali_nvae.avg)
    writer.add_scalar('train/vali_re', torch.tensor(vali_rel.avg), global_step)
    return vali_rel.avg

def validate_vis(valid_queue, load_file, model, gap = 10, device=torch.device('cuda')):
    checkpoint = torch.load(load_file, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda()
    model.eval()
    vis_gt_ang = []
    vis_pred_ang = []
    # cond_list = []
    loss_total = utils.AvgrageMeter()
    for step, data in enumerate(valid_queue):
        if step % gap == 0:
            images, finger_ang, centers, types = data
            orig_ang = finger_ang.clone()
            # zeros = torch.zeros(finger_ang.shape[0], finger_ang.shape[1], 1)
            # ang = torch.cat([finger_ang[:,:,:3], zeros, finger_ang[:,:,3:4], zeros], dim=-1)
            # cond = cond.cuda().to(ang.dtype)
            # ang = ang.cuda()
            orig_ang = orig_ang.cuda()
            images = images.cuda()
            centers = centers.cuda()
            types = types.cuda()
            with torch.no_grad():
                # input_angle = ang / np.pi
                recovered_data = model(images, centers, types, orig_ang)
                # calulate the KL and re loss
                # loss = 0
                # loss += kl_loss
                # recovered_data = recovered_data * np.pi
                # angle_loss = 2 * (1 - torch.cos(recovered_data - orig_ang))
                # angle_loss = torch.mean(angle_loss) # do not times the curl weight
                orig_ang = orig_ang.reshape(-1,5,4)
                recovered_data = recovered_data.reshape(-1,5,4)
                rec_cor = angle2cord(recovered_data.to(torch.float), device)
                org_cor = angle2cord(orig_ang, device)
                # import pdb;pdb.set_trace()
                cor_loss = torch.sum(torch.norm(rec_cor - org_cor, dim=-1, p=2)) / (15 * recovered_data.shape[0]) # angle loss
                loss_total.update(cor_loss.item())
                # angle_loss = torch.mean(torch.norm(recovered_data - orig_ang, dim=(-1,-2), p=1))
                
                vis_gt_ang.append(orig_ang)
                vis_pred_ang.append(recovered_data)
                # cond_list.append(cond)
                # loss += angle_loss
    print(f'The total loss is {loss_total.avg}')
    vis_gt_ang = torch.cat(vis_gt_ang, dim=0)
    vis_pred_ang = torch.cat(vis_pred_ang, dim=0)
    # cond_list = torch.cat(cond_list, dim=0)
    vis_gt_ang = vis_gt_ang.cpu().numpy()
    vis_pred_ang = vis_pred_ang.cpu().numpy()
    # contact_info = cond_list.cpu().numpy()
    vis(vis_gt_ang, 'vis_gt_finger_wo_nave/')
    vis(vis_pred_ang, 'vis_pred_finger_wo_nave/')

def print_para(model):
    for name, paras in model.named_parameters():
        print(name, '        ', paras)
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser('VAE')
    parser.add_argument('--save', type=str, default='output_0309_v2',
                        help='id used for storing intermediate results')
    # gat model
    parser.add_argument('--latent_dims', type=int, default=4,
                        help='the dimensions of latent scales')
    parser.add_argument('--num_per_group', type=int, default=2,
                        help='num_per_group')
    parser.add_argument('--num_headers', type=int, default=4,
                        help='headers')
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

    parser.add_argument('--cont_training', action='store_true', default=False,
                        help='This flag enables training from an existing checkpoint.')

    parser.add_argument('--sample', action='store_true', default=False,
                        help='This flag enables sampling from an existing bestmodel.')

    parser.add_argument('--data_vis', action='store_true', default=False,
                        help='This flag enables visualization from collected datasets.')

    parser.add_argument('--model', type=str, default='finger_gat_embeding_model',
                        help='choices from finger_gat_embeding_model and ...')

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
    parser.add_argument('--seed', type=int, default=42,
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

