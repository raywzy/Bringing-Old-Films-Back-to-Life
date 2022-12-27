import sys
import os
import cv2
import time
import math
import glob
import shutil
import importlib
import datetime
import numpy as np
from PIL import Image
from math import log10
from collections import OrderedDict
from functools import partial

sys.path.append(os.path.dirname(sys.path[0]))

import torch

from VP_code.data.dataset import Film_dataset_1, Film_dataset_2, Film_dataset_3, Film_dataset_4
from VP_code.utils.util import worker_set_seed, get_root_logger, set_device, seed_worker
from VP_code.utils.data_util import tensor2img
from VP_code.metrics.psnr_ssim import calculate_psnr,calculate_ssim
from VP_code.models.loss import AdversarialLoss, VGGLoss_torch
from VP_code.models.discriminator import Discriminator

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import functional as F
from torch.nn import DataParallel
import torchvision.utils as vutils

def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target)**2 + eps).mean()

def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='mean')

def _get_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm 

class Trainer():
    def __init__(self, config, opts, this_logger, debug=False):

        self.config = config
        self.opts = opts
        self.epoch = self.opts.epoch
        self.iteration = 0
        self.logger = this_logger
        self.old_lr = config['trainer']['lr']
        self.old_gan_lr = config['trainer']['gan_lr']

        #### Define dataset and dataloader first
        if config['datasets']['train']['name']=='REDS':
            self.train_dataset=Film_dataset_1(config['datasets']['train'])
        if config['datasets']['train']['name']=='Vimeo':
            self.train_dataset=Film_dataset_2(config['datasets']['train'])
        if config['datasets']['train']['name']=='Colorization':
            self.train_dataset=Film_dataset_3(config['datasets']['train'])
        if config['datasets']['train']['name']=='REDS_128':
            self.train_dataset=Film_dataset_4(config['datasets']['train'])

        self.val_dataset=Film_dataset_1(config['datasets']['val'])           


        #worker_init_fn = partial(worker_set_seed, base=config['seed'], num_workers=config['datasets']['train']['num_worker_per_gpu'], rank=self.opts.global_rank)

        self.train_sampler = DistributedSampler(self.train_dataset, num_replicas=opts.world_size, rank=opts.global_rank)
        # self.val_sampler = DistributedSampler(self.val_dataset, num_replicas=opts.world_size, rank=opts.global_rank)
        ## Total # of batch and worker will be multiplied by GPU number
        self.train_loader = DataLoader(self.train_dataset, batch_size=config['datasets']['train']['batch_size_per_gpu'],
                            shuffle=(self.train_sampler is None), num_workers=config['datasets']['train']['num_worker_per_gpu'],
                            pin_memory=True, sampler=self.train_sampler, worker_init_fn=seed_worker)
        self.val_loader = DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False, sampler=None)

        num_iter_per_epoch = len(self.train_dataset) / (config['datasets']['train']['batch_size_per_gpu'] * self.opts.world_size)
        self.total_iters = self.opts.epoch * num_iter_per_epoch

        # print(config['datasets']['train']['batch_size_per_gpu'])
        self.logger.info(
            'Training statistics:'
            f'\n\tNumber of train images: {len(self.train_dataset)}'
            f"\n\tBatch size per gpu: {config['datasets']['train']['batch_size_per_gpu']}"
            f'\n\tWorld size (gpu number): {self.opts.world_size}'
            f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
            f'\n\tTotal epochs: {self.opts.epoch}; iters: {self.total_iters}.')
        self.logger.info(
            '\nValidation statistics:'
            f"\n\tNumber of val images/folders in {config['datasets']['val']['name']}: "f'{len(self.val_dataset)}')
        ####

        #### TODO: Set loss functions: GAN Loss [√], Perception Loss [√]
        self.l1_loss = torch.nn.L1Loss()
        self.adversarial_loss = AdversarialLoss(type=self.opts.which_gan).cuda()
        self.perceptual_loss = VGGLoss_torch()
        ####

        ####
        net = importlib.import_module('VP_code.models.' + opts.model_name)
        self.netG = set_device(net.Video_Backbone(spynet_path=opts.flow_path))
        self.print_network(self.netG, self.opts.model_name)
        #### TODO: use Discriminator [√]
        d_net = importlib.import_module('VP_code.models.' + opts.discriminator_name)
        self.netD = set_device(d_net.Discriminator(in_channels=3, use_sigmoid=self.opts.which_gan != 'hinge'))
        self.print_network(self.netD, "Discriminator")

        self.setup_optimizers()
        
        if config['distributed']:
            ### TODO: modify to local rank [√]
            # self.netG = DDP(self.netG, device_ids=[opts.global_rank], output_device=opts.global_rank, broadcast_buffers=True) 
            self.netG = DDP(self.netG, device_ids=[opts.local_rank],find_unused_parameters=True)           
            self.netD = DDP(self.netD, device_ids=[opts.local_rank])

    def print_network(self, net, model_name):

        if isinstance(net, (DataParallel, DDP)):
            net = net.module
        
        net_str = str(net)
        net_params = sum(map(lambda x: x.numel(), net.parameters()))

        self.logger.info(
            f'Network: {model_name}, with parameters: {net_params:,d}')
        self.logger.info(net_str)


    def update_learning_rate(self): ## TODO: separatly modify the learning rate (flow and generator)
        lrd = self.config['trainer']['lr'] / self.config['trainer']['nepoch_decay']
        gan_lrd = self.config['trainer']['gan_lr'] / self.config['trainer']['nepoch_decay']

        lr = self.old_lr - lrd
        gan_lr = self.old_gan_lr - gan_lrd        
        # for param_group in self.optimizer_D.param_groups:
        #     param_group['lr'] = lr
        flow_update = False
        for param_group in self.optimizer_G.param_groups:
            if param_group['lr'] == self.old_lr*self.config['trainer']['flow_lr_mul']:
                param_group['lr'] = lr * self.config['trainer']['flow_lr_mul']
                flow_update = True
            else:
                param_group['lr'] = lr

        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = gan_lr
        
        self.logger.info('update G learning rate: %f -> %f' % (self.old_lr, lr))
        self.logger.info('update D learning rate: %f -> %f' % (self.old_gan_lr, gan_lr))
        if flow_update:
            self.logger.info('flow learning rate is updated')

        self.old_lr = lr
        self.old_gan_lr = gan_lr

    def optimize_parameters(self, lq, gt, current_iter):


        if self.opts.fix_iter>0:
            if current_iter == 1:
                for k, v in self.netG.named_parameters():
                    if 'spynet' in k:
                        v.requires_grad = False
            elif current_iter == self.opts.fix_iter + 1:
                for v in self.netG.parameters():
                    v.requires_grad = True

        Discriminator_Loss = 0
        Generator_Loss = 0
        loss_dict = OrderedDict()

        ## Feed Forward
        predicted = self.netG(lq)

        ## Calculate the D Loss and update D
        real_vid_feat = self.netD(gt)
        fake_vid_feat = self.netD(predicted.detach())
        dis_real_loss = self.adversarial_loss(real_vid_feat, True, True)
        dis_fake_loss = self.adversarial_loss(fake_vid_feat, False, True)
        Discriminator_Loss += (dis_real_loss + dis_fake_loss) / 2 * self.config['trainer']['D_adv_loss_weight']
        loss_dict['loss_adv_D'] = Discriminator_Loss 
        self.optimizer_D.zero_grad()
        Discriminator_Loss.backward()

        # D_norm_gradient = _get_grad_norm(self.netD)
        # No_Update_Flag = False
        # self.logger.info("The L2 Norm (Befor Clip) of Discriminator gradient is %.6f"%(D_norm_gradient))
        # self.logger.info("-------------------------------------------------")
        # if D_norm_gradient >= 10.0:
        #     self.logger.info("The L2 Norm (Befor Clip) of Discriminator gradient of %d-iteration is %.6f"%(current_iter,D_norm_gradient))
        #     self.logger.info("-----------------------Don't update--------------------------")
        #     # No_Update_Flag = True
        # else:
        # torch.nn.utils.clip_grad_norm_(self.netD.parameters(), 1.0)
        self.optimizer_D.step()
 
        ## Calculate the adversarial loss of Generator
        gen_vid_feat = self.netD(predicted)
        gan_loss = self.adversarial_loss(gen_vid_feat, True, False) * self.config['trainer']['G_adv_loss_weight']
        Generator_Loss += gan_loss
        loss_dict['loss_adv_G'] = gan_loss

        ## Calculate pixel-wise loss
        loss_pix = self.l1_loss(predicted,gt) * self.config['trainer']['pix_loss_weight']
        loss_dict['loss_pix'] = loss_pix
        Generator_Loss += loss_pix

        ## Calculate perceptual loss
        _,_,c,h,w=predicted.shape
        loss_perceptual = self.perceptual_loss(predicted.view(-1,c,h,w),gt.view(-1,c,h,w)) * self.config['trainer']['perceptual_loss_weight']
        loss_dict['loss_perceptual'] = loss_perceptual
        Generator_Loss += loss_perceptual

        self.optimizer_G.zero_grad()
        Generator_Loss.backward()

        # G_norm_gradient = _get_grad_norm(self.netG)
        # self.logger.info("The L2 Norm (Befor Clip) of Generator gradient is %.6f"%(G_norm_gradient))
        # self.logger.info("-------------------------------------------------")
        # if G_norm_gradient >= 10.0:
        #     self.logger.info("The L2 Norm (Befor Clip) of Generator gradient of %d-iteration is %.6f"%(current_iter,G_norm_gradient))
        #     self.logger.info("-----------------------Don't update--------------------------")
        # else:
            # torch.nn.utils.clip_grad_norm_(self.netG.parameters(), 1.0)
        self.optimizer_G.step()

        if self.opts.global_rank==0 and current_iter % self.config['train_visualization_iter'] == 0:  # Save the training samples

            saved_results = torch.cat((gt[0], lq[0], predicted[0]),0)
            if self.config['datasets']['train']['normalizing']:
                saved_results=(saved_results + 1.)/2.
            vutils.save_image(saved_results.data.cpu(),os.path.join(self.config['path']['experiments_root'],'training_show_%s.png'%(current_iter)),nrow=self.config['datasets']['train']['num_frame'], padding=0, normalize=False)


        self.log_dict = self.reduce_loss_dict(loss_dict) ## Gather the loss from other GPUs

    def setup_optimizers(self):

        if self.config['trainer']['flow_lr_mul']==0.0: ## Don't use flow
            optim_params = self.netG.parameters()
        else:
            normal_params = []
            spynet_params = []
            for name, param in self.netG.named_parameters():
                if 'spynet' in name:
                    spynet_params.append(param)
                else:
                    normal_params.append(param)
            optim_params = [
                {  # add normal params first
                    'params': normal_params,
                    'lr': self.config['trainer']['lr']
                },
                {
                    'params': spynet_params,
                    'lr': self.config['trainer']['lr'] * self.config['trainer']['flow_lr_mul']
                },
            ]

        self.optimizer_G = torch.optim.Adam(optim_params, lr=self.config['trainer']['lr'], betas=(self.config['trainer']['beta1'], self.config['trainer']['beta2']))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=self.config['trainer']['gan_lr'], betas=(self.config['trainer']['beta1'], self.config['trainer']['beta2']))

    
    def reduce_loss_dict(self, loss_dict):
        """reduce loss dict.
        In distributed training, it averages the losses among different GPUs .
        Args:
            loss_dict (OrderedDict): Loss dict.
        """
        with torch.no_grad():
            if self.config['distributed']:
                keys = []
                losses = []
                for name, value in loss_dict.items():
                    keys.append(name)
                    losses.append(value)
                losses = torch.stack(losses, 0)
                torch.distributed.reduce(losses, dst=0)
                if self.opts.global_rank == 0:
                    losses /= self.opts.world_size
                loss_dict = {key: loss for key, loss in zip(keys, losses)}

            log_dict = OrderedDict()
            for name, value in loss_dict.items():
                log_dict[name] = value.mean().item()

            return log_dict

    def print_iter_message(self, log_vars):

        message = (f"[{self.opts.name[:5]}..][epoch:{log_vars['epoch']:3d}, "f"iter:{log_vars['iter']:8,d}, lr:(")
        for v in log_vars['lrs']:
            message += f'{v:.3e},'
        message += ')] '
        ### Timer
        total_time = time.time() - log_vars['start_time']
        time_sec_avg = total_time / log_vars['iter']  ## TODO: if resume training --> total_time / (log_vars['iter'] - start_iter +1)
        eta_sec = time_sec_avg * (self.total_iters - log_vars['iter'] - 1)
        eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
        message += f'[eta: {eta_str}, '
        message += f'time (data): {log_vars["iter_time"]:.3f} ({log_vars["data_time"]:.3f})] '
        ### Loss
        for k, v in log_vars.items():
            if k.startswith('loss_'):
                message += f'{k}: {v:.4e} '

        self.logger.info(message)

            # tensorboard logger
            # if self.use_tb_logger and 'debug' not in self.exp_name:
            #     if k.startswith('l_'):
            #         self.tb_logger.add_scalar(f'losses/{k}', v, current_iter)
            #     else:
            #         self.tb_logger.add_scalar(k, v, current_iter)


    def save_model(self, epoch, it):

        net_G_path = os.path.join(self.config['path']['models'], 'net_G_{}.pth'.format(str(it).zfill(5)))
        net_D_path = os.path.join(self.config['path']['models'], 'net_D_{}.pth'.format(str(it).zfill(5)))
        # dis_path = os.path.join(self.config['path']['models'], 'net_D_{}.pth'.format(str(it).zfill(5)))
        optimizer_path = os.path.join(self.config['path']['models'], 'optimizer_{}.pth'.format(str(it).zfill(5)))

        if isinstance(self.netG, torch.nn.DataParallel) or isinstance(self.netG, DDP):
            netG = self.netG.module
            netD = self.netD.module 
        else:
            netG = self.netG
            netD = self.netD
        
        torch.save({'netG': netG.state_dict()}, net_G_path)
        torch.save({'netD': netD.state_dict()}, net_D_path)
        # torch.save({'netD': netD.state_dict()}, dis_path)
        torch.save({'epoch': epoch, 
                    'iteration': it,
                    'optimG': self.optimizer_G.state_dict(),
                    'optimD': self.optimizer_D.state_dict()}, optimizer_path) ## TODO: Save the schedulers
        

    def validation(self):

        if "metrics" in self.config['val']:
            calculate_metric = True
            self.PSNR=0.0
            self.SSIM=0.0
        else:
            calculate_metric = False


        for val_data in self.val_loader:  ### Once load all frames

            val_frame_num = self.config['val']['val_frame_num']
            all_len = val_data['lq'].shape[1]
            all_output = []

            clip_name,frame_name = val_data['key'][0].split('/')

            ############# TODO: other different inference strategies
            for i in range(0,all_len,val_frame_num):  ### Fetch parts of the input video
                current_part = {}
                current_part['lq'] = val_data['lq'][:,i:min(i+val_frame_num,all_len),:,:,:]
                current_part['gt'] = val_data['gt'][:,i:min(i+val_frame_num,all_len),:,:,:]
                current_part['key'] = val_data['key']
                current_part['frame_list'] = val_data['frame_list'][i:min(i+val_frame_num,all_len)]

                self.part_lq = current_part['lq'].cuda()
                self.netG.eval()
                with torch.no_grad():
                    self.part_output = self.netG(self.part_lq)
                self.netG.train()
                
                if self.opts.fix_iter == float("inf"):  ## Eval mode for the flow estimation, even though for training
                    self.netG.module.spynet.eval()

                all_output.append(self.part_output.detach().cpu().squeeze(0))
                del self.part_lq
                del self.part_output
            #############


            self.val_output = torch.cat(all_output, dim=0)
            self.gt = val_data['gt'].squeeze(0)
            self.lq = val_data['lq'].squeeze(0)
            if self.config['datasets']['val']['normalizing']:
                self.val_output = (self.val_output + 1)/2
                self.gt = (self.gt + 1)/2
                self.lq = (self.lq + 1)/2
            torch.cuda.empty_cache()

            gt_imgs = []
            sr_imgs = []
            for j in range(len(self.val_output)):
                gt_imgs.append(tensor2img(self.gt[j]))
                sr_imgs.append(tensor2img(self.val_output[j]))
            
            ### Save the image
            for id, sr_img in zip(val_data['frame_list'],sr_imgs):
                save_place = os.path.join(self.config['path']['visualization'],self.config['datasets']['val']['name'],clip_name,str(id.item()).zfill(8)+'.png')
                dir_name = os.path.abspath(os.path.dirname(save_place))
                os.makedirs(dir_name, exist_ok=True)
                cv2.imwrite(save_place,sr_img)


            if calculate_metric:
                PSNR_this_video = [calculate_psnr(sr,gt) for sr,gt in zip(sr_imgs,gt_imgs)]
                SSIM_this_video = [calculate_ssim(sr,gt) for sr,gt in zip(sr_imgs,gt_imgs)]
                self.PSNR += sum(PSNR_this_video) / len(PSNR_this_video)
                self.SSIM += sum(SSIM_this_video) / len(SSIM_this_video)


        
        if calculate_metric:
            self.PSNR /= len(self.val_loader)
            self.SSIM /= len(self.val_loader)

            log_str = f"Validation on {self.config['datasets']['val']['name']}\n"
            log_str += f'\t # PSNR: {self.PSNR:.4f}\n'
            log_str += f'\t # SSIM: {self.SSIM:.4f}\n'

            self.logger.info(log_str)


    def train(self):
        
        self.logger.info(f'Start training from epoch: {0}, iter: {0}')
        data_time, iter_time = time.time(), time.time()
        start_time = time.time()

        for epoch in range(self.epoch):

            ## Reset the status
            self.train_sampler.set_epoch(epoch)
            # previous_seed = np.random.get_state()[1][0]
            # np.random.seed(previous_seed+epoch)
            # random.seed(previous_seed+epoch)

            for data in self.train_loader:
                data_time = time.time() - data_time

                lq, gt = data['lq'].cuda(), data['gt'].cuda()
                self.iteration+=1
                self.optimize_parameters(lq, gt, self.iteration)
                iter_time = time.time() - iter_time

                if self.iteration % self.config['logger']['print_freq']==0 and self.opts.global_rank==0:
                    log_vars = {'epoch': epoch, 'iter': self.iteration}
                    log_vars.update({'lrs': [self.old_lr, self.old_gan_lr]}) ## List learning rate: include the lrs of flow and generator
                    log_vars.update({'iter_time': iter_time, 'data_time': data_time, 'start_time': start_time})
                    log_vars.update(self.log_dict) ## Record the loss values
                    self.print_iter_message(log_vars)

                if self.iteration % self.config['logger']['save_checkpoint_freq']==0 and self.opts.global_rank==0:
                    self.logger.info('Saving models and training states.')
                    self.save_model(epoch, self.iteration)


                if self.iteration % self.config['val']['val_freq']==0 and self.opts.global_rank==0:
                    self.validation()

                data_time = time.time()
                iter_time = time.time()


            if epoch > self.config['trainer']['nepoch_steady']:
                self.update_learning_rate()

