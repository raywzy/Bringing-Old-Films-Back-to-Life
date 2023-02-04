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
import argparse
import yaml

sys.path.append(os.path.dirname(sys.path[0]))

import torch

from VP_code.data.dataset import Film_dataset_1
from VP_code.utils.util import worker_set_seed, get_root_logger, set_device, frame_to_video
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



def Load_model(opts,config_dict):

    net = importlib.import_module('VP_code.models.' + opts.model_name)
    netG = net.Video_Backbone()
    model_path = os.path.join('OUTPUT',opts.name,'models','net_G_{}.pth'.format(str(opts.which_iter).zfill(5)))
    checkpoint = torch.load(model_path)
    netG.load_state_dict(checkpoint['netG'])
    netG.cuda()
    print("Finish loading model ...")

    return netG

def Load_dataset(opts,config_dict):

    val_dataset = Film_dataset_1(config_dict['datasets']['val'])
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False, sampler=None)
    print("Finish loading dataset ...")
    print("Test set statistics:")
    print(f'\n\tNumber of test videos: {len(val_dataset)}')

    return val_loader

def validation(opts, config_dict, loaded_model, val_loader):

    if "metrics" in config_dict['val']:
        calculate_metric = True
        PSNR=0.0
        SSIM=0.0
    else:
        calculate_metric = False

    loaded_model.eval()
    # test_clip_par_folder = config_dict['datasets']['val']['dataroot_lq'].split('/')[-1]


    for val_data in val_loader:  ### Once load all frames

        val_frame_num = config_dict['val']['val_frame_num']
        all_len = val_data['lq'].shape[1]
        all_output = []

        clip_name,frame_name = val_data['key'][0].split('/')
        test_clip_par_folder = val_data['video_name'][0]  ## The video name

        frame_name_list = val_data['name_list']

        part_output=None
        for i in range(0,all_len,opts.temporal_stride):
                        
            current_part = {}
            current_part['lq'] = val_data['lq'][:,i:min(i+val_frame_num,all_len),:,:,:]
            current_part['gt'] = val_data['gt'][:,i:min(i+val_frame_num,all_len),:,:,:]
            current_part['key'] = val_data['key']
            current_part['frame_list'] = val_data['frame_list'][i:min(i+val_frame_num,all_len)]   

            part_lq = current_part['lq'].cuda()
            # if part_output is not None:
            #     part_lq[:,:val_frame_num-opts.temporal_stride,:,:,:] = part_output[:,opts.temporal_stride-val_frame_num:,:,:,:]

            with torch.no_grad():
                part_output = loaded_model(part_lq)
            
            if i == 0:
                all_output.append(part_output.detach().cpu().squeeze(0))
            else:
                restored_temporal_length = min(i+val_frame_num,all_len)-i-(val_frame_num-opts.temporal_stride)
                all_output.append(part_output[:,0-restored_temporal_length:,:,:,:].detach().cpu().squeeze(0))
            
            del part_lq

            if (i+val_frame_num)>=all_len:
                break
        #############

        val_output = torch.cat(all_output, dim=0)
        gt = val_data['gt'].squeeze(0)
        lq = val_data['lq'].squeeze(0)
        if config_dict['datasets']['val']['normalizing']:
            val_output = (val_output + 1)/2
            gt = (gt + 1)/2
            lq = (lq + 1)/2
        torch.cuda.empty_cache()

        gt_imgs = []
        sr_imgs = []
        for j in range(len(val_output)):
            gt_imgs.append(tensor2img(gt[j]))
            sr_imgs.append(tensor2img(val_output[j]))
        
        ### Save the image
        for id, sr_img in enumerate(sr_imgs):
            save_place = os.path.join(opts.save_place,opts.name,'test_results_'+str(opts.temporal_length)+"_"+str(opts.which_iter), test_clip_par_folder, clip_name, frame_name_list[id][0])
            dir_name = os.path.abspath(os.path.dirname(save_place))
            os.makedirs(dir_name, exist_ok=True)
            cv2.imwrite(save_place,sr_img)

        ### To Video directly TODO: currently only support 1-depth sub-folder test clip [√]
        if test_clip_par_folder==os.path.basename(opts.input_video_url):
            input_clip_url = os.path.join(opts.input_video_url, clip_name)
        else:
            input_clip_url = os.path.join(opts.input_video_url, test_clip_par_folder, clip_name)
        
        restored_clip_url = os.path.join(opts.save_place, opts.name, 'test_results_'+str(opts.temporal_length)+"_"+str(opts.which_iter), test_clip_par_folder, clip_name)
        video_save_url = os.path.join(opts.save_place, opts.name, 'test_results_'+str(opts.temporal_length)+"_"+str(opts.which_iter), test_clip_par_folder, clip_name+'.avi')
        frame_to_video(input_clip_url, restored_clip_url, video_save_url)
        ### 

        if calculate_metric:
            PSNR_this_video = [calculate_psnr(sr,gt) for sr,gt in zip(sr_imgs,gt_imgs)]
            SSIM_this_video = [calculate_ssim(sr,gt) for sr,gt in zip(sr_imgs,gt_imgs)]
            PSNR += sum(PSNR_this_video) / len(PSNR_this_video)
            SSIM += sum(SSIM_this_video) / len(SSIM_this_video)


    
    if calculate_metric:
        PSNR /= len(val_loader)
        SSIM /= len(val_loader)

        log_str = f"Validation on {opts.input_video_url}\n"
        log_str += f'\t # PSNR: {PSNR:.4f}\n'
        log_str += f'\t # SSIM: {SSIM:.4f}\n'

        print(log_str)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--name',type=str,default='',help='The name of this experiment')
    parser.add_argument('--model_name',type=str,default='',help='The name of adopted model')
    parser.add_argument('--which_iter',type=str,default='latest',help='Load which iteraiton')
    parser.add_argument('--input_video_url',type=str,default='',help='degraded video input')
    parser.add_argument('--gt_video_url',type=str,default='',help='gt video')
    parser.add_argument('--temporal_length',type=int,default=15,help='How many frames should be processed in one forward')
    parser.add_argument('--temporal_stride',type=int,default=3,help='Stride value while sliding window')
    parser.add_argument('--save_place',type=str,default='OUTPUT',help='save place')

    opts = parser.parse_args()

    with open(os.path.join('./configs',opts.name+'.yaml'), 'r') as stream:
        config_dict = yaml.safe_load(stream)

    config_dict['datasets']['val']['dataroot_gt'] = opts.gt_video_url
    config_dict['datasets']['val']['dataroot_lq'] = opts.input_video_url
    config_dict['val']['val_frame_num'] = opts.temporal_length

    loaded_model = Load_model(opts,config_dict)
    val_loader = Load_dataset(opts,config_dict)

    validation(opts,config_dict,loaded_model,val_loader)
