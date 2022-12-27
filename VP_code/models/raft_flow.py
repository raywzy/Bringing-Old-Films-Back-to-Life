import argparse
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
from VP_code.models.arch_util import flow_warp
from VP_code.models.RAFT_core.raft import RAFT

import math
from collections import OrderedDict
from torch import nn as nn
from torch.nn import functional as F

import cv2
import os
import numpy as np
from torchvision.utils import save_image
import time
import seaborn as sns

def Get_RAFT():
    opts = argparse.Namespace()
    opts.model = "pretrained_models/raft-sintel.pth"
    # opts.model = "/home/wanziyu/workspace/project/Video_Restoration/pretrained_models/raft-sintel.pth"
    opts.dataset = None
    opts.small = False
    opts.mixed_precision = False
    opts.alternate_corr = False

    model = torch.nn.DataParallel(RAFT(opts))
    model.load_state_dict(torch.load(opts.model))

    model = model.module
    model.eval()

    return model


def check_flow_occlusion(flow_f, flow_b):
    """
    Compute occlusion map through forward/backward flow consistency check
    """
    def get_occlusion(flow1, flow2):
        grid_flow = grid + flow1
        grid_flow[0, :, :] = 2.0 * grid_flow[0, :, :] / max(W - 1, 1) - 1.0
        grid_flow[1, :, :] = 2.0 * grid_flow[1, :, :] / max(H - 1, 1) - 1.0
        grid_flow = grid_flow.permute(1, 2, 0)        
        flow2_inter = torch.nn.functional.grid_sample(flow2[None, ...], grid_flow[None, ...])[0]
        score = torch.exp(- torch.sum((flow1 + flow2_inter) ** 2, dim=0) / 2.)
        occlusion = (score > 0.5)
        return occlusion[None, ...].float()

    C, H, W = flow_f.size()
    # Mesh grid 
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, H, W)
    yy = yy.view(1, H, W)
    grid = torch.cat((xx, yy), 0).float().cuda()

    occlusion_f = get_occlusion(flow_f, flow_b)
    occlusion_b = get_occlusion(flow_b, flow_f)
    flow_f = torch.cat((flow_f, occlusion_f), 0)
    flow_b = torch.cat((flow_b, occlusion_b), 0)

    return flow_f,flow_b



if __name__=='__main__':

    # spynet=SpyNet(load_path='/home/wanziyu/workspace/project/BasicSR/pretrained_models/network-sintel-final.pytorch')
    # spynet=spynet.cuda()
    #### First we read a video clip
    spynet = Get_RAFT()
    spynet.cuda()

    net_params = sum(map(lambda x: x.numel(), spynet.parameters()))
    
    print("RAFT Parameter #: %d"%(net_params))

    # url = '/home/wanziyu/workspace/datasets/Old_Film/video_clips/Around_the_world_in_1896/0001'
    # url='/home/wanziyu/workspace/project/Video_Restoration/VP_code/models/test_flow/clip_1'
    url = '/home/wanziyu/workspace/project/VSR/basicsr/data/Data_Degradation/Validation_Set/input/001'

    save_url='/home/wanziyu/workspace/project/Video_Restoration/VP_code/models/test_flow/001/warped_1'
    save_occulusion_url='/home/wanziyu/workspace/project/Video_Restoration/VP_code/models/test_flow/001/occulusion_1'
    save_flow_url='/home/wanziyu/workspace/project/Video_Restoration/VP_code/models/test_flow/001/flow_1'

    save_original_frame_url='/home/wanziyu/workspace/project/Video_Restoration/VP_code/models/test_flow/001/original_1'
    save_residual_url = '/home/wanziyu/workspace/project/Video_Restoration/VP_code/models/test_flow/001/residual_1'
    # save_url_256='/home/wanziyu/workspace/project/BasicSR/basicsr/models/archs/original_256'

    os.makedirs(save_url,exist_ok=True)
    os.makedirs(save_occulusion_url,exist_ok=True)
    os.makedirs(save_flow_url,exist_ok=True)
    os.makedirs(save_original_frame_url,exist_ok=True)
    os.makedirs(save_residual_url,exist_ok=True)

    frame_list=sorted(os.listdir(url))
    frame_list=frame_list[:20]
    tensor_list=[]

    for x in frame_list:
        img=cv2.imread(os.path.join(url,x))
        # img = cv2.resize(img, (256,256), interpolation = cv2.INTER_AREA)
        # cv2.imwrite(os.path.join(save))
        img = img.astype(np.float32) / 255.
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        tensor_list.append(img)

    lrs=torch.stack(tensor_list,dim=0) # 1*t*c*h*w
    lrs=lrs.unsqueeze(0).cuda()
    n, t, c, h, w = lrs.size()
    forward_lrs = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)    # n t c h w -> (n t) c h w
    backward_lrs = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)  # n t c h w -> (n t) c h w
    
    # print("++++++start timing++++++")
    # s_time = time.time()

    with torch.no_grad():
        _, forward_flow = spynet(forward_lrs*255., backward_lrs*255., iters=24, test_mode=True)
        forward_flow = forward_flow.view(n, t-1, 2, h, w)
        # print("++++++end timing++++++")
        # e_time = time.time()

        # print("Total time: %.5f" %(e_time-s_time))

        _,backward_flow = spynet(backward_lrs*255., forward_lrs*255., iters=24, test_mode=True)
        backward_flow = backward_flow.view(n, t-1, 2, h, w)

    import flow_vis

    print("Start to warp")
    for i in range(t-1):
        input_img=backward_lrs[i,:,:,:].unsqueeze(0)
        final_flow=forward_flow[:,i,:,:,:]
        warped_img=flow_warp(input_img,final_flow.permute(0, 2, 3, 1))
        save_image(warped_img[0],os.path.join(save_url,frame_list[i]))
        save_image(input_img[0],os.path.join(save_original_frame_url,frame_list[i]))

        residual_tensor = warped_img[0]-backward_lrs[i+1,0,:,:]
        residual_numpy = np.abs(residual_tensor[0].cpu().numpy())
        sns_plot=sns.heatmap(residual_numpy,cmap="YlGnBu",yticklabels=False,xticklabels=False)
        sns_plot.figure.savefig(os.path.join(save_residual_url,frame_list[i+1]),bbox_inches='tight')
        sns_plot.figure.clf()

        mask_f,mask_b=check_flow_occlusion(forward_flow[0,i,:,:,:],backward_flow[0,i,:,:,:])
        save_image(mask_f,os.path.join(save_occulusion_url,frame_list[i][:-4]+'_mask_f.png'))
        save_image(mask_b,os.path.join(save_occulusion_url,frame_list[i][:-4]+'_mask_b.png'))

        temp = final_flow[0,:,:,:]
        temp = temp.permute(1,2,0).cpu().detach().numpy()

        flow_color = flow_vis.flow_to_color(temp, convert_to_bgr=False)

        cv2.imwrite(os.path.join(save_flow_url,"flow_f_%s.png"%(str(i))), flow_color)


    # import flow_vis

    # temp = forward_flow[0,2,:,:,:]
    # temp = temp.permute(1,2,0).cpu().detach().numpy()

    # flow_color = flow_vis.flow_to_color(temp, convert_to_bgr=False)

    # cv2.imwrite("flow.png", flow_color)