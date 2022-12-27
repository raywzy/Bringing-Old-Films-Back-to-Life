import math
from collections import OrderedDict
import torch
from torch import nn as nn
from torch.nn import functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
from VP_code.models.arch_util import flow_warp

import cv2
import os
import numpy as np
from torchvision.utils import save_image
import time

class BasicModule(nn.Module):
    """Basic Module for SpyNet.
    """

    def __init__(self):
        super(BasicModule, self).__init__()

        self.basic_module = nn.Sequential(
            nn.Conv2d(
                in_channels=8,
                out_channels=32,
                kernel_size=7,
                stride=1,
                padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=7,
                stride=1,
                padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=7,
                stride=1,
                padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=32,
                out_channels=16,
                kernel_size=7,
                stride=1,
                padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=16,
                out_channels=2,
                kernel_size=7,
                stride=1,
                padding=3))

    def forward(self, tensor_input):
        return self.basic_module(tensor_input)


class SpyNet(nn.Module):
    """SpyNet architecture.

    Args:
        load_path (str): path for pretrained SpyNet. Default: None.
    """

    def __init__(self, load_path=None):
        super(SpyNet, self).__init__()
        self.basic_module = nn.ModuleList([BasicModule() for _ in range(6)])
        
        if load_path:
            state_dict = OrderedDict()
            for k, v in torch.load(load_path).items():
                k = k.replace('moduleBasic', 'basic_module')
                state_dict[k] = v
            self.load_state_dict(state_dict)

        self.register_buffer(
            'mean',
            torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            'std',
            torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def preprocess(self, tensor_input):
        tensor_output = (tensor_input - self.mean) / self.std
        return tensor_output

    def process(self, ref, supp):
        flow = []

        ref = [self.preprocess(ref)]
        supp = [self.preprocess(supp)]

        for level in range(5):
            ref.insert(
                0,
                F.avg_pool2d(
                    input=ref[0],
                    kernel_size=2,
                    stride=2,
                    count_include_pad=False))
            supp.insert(
                0,
                F.avg_pool2d(
                    input=supp[0],
                    kernel_size=2,
                    stride=2,
                    count_include_pad=False))

        flow = ref[0].new_zeros([
            ref[0].size(0), 2,
            int(math.floor(ref[0].size(2) / 2.0)),
            int(math.floor(ref[0].size(3) / 2.0))
        ])

        for level in range(len(ref)):
            upsampled_flow = F.interpolate(
                input=flow,
                scale_factor=2,
                mode='bilinear',
                align_corners=True) * 2.0

            if upsampled_flow.size(2) != ref[level].size(2):
                upsampled_flow = F.pad(
                    input=upsampled_flow, pad=[0, 0, 0, 1], mode='replicate')
            if upsampled_flow.size(3) != ref[level].size(3):
                upsampled_flow = F.pad(
                    input=upsampled_flow, pad=[0, 1, 0, 0], mode='replicate')

            flow = self.basic_module[level](torch.cat([
                ref[level],
                flow_warp(
                    supp[level],
                    upsampled_flow.permute(0, 2, 3, 1),
                    interp_mode='bilinear',
                    padding_mode='border'), upsampled_flow
            ], 1)) + upsampled_flow
        return flow

    def forward(self, ref, supp):
        assert ref.size() == supp.size()

        h, w = ref.size(2), ref.size(3)
        w_floor = math.floor(math.ceil(w / 32.0) * 32.0)
        h_floor = math.floor(math.ceil(h / 32.0) * 32.0)

        ref = F.interpolate(
            input=ref,
            size=(h_floor, w_floor),
            mode='bilinear',
            align_corners=False)
        supp = F.interpolate(
            input=supp,
            size=(h_floor, w_floor),
            mode='bilinear',
            align_corners=False)

        flow = F.interpolate(
            input=self.process(ref, supp),
            size=(h, w),
            mode='bilinear',
            align_corners=False)

        flow[:, 0, :, :] *= float(w) / float(w_floor)
        flow[:, 1, :, :] *= float(h) / float(h_floor)

        return flow



# def flow_warp(x,
#               flow,
#               interp_mode='bilinear',
#               padding_mode='zeros',
#               align_corners=True):
#     """Warp an image or feature map with optical flow.

#     Args:
#         x (Tensor): Tensor with size (n, c, h, w).
#         flow (Tensor): Tensor with size (n, h, w, 2), normal value.
#         interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
#         padding_mode (str): 'zeros' or 'border' or 'reflection'.
#             Default: 'zeros'.
#         align_corners (bool): Before pytorch 1.3, the default value is
#             align_corners=True. After pytorch 1.3, the default value is
#             align_corners=False. Here, we use the True as default.

#     Returns:
#         Tensor: Warped image or feature map.
#     """
#     assert x.size()[-2:] == flow.size()[1:3]
#     _, _, h, w = x.size()
#     # create mesh grid
#     grid_y, grid_x = torch.meshgrid(
#         torch.arange(0, h).type_as(x),
#         torch.arange(0, w).type_as(x))
#     grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
#     grid.requires_grad = False

#     vgrid = grid + flow
#     # scale grid to [-1,1]
#     vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
#     vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
#     vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
#     output = F.grid_sample(
#         x,
#         vgrid_scaled,
#         mode=interp_mode,
#         padding_mode=padding_mode,
#         align_corners=align_corners)

#     # TODO, what if align_corners=False
#     return output

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
    # flow_f = torch.cat((flow_f, occlusion_f), 0)
    # flow_b = torch.cat((flow_b, occlusion_b), 0)

    return occlusion_f,occlusion_b

if __name__=='__main__':

    spynet=SpyNet(load_path='/home/wanziyu/workspace/project/BasicSR/pretrained_models/network-sintel-final.pytorch')
    spynet=spynet.cuda()
    #### First we read a video clip

    url='/home/wanziyu/workspace/project/Video_Restoration/OUTPUT/RNN_5_gt_flow/bad_case/821/gt'
    save_url='/home/wanziyu/workspace/project/Video_Restoration/VP_code/models/test_flow/spynet_821/warped_1'
    save_occulusion_url='/home/wanziyu/workspace/project/Video_Restoration/VP_code/models/test_flow/spynet_821/occulusion_1'
    save_flow_url='/home/wanziyu/workspace/project/Video_Restoration/VP_code/models/test_flow/spynet_821/flow_1'
    # save_url_256='/home/wanziyu/workspace/project/BasicSR/basicsr/models/archs/original_256'

    os.makedirs(save_url,exist_ok=True)
    os.makedirs(save_occulusion_url,exist_ok=True)
    os.makedirs(save_flow_url,exist_ok=True)

    frame_list=sorted(os.listdir(url))
    frame_list=frame_list[:]
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
    
    print("++++++start timing++++++")
    s_time = time.time()
    forward_flow = spynet(forward_lrs, backward_lrs).view(n, t-1, 2, h, w)
    print("++++++end timing++++++")
    e_time = time.time()

    print("Total time: %.5f" %(e_time-s_time))

    backward_flow = spynet(backward_lrs, forward_lrs).view(n, t-1, 2, h, w)

    import flow_vis

    print("Start to warp")
    for i in range(t-1):
        input_img=backward_lrs[i,:,:,:].unsqueeze(0)
        final_flow=forward_flow[:,i,:,:,:]
        warped_img=flow_warp(input_img,final_flow.permute(0, 2, 3, 1))
        save_image(warped_img[0],os.path.join(save_url,frame_list[i]))

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