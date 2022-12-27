import torch
import torch.nn.functional as F 
from torch import nn 

# from einops import rearrange
from VP_code.models.raft_flow import Get_RAFT
from VP_code.models.arch_util import ResidualBlockNoBN, flow_warp, make_layer 
from VP_code.models.Spatial_Restoration_2 import Swin_Spatial_2

class Gated_Aggregation(nn.Module):

    def __init__(self, hidden_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(Gated_Aggregation, self).__init__()

        self.activation = activation
        self.conv2d_projection_head = torch.nn.Conv2d(in_channels=3, out_channels=hidden_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.conv2d_1 = torch.nn.Conv2d(in_channels=hidden_channels+hidden_channels+1, out_channels=int(hidden_channels/2), kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.conv2d_2 = torch.nn.Conv2d(in_channels=int(hidden_channels/2), out_channels=1, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        # self.batch_norm2d = torch.nn.BatchNorm2d(out_channels)
        self.sigmoid = torch.nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, hidden_state, curr_lr, residual_indicator, head = False, return_mask = False, return_feat = False):

        latent_feature = self.conv2d_projection_head(curr_lr)
        x = self.activation(self.conv2d_1(torch.cat([hidden_state, latent_feature, residual_indicator], dim=1)))
        x = self.sigmoid(self.conv2d_2(x))

        if return_mask:
            if head:
                if return_feat:
                    return (latent_feature, latent_feature)
                else:
                    return (latent_feature, x)
            else:
                if return_feat:
                    return (x*hidden_state+(1-x)*latent_feature, latent_feature)
                else:    
                    return (x*hidden_state+(1-x)*latent_feature, x)

        else:
            if head:
                return latent_feature
            else:
                return x*hidden_state+(1-x)*latent_feature



class Video_Backbone(nn.Module):
    """BasicVSR: The Search for Essential Components in Video Super-Resolution and Beyond

    Only support x4 upsampling.

    Args:
        num_feat (int): Channel number of intermediate features. 
            Default: 64.
        num_block (int): Block number of residual blocks in each propagation branch.
            Default: 30.
        spynet_path (str): The path of Pre-trained SPyNet model.
            Default: None.
    """
    def __init__(self, num_feat=16, num_block=6, spynet_path=None):
        super(Video_Backbone, self).__init__()
        self.num_feat = num_feat
        self.num_block = num_block

        # Flow-based Feature Alignment
        self.spynet = Get_RAFT()

        # Bidirectional Propagation
        # self.forward_resblocks = ConvResBlock(num_feat + 3, num_feat, num_block)
        # self.backward_resblocks = ConvResBlock(num_feat + 3, num_feat, num_block)
        self.forward_resblocks = Swin_Spatial_2(embed_dim=64, depths=[2, 2, 2], num_heads=[4, 4, 4], mlp_ratio = 2, in_chans = num_feat)
        self.backward_resblocks = Swin_Spatial_2(embed_dim=64, depths=[2, 2, 2], num_heads=[4, 4, 4], mlp_ratio = 2, in_chans = num_feat)

        # Concatenate Aggregation
        self.concate = nn.Conv2d(num_feat * 2, num_feat, kernel_size=3, stride=1, padding=1, bias=True)

        # Hidden State Aggregation
        self.Forward_Aggregation = Gated_Aggregation(hidden_channels=num_feat,kernel_size=3,padding=1)
        self.Backward_Aggregation = Gated_Aggregation(hidden_channels=num_feat,kernel_size=3,padding=1)

        # Pixel-Shuffle Upsampling
        self.up1 = PSUpsample(num_feat, num_feat, scale_factor=1)
        self.up2 = PSUpsample(num_feat, num_feat, scale_factor=1)

        # The channel of the tail layers is 64
        self.conv_hr = nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=1, padding=1)
        self.conv_last = nn.Conv2d(num_feat, 3, kernel_size=3, stride=1, padding=1)

        # Global Residual Learning
        self.img_up = nn.Upsample(scale_factor=1, mode='bilinear', align_corners=False)

        # Activation Function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def comp_flow(self, lrs):
        """Compute optical flow using SPyNet for feature warping.

        Args:
            lrs (tensor): LR frames, the shape is (n, t, c, h, w)

        Return:
            tuple(Tensor): Optical flow. 
            forward_flow refers to the flow from current frame to the previous frame. 
            backward_flow is the flow from current frame to the next frame.
        """
        n, t, c, h, w = lrs.size()
        forward_lrs = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)    # n t c h w -> (n t) c h w
        backward_lrs = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)  # n t c h w -> (n t) c h w
        
        with torch.no_grad():
            _, forward_flow = self.spynet(forward_lrs*255, backward_lrs*255, iters=24, test_mode=True)
            forward_flow = forward_flow.view(n, t-1, 2, h, w)
            _ ,backward_flow = self.spynet(backward_lrs*255, forward_lrs*255, iters=24, test_mode=True)
            backward_flow = backward_flow.view(n, t-1, 2, h, w)
        return forward_flow, backward_flow

    def forward(self, lrs):
        n, t, c, h, w = lrs.size()
    
        assert h >= 64 and w >= 64, (
            'The height and width of input should be at least 64, '
            f'but got {h} and {w}.')
        
        forward_flow, backward_flow = self.comp_flow((lrs+1.)/2)

        # forward_flow = rearrange(forward_flow, 'n t c h w -> t n h w c').contiguous()
        # backward_flow = rearrange(backward_flow, 'n t c h w -> t n h w c').contiguous()
        # lrs = rearrange(lrs, 'n t c h w -> t n c h w').contiguous()

        # Backward Propagation
        rlt = []
        feat_prop = lrs.new_zeros(n, self.num_feat, h, w)
        residual_indicator = torch.zeros(n,1,h,w).cuda()
        for i in range(t-1, -1, -1):
            curr_lr = lrs[:, i, :, :, :]
            if i < t-1:
                flow = backward_flow[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

                pixel_prop = flow_warp(lrs[:, i+1, :, :, :], flow.permute(0, 2, 3, 1))
                residual_indicator = torch.abs(pixel_prop[:,:1,:,:] - curr_lr[:,:1,:,:])

                feat_prop = self.backward_resblocks(self.Backward_Aggregation(feat_prop,curr_lr,residual_indicator,head=False))
            else:
                feat_prop = self.backward_resblocks(self.Backward_Aggregation(feat_prop,curr_lr,residual_indicator,head=True))


            rlt.append(feat_prop)
        rlt = rlt[::-1]

        # Forward Propagation
        feat_prop = torch.zeros_like(feat_prop)
        residual_indicator = torch.zeros_like(residual_indicator)
        for i in range(0, t):
            curr_lr = lrs[:, i, :, :, :]
            if i > 0:
                flow = forward_flow[:, i-1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

                pixel_prop = flow_warp(lrs[:, i-1, :, :, :], flow.permute(0, 2, 3, 1))
                residual_indicator = torch.abs(pixel_prop[:,:1,:,:] - curr_lr[:,:1,:,:])

                feat_prop = self.forward_resblocks(self.Forward_Aggregation(feat_prop,curr_lr,residual_indicator,head=False))
            else:
                feat_prop = self.forward_resblocks(self.Forward_Aggregation(feat_prop,curr_lr,residual_indicator,head=True))
            

            # Fusion and Upsampling
            cat_feat = torch.cat([rlt[i], feat_prop], dim=1)
            sr_rlt = self.lrelu(self.concate(cat_feat))
            sr_rlt = self.lrelu(self.up1(sr_rlt))
            sr_rlt = self.lrelu(self.up2(sr_rlt))
            sr_rlt = self.lrelu(self.conv_hr(sr_rlt))
            sr_rlt = self.conv_last(sr_rlt)

            # Global Residual Learning
            base = self.img_up(curr_lr)

            sr_rlt += base
            rlt[i] = torch.tanh(sr_rlt)

        return torch.stack(rlt, dim=1)


    def visualiza_mask(self, lrs):
        n, t, c, h, w = lrs.size()
    
        assert h >= 64 and w >= 64, (
            'The height and width of input should be at least 64, '
            f'but got {h} and {w}.')
        
        forward_flow, backward_flow = self.comp_flow((lrs+1.)/2)

        # forward_flow = rearrange(forward_flow, 'n t c h w -> t n h w c').contiguous()
        # backward_flow = rearrange(backward_flow, 'n t c h w -> t n h w c').contiguous()
        # lrs = rearrange(lrs, 'n t c h w -> t n c h w').contiguous()

        # Backward Propagation
        backward_mask = []
        feat_prop = lrs.new_zeros(n, self.num_feat, h, w)
        residual_indicator = torch.zeros(n,1,h,w).cuda()
        for i in range(t-1, -1, -1):
            curr_lr = lrs[:, i, :, :, :]
            if i < t-1:
                flow = backward_flow[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

                pixel_prop = flow_warp(lrs[:, i+1, :, :, :], flow.permute(0, 2, 3, 1))
                residual_indicator = torch.abs(pixel_prop[:,:1,:,:] - curr_lr[:,:1,:,:])

                temp, learned_mask=self.Backward_Aggregation(feat_prop,curr_lr,residual_indicator,head=False,return_mask=True)
                feat_prop = self.backward_resblocks(temp)
            else:
                temp, learned_mask=self.Backward_Aggregation(feat_prop,curr_lr,residual_indicator,head=True,return_mask=True)
                feat_prop = self.backward_resblocks(temp)


            backward_mask.append(learned_mask)
        backward_mask = backward_mask[::-1]

        # Forward Propagation
        forward_mask = []
        feat_prop = torch.zeros_like(feat_prop)
        residual_indicator = torch.zeros_like(residual_indicator)
        for i in range(0, t):
            curr_lr = lrs[:, i, :, :, :]
            if i > 0:
                flow = forward_flow[:, i-1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

                pixel_prop = flow_warp(lrs[:, i-1, :, :, :], flow.permute(0, 2, 3, 1))
                residual_indicator = torch.abs(pixel_prop[:,:1,:,:] - curr_lr[:,:1,:,:])
                temp, learned_mask=self.Forward_Aggregation(feat_prop,curr_lr,residual_indicator,head=False,return_mask=True)
                feat_prop = self.forward_resblocks(temp)
            else:
                temp, learned_mask=self.Forward_Aggregation(feat_prop,curr_lr,residual_indicator,head=True,return_mask=True)
                feat_prop = self.forward_resblocks(temp)
            

            forward_mask.append(learned_mask)

        return torch.stack(backward_mask, dim=1), torch.stack(forward_mask, dim=1) 

    def visualiza_feature(self, lrs):
        n, t, c, h, w = lrs.size()
    
        assert h >= 64 and w >= 64, (
            'The height and width of input should be at least 64, '
            f'but got {h} and {w}.')
        
        forward_flow, backward_flow = self.comp_flow((lrs+1.)/2)



        # Backward Propagation
        backward_feat = []
        backward_state = []
        feat_prop = lrs.new_zeros(n, self.num_feat, h, w)
        residual_indicator = torch.zeros(n,1,h,w).cuda()
        for i in range(t-1, -1, -1):
            curr_lr = lrs[:, i, :, :, :]
            if i < t-1:
                flow = backward_flow[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

                pixel_prop = flow_warp(lrs[:, i+1, :, :, :], flow.permute(0, 2, 3, 1))
                residual_indicator = torch.abs(pixel_prop[:,:1,:,:] - curr_lr[:,:1,:,:])

                temp, learned_mask=self.Backward_Aggregation(feat_prop,curr_lr,residual_indicator,head=False,return_mask=True, return_feat=True)
                feat_prop = self.backward_resblocks(temp)
            else:
                temp, learned_mask=self.Backward_Aggregation(feat_prop,curr_lr,residual_indicator,head=True,return_mask=True, return_feat=True)
                feat_prop = self.backward_resblocks(temp)

            backward_feat.append(torch.mean(learned_mask,dim=1,keepdim=True))
            backward_state.append(torch.mean(temp,dim=1,keepdim=True))
        backward_feat = backward_feat[::-1]
        backward_state = backward_state[::-1]

        # Forward Propagation
        forward_feat = []
        forward_state = []
        feat_prop = torch.zeros_like(feat_prop)
        residual_indicator = torch.zeros_like(residual_indicator)
        for i in range(0, t):
            curr_lr = lrs[:, i, :, :, :]
            if i > 0:
                flow = forward_flow[:, i-1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

                pixel_prop = flow_warp(lrs[:, i-1, :, :, :], flow.permute(0, 2, 3, 1))
                residual_indicator = torch.abs(pixel_prop[:,:1,:,:] - curr_lr[:,:1,:,:])
                temp, learned_mask=self.Forward_Aggregation(feat_prop,curr_lr,residual_indicator,head=False,return_mask=True, return_feat=True)
                feat_prop = self.forward_resblocks(temp)
            else:
                temp, learned_mask=self.Forward_Aggregation(feat_prop,curr_lr,residual_indicator,head=True,return_mask=True, return_feat=True)
                feat_prop = self.forward_resblocks(temp)
            

            forward_feat.append(torch.mean(learned_mask,dim=1,keepdim=True))
            forward_state.append(torch.mean(temp,dim=1,keepdim=True))

        A = torch.stack(forward_feat, dim=1)
        B = torch.stack(forward_state, dim=1)
        print(A)
        print(B)
        return A,B

#############################
# Conv + ResBlock
class ConvResBlock(nn.Module):
    def __init__(self, in_feat, out_feat=64, num_block=30):
        super(ConvResBlock, self).__init__()

        conv_resblock = []
        conv_resblock.append(nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=1, padding=1, bias=True))
        conv_resblock.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        conv_resblock.append(make_layer(ResidualBlockNoBN, num_block, num_feat=out_feat))

        self.conv_resblock = nn.Sequential(*conv_resblock)

    def forward(self, x):
        return self.conv_resblock(x)

#############################
# Upsampling with Pixel-Shuffle
class PSUpsample(nn.Module):
    def __init__(self, in_feat, out_feat, scale_factor):
        super(PSUpsample, self).__init__()

        self.scale_factor = scale_factor
        self.up_conv = nn.Conv2d(in_feat, out_feat*scale_factor*scale_factor, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        x = self.up_conv(x)
        return F.pixel_shuffle(x, upscale_factor=self.scale_factor)


if __name__ == '__main__':
    model = BasicVSR()
    lrs = torch.randn(3, 4, 3, 64, 64)
    rlt = model(lrs)
    print(rlt.size())

