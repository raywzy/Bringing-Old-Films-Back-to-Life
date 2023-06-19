import os
import cv2
import torch
from PIL import Image
import numpy as np
import importlib
import torchvision.transforms as transforms
from VP_code.utils.data_util import img2tensor
from VP_code.utils.data_util import tensor2img
from VP_code.data.Data_Degradation.util import degradation_video_list, degradation_video_list_2, degradation_video_list_3, degradation_video_list_4, degradation_video_list_simple_debug, degradation_video_colorization, transfer_1, transfer_2, degradation_video_colorization_v2, degradation_video_colorization_v3, degradation_video_colorization_v4

import sys
from tqdm import tqdm
sys.path.append(os.path.dirname(sys.path[0]))

def load_model(model_path) :

    net = importlib.import_module('VP_code.models.RNN_Swin_4')
    netG = net.Video_Backbone()
    checkpoint = torch.load(model_path)
    netG.load_state_dict(checkpoint['netG'])
    netG.cuda()
    print('Finish loading model ...')

    return netG

def resize_368_short_side(img):
    
    frame_pil = transfer_1(img)

    width, height = frame_pil.size

    if width<height:
        new_height =  int (368 * height / width)
        new_height = new_height // 16 * 16
        new_width = 368
    else:
        new_width =  int (368 * width / height)
        new_width = new_width // 16 * 16
        new_height = 368
    
    frame_pil = frame_pil.resize((new_width,new_height),resample=Image.BILINEAR)
    return transfer_2(frame_pil.convert("RGB"))

if __name__ == '__main__' :
    
    '''
    -----------------
    Load data
    -----------------
    '''
    save_dir = './test_result/'
    dir_path = './test_data/'
    frames_dir = sorted(os.listdir(dir_path))
    current_len = len(frames_dir)

    model = load_model('./OUTPUT/RNN_Swin_4/models/net_G_200000.pth')
    model.eval()
    model.cuda()

    temporal_length = 20
    temporal_stride = 10

    img_list = []
    for i in range(0, len(frames_dir)) :
        frame_path = frames_dir[i]
        img = cv2.imread(os.path.join(dir_path, frame_path))
        img = img.astype(np.float32) / 255.
        img_list.append(img)
    
    for i in range(len(img_list)):
        img_list[i] = resize_368_short_side(img_list[i])
    
    img_results = img2tensor(img_list)
    transform_normalize=transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
    for i in range(len(img_results)):
        img_results[i]=transform_normalize(img_results[i])
    img_lqs = torch.stack(img_results[:current_len], dim=0).unsqueeze(0)
    print(img_lqs.shape)

    '''
    -----------------
    Testing
    -----------------
    '''
    print('testing ...')
    all_output = []

    for i in tqdm(range(0, current_len, temporal_stride)) :
        data = img_lqs[:, i:min(i+temporal_length, current_len), :, :, :].cuda()

        with torch.no_grad() :
            output = model(data)
        
        if i == 0 :
            all_output.append(output.detach().cpu().squeeze(0))
        else:
            restored__length = min(i+temporal_length,current_len)-i-(temporal_length-temporal_stride)
            all_output.append(output[:,0-restored__length:,:,:,:].detach().cpu().squeeze(0))

        del data

        if (i + temporal_length) >= current_len :
            break

    output = torch.cat(all_output, dim=0)
    output = (output + 1) / 2
    torch.cuda.empty_cache()

    '''
    -----------------
    Saving
    -----------------
    '''
    print('saving ...')
    sr_imgs = []
    for j in range(len(output)):
        sr_imgs.append(tensor2img(output[j]))

    for id, sr_img in tqdm(enumerate(sr_imgs)) :
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, str(id)+zfill(5) + '.png'))