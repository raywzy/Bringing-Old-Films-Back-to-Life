import os
import argparse
import numpy as np
from fid import FID
from PIL import Image
# from natsort import natsorted
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
import torch
import lpips
import shutil
import sys
import piq
import skvideo.measure

def set_niqe(img):
    if img is None:
        return None
    return skvideo.measure.niqe(img)[0]


def getfilelist(file_path):
    all_file = []
    for dir,folder,file in os.walk(file_path):
        for i in file:
            t = "%s/%s"%(dir,i)
            all_file.append(t)
    all_file = sorted(all_file)
    return all_file

def check(a, b):
    clip_name_a = a.split("/")[-2]
    frame_name_a = a.split("/")[-1]
    clip_name_b = b.split("/")[-2]
    frame_name_b = b.split("/")[-1]

    if clip_name_a == clip_name_b and frame_name_a == frame_name_b:
        return True
    print(clip_name_a+"/"+frame_name_a)
    print(clip_name_b+"/"+frame_name_b)
    return False

def img_to_tensor(img):
    # image_size_w = img.width
    # image_size_h = img.height
    # image = (np.asarray(img) / 255.0).reshape(image_size_w * image_size_h, 3).transpose().reshape(3, image_size_h, image_size_w)
    # image = (np.asarray(img) / 255.0).reshape(image_size_w * image_size_h, 3).transpose().reshape(3, image_size_h, image_size_w)
    # torch_image = torch.from_numpy(image).float()
    # torch_image = torch_image * 2.0 - 1.0
    # torch_image = torch_image.unsqueeze(0)
    torch_image = torch.tensor(np.asarray(img)).permute(2, 0, 1)[None, ...] / 255.
    torch_image = torch_image * 2.0 - 1.0
    return torch_image




if __name__ == "__main__":
    fake_path = '/home/wanziyu/workspace/project/DeOldify/video/bwframes'
    # get_metric(fake_path, real_path)

    video_list=os.listdir(fake_path)
    brisque=[]
    niqe=[]
    for x in video_list:

        # for y in os.listdir(os.path.join(fake_path,x)):
        #     if y.endswith('.avi'):
        #         continue
            for z in os.listdir(os.path.join(fake_path,x)):

                img_url = os.path.join(fake_path,x,z)


                PIL_fake = Image.open(img_url).convert('L')
                PIL_fake = PIL_fake.convert('RGB')


                fake_torch_image = img_to_tensor(PIL_fake)
                brisque_each_img=piq.brisque((fake_torch_image+1)/2, data_range=1., reduction='none')
                niqe_each_img = set_niqe(np.array(PIL_fake.convert("L")).astype(np.float32))

                brisque.append(brisque_each_img)
                niqe.append(niqe_each_img)
            print("Finish %s"%(os.path.join(fake_path,x)))
    print(
        "BRISQUE: %.4f" % np.round(np.mean(brisque), 4),
        "BRISQUE Variance: %.4f" % np.round(np.var(brisque), 4))
    print(
            "NIQE: %.4f" % np.round(np.mean(niqe), 4),
            "NIQE Variance: %.4f" % np.round(np.var(niqe), 4))