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

class Reconstruction_Metrics:
    def __init__(self, metric_list=['ssim', 'psnr', 'fid'], data_range=1, win_size=21, multichannel=True):
        self.data_range = data_range
        self.win_size = win_size
        self.multichannel = multichannel
        self.fid_calculate = FID()
        self.loss_fn_vgg = lpips.LPIPS(net='alex')
        for metric in metric_list:
            setattr(self, metric, True)

    def calculate_metric(self, fake_image_path, real_image_path):
        """
            inputs: .txt files, floders, image files (string), image files (list)
            gts: .txt files, floders, image files (string), image files (list)
        """
        fid_value = self.fid_calculate.calculate_from_disk(fake_image_path, real_image_path)
        # fid_value = 0
        psnr = []
        ssim = []
        lipis = []
        brisque = []
        niqe = []
        # image_name_list = [name for name in os.listdir(real_image_path) if
        #                    name.endswith((('.png', '.jpg', '.jpeg', '.JPG', '.bmp')))]

        frame_path_real = getfilelist(real_image_path)
        frame_path_fake = getfilelist(fake_image_path)

        # fake_image_name_list = os.listdir(fake_image_path)
        for i, (image_path_real,image_path_fake) in enumerate(zip(frame_path_real,frame_path_fake)):
            
            path_real = image_path_real
            path_fake = image_path_fake

            if not check(path_real,path_fake):
                print("Error")
                sys.exit(1)

            PIL_real = Image.open(path_real).convert('RGB')
            PIL_fake = Image.open(path_fake).convert('RGB')
            fake_torch_image = img_to_tensor(PIL_fake)
            real_torch_image = img_to_tensor(PIL_real)
            img_content_real = np.array(PIL_real).astype(np.float32) / 255.0
            img_content_fake = np.array(PIL_fake).astype(np.float32) / 255.0
            brisque_each_img=piq.brisque((fake_torch_image+1)/2, data_range=1., reduction='none')
            niqe_each_img = set_niqe(np.array(PIL_fake.convert("L")).astype(np.float32))
            psnr_each_img = peak_signal_noise_ratio(img_content_real, img_content_fake)
            ssim_each_image = structural_similarity(img_content_real, img_content_fake, data_range=self.data_range,
                                                    win_size=self.win_size, multichannel=self.multichannel)
            lipis_each_image = self.loss_fn_vgg(fake_torch_image, real_torch_image)
            lipis_each_image = lipis_each_image.detach().numpy()
            psnr.append(psnr_each_img)
            ssim.append(ssim_each_image)
            lipis.append(lipis_each_image)
            brisque.append(brisque_each_img)
            niqe.append(niqe_each_img)
        print(
              "PSNR: %.4f" % np.round(np.mean(psnr), 4),
              "PSNR Variance: %.4f" % np.round(np.var(psnr), 4))
        print(
              "SSIM: %.4f" % np.round(np.mean(ssim), 4),
              "SSIM Variance: %.4f" % np.round(np.var(ssim), 4))
        print(
              "LPIPS: %.4f" % np.round(np.mean(lipis), 4),
              "LPIPS Variance: %.4f" % np.round(np.var(lipis), 4))
        print(
              "BRISQUE: %.4f" % np.round(np.mean(brisque), 4),
              "BRISQUE Variance: %.4f" % np.round(np.var(brisque), 4))
        print(
              "NIQE: %.4f" % np.round(np.mean(niqe), 4),
              "NIQE Variance: %.4f" % np.round(np.var(niqe), 4))
        return np.round(np.mean(psnr), 4), np.round(np.mean(ssim), 4), np.round(np.mean(lipis), 4), fid_value, np.round(np.mean(brisque), 4), np.round(np.mean(niqe), 4)


def get_metric(fake_dir, real_dir):
    Get_metric = Reconstruction_Metrics()
    psnr_out, ssim_out, lipis_out, fid_value_out, brisque_out, niqe_out = Get_metric.calculate_metric(fake_dir, real_dir)
    save_txt = os.path.join('.', 'metric.txt')
    with open(save_txt, 'a') as txt2:
        txt2.write("psnr:")
        txt2.write(str(psnr_out))
        txt2.write("  ")
        txt2.write("ssim:")
        txt2.write(str(ssim_out))
        txt2.write(" ")
        txt2.write("lipis:")
        txt2.write(str(lipis_out))
        txt2.write(" ")
        txt2.write("fid:")
        txt2.write(str(fid_value_out))
        txt2.write(" ")
        txt2.write("brisque:")
        txt2.write(str(brisque_out))
        txt2.write(" ")
        txt2.write("niqe:")
        txt2.write(str(niqe_out))
        txt2.write('\n')



if __name__ == "__main__":
    real_path = './'
    fake_path = './'
    get_metric(fake_path, real_path)






