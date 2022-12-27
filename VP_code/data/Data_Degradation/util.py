### Now try to simulate the degradation of real-world old film
# import sys
# sys.path.append('/home/wanziyu/workspace/project/Video_Restoration/')

from .blend_modes import *
import cv2
import PIL
from PIL import Image
from skimage.color import rgb2gray,rgb2lab
import random
from io import BytesIO
import albumentations as A
import os
import numpy as np

from scipy import ndimage
import scipy
import scipy.stats as ss
from scipy.interpolate import interp2d
from scipy.linalg import orth
from .texture_augmentation import texture_generator, moving_line_texture_generator
# from VP_code.utils.data_util import img2tensor, paired_random_crop, augment

# def texture_augmentation(texture_image, target_h, target_w):

#     # h,w = texture_image.shape
#     ## Random resize
#     texture_image = texture_image.resize((random.randint(target_w,target_w*2),random.randint(target_h,target_h*2)))
#     w,h = texture_image.size
#     ## Random crop
#     top = random.randint(0, h - target_h)
#     left = random.randint(0, w - target_w)
#     croped_texture = texture_image.crop((left,top,left+target_w,top+target_h))
    
#     return croped_texture


def texture_blending(image, texture, folder_name, verbose=False): ## Input: PIL | Return: PIL

    ## Keep the same dimension  TODO: More augmentation
    w,h=image.size
    # texture=texture.resize((h,w))
    # texture=texture_augmentation(texture,h,w)
    texture = texture_generator(texture, folder_name, h, w)
    ##


    image = np.array(image.convert('RGBA')).astype(float)
    texture = np.array(texture.convert('RGBA')).astype(float)

    mode = random.randint(0, 2)
    if folder_name == "011":
        mode = 0
    if verbose:
        distortion_type = ['addition', 'subtract', 'multiply']
        print(distortion_type[mode])

    if mode==0:
        scratched_image=addition(image,texture,opacity=random.uniform(0.6, 1.0))
    elif mode==1:
        scratched_image=subtract(image,texture,opacity=random.uniform(0.6, 1.0))
    elif mode==2:
        scratched_image=multiply(image,texture,opacity=random.uniform(0.6, 1.0))

    scratched_image = np.uint8(scratched_image)[:, :, :3]
    x=Image.fromarray(scratched_image).convert("L")

    return x

def moving_line_texture_blending(image, texture, last_texture, folder_name, mode, verbose=False): ## Input: PIL | Return: PIL

    ## Keep the same dimension  TODO: More augmentation
    w,h=image.size
    # texture=texture.resize((h,w))
    # texture=texture_augmentation(texture,h,w)
    texture_P = moving_line_texture_generator(texture, last_texture, h, w)
    ##


    image = np.array(image.convert('RGBA')).astype(float)
    texture = np.array(texture_P.convert('RGBA')).astype(float)

    if verbose:
        distortion_type = ['addition', 'subtract', 'multiply']
        print(distortion_type[mode])

    if mode==0:
        scratched_image=addition(image,texture,opacity=random.uniform(0.6, 1.0))
    elif mode==1:
        scratched_image=subtract(image,texture,opacity=random.uniform(0.6, 1.0))
    elif mode==2:
        scratched_image=multiply(image,texture,opacity=random.uniform(0.6, 1.0))

    scratched_image = np.uint8(scratched_image)[:, :, :3]
    x=Image.fromarray(scratched_image).convert("L")

    return x, texture_P

def texture_blending_generate_validation(image, texture, folder_name, verbose=False): ## Remove the texture augmentation

    ## Keep the same dimension  TODO: More augmentation
    h,w=image.size
    # texture=texture.resize((h,w))
    # texture=texture_augmentation(texture,h,w)
    texture = texture.resize((h,w))
    ##


    image = np.array(image.convert('RGBA')).astype(float)
    texture = np.array(texture.convert('RGBA')).astype(float)

    mode = random.randint(0, 2)
    if folder_name == "011":
        mode = 0
    if verbose:
        distortion_type = ['addition', 'subtract', 'multiply']
        print(distortion_type[mode])

    if mode==0:
        scratched_image=addition(image,texture,opacity=random.uniform(0.6, 1.0))
    elif mode==1:
        scratched_image=subtract(image,texture,opacity=random.uniform(0.6, 1.0))
    elif mode==2:
        scratched_image=multiply(image,texture,opacity=random.uniform(0.6, 1.0))

    scratched_image = np.uint8(scratched_image)[:, :, :3]
    x=Image.fromarray(scratched_image).convert("L")

    return x

##################################################################################

def pil_to_np(img_PIL):
    '''Converts image in PIL format to np.array.
    From W x H x C [0...255] to C x W x H [0..1]
    '''
    ar = np.array(img_PIL)

    if len(ar.shape) == 3:
        ar = ar.transpose(2, 0, 1)
    else:
        ar = ar[None, ...]

    return ar.astype(np.float32) / 255.


def np_to_pil(img_np):
    '''Converts image in np.array format to PIL image.
    From C x W x H [0..1] to  W x H x C [0...255]
    '''
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)



def jpeg_artifact(image, verbose=False):
    image = np.array(image)
    image[image >= 255] = 255
    image[image < 0] = 0
    image = image.astype(np.uint8)
    image = Image.fromarray(image)

    quality = random.randint(40, 100)
    with BytesIO() as f:
        image.save(f, format='JPEG', quality=quality)
        f.seek(0)
        image_jpeg = Image.open(f).convert('L')

    if verbose:
        print('JPEG quality =', quality)

    return image_jpeg

def jpeg_artifact_v2(image, quality, verbose=False):
    image = np.array(image)
    image[image >= 255] = 255
    image[image < 0] = 0
    image = image.astype(np.uint8)
    image = Image.fromarray(image)

    quality_variance = random.randint(-15, 15)
    new_quality = np.clip(quality+quality_variance,40,100)

    with BytesIO() as f:
        image.save(f, format='JPEG', quality=int(new_quality))
        f.seek(0)
        image_jpeg = Image.open(f).convert('L')

    if verbose:
        print('JPEG quality =', new_quality)

    return image_jpeg

def downsampling_artifact(img, verbose=False):
    w,h=img.size

    new_w=random.randint(int(w/2),w)
    new_h=random.randint(int(h/2),h)

    img=img.resize((new_w,new_h),Image.BICUBIC)

    if random.uniform(0,1)<0.5:
        img=img.resize((w,h),Image.NEAREST)
    else:
        img = img.resize((w, h), Image.BILINEAR)

    if verbose:
        print('down-sampling size =(%d,%d)'%(new_w,new_h))

    return img

def random_scaling(img,x,y):

    down_method = ['bicubic','bilinear','lanczos']
    selected_method = random.sample(down_method,1)[0]
    if selected_method == 'bicubic':
        img=img.resize((x,y),Image.BICUBIC)
    if selected_method == 'bilinear':
        img=img.resize((x,y),Image.BILINEAR)
    if selected_method == 'lanczos':
        img=img.resize((x,y),Image.LANCZOS)
    
    return img

def downsampling_artifact_v2(img, verbose=False):
    w,h=img.size

    new_w = int(w/4)
    new_h = int(h/4)

    img = random_scaling(img,new_w,new_h)
    img = random_scaling(img,w,h)

    if verbose:
        print('down-sampling size =(%d,%d)'%(new_w,new_h))

    return img

def downsampling_artifact_v3(img, verbose=False):
    w,h=img.size

    rnum = np.random.rand()
    if rnum > 0.8:  # up
        sf1 = random.uniform(1, 2)
    elif rnum < 0.7:  # down
        sf1 = random.uniform(0.5/4, 1)
    else:
        sf1 = 1.0

    new_w = int(sf1 * w)
    new_h = int(sf1 * h)

    img = random_scaling(img,new_w,new_h)
    # img = random_scaling(img,w,h)

    if verbose:
        print('down-sampling size =(%d,%d)'%(new_w,new_h))

    return img


def downsampling_artifact_v3_fixed(img,params,verbose=False):
    w,h=img.size

    rnum = params['rnum']
    if rnum > 0.8:  # up
        sf1 = params['up_scale']
    elif rnum < 0.7:  # down
        sf1 = params['down_scale']
    else:
        sf1 = 1.0

    new_w = int(sf1 * w)
    new_h = int(sf1 * h)

    img = random_scaling(img,new_w,new_h)
    # img = random_scaling(img,w,h)

    if verbose:
        print('down-sampling size =(%d,%d)'%(new_w,new_h))

    return img


def blur_artifact(img, verbose=False):


    x=np.array(img)
    kernel_size_candidate=[(3,3),(5,5),(7,7)]
    kernel_size=random.sample(kernel_size_candidate,1)[0]
    std=random.uniform(1.,5.)

    #print("The gaussian kernel size: (%d,%d) std: %.2f"%(kernel_size[0],kernel_size[1],std))
    blur=cv2.GaussianBlur(x,kernel_size,std)

    if verbose:
        print("Blur kernel =",kernel_size)

    return Image.fromarray(blur.astype(np.uint8))

def blur_artifact_v2(img, kernel_size, std, verbose=False):


    x=np.array(img)
    # kernel_size_candidate=[(3,3),(5,5),(7,7)]
    # kernel_size=random.sample(kernel_size_candidate,1)[0]
    # std=random.uniform(1.,5.)

    std_variance=random.uniform(-1.,1.)
    new_std = np.clip(std + std_variance, 1., 5.)

    #print("The gaussian kernel size: (%d,%d) std: %.2f"%(kernel_size[0],kernel_size[1],std))
    blur=cv2.GaussianBlur(x,kernel_size,new_std)

    if verbose:
        print("Blur kernel =",kernel_size)

    return Image.fromarray(blur.astype(np.uint8))

def gm_blur_kernel(mean, cov, size=15):
    center = size / 2.0 + 0.5
    k = np.zeros([size, size])
    for y in range(size):
        for x in range(size):
            cy = y - center + 1
            cx = x - center + 1
            k[y, x] = ss.multivariate_normal.pdf([cx, cy], mean=mean, cov=cov)

    k = k / np.sum(k)
    return k

def anisotropic_Gaussian(ksize=15, theta=np.pi, l1=6, l2=6):
    """ generate an anisotropic Gaussian kernel
    Args:
        ksize : e.g., 15, kernel size
        theta : [0,  pi], rotation angle range
        l1    : [0.1,50], scaling of eigenvalues
        l2    : [0.1,l1], scaling of eigenvalues
        If l1 = l2, will get an isotropic Gaussian kernel.
    Returns:
        k     : kernel
    """
    v = np.dot(np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]), np.array([1., 0.]))
    V = np.array([[v[0], v[1]], [v[1], -v[0]]])
    D = np.array([[l1, 0], [0, l2]])
    Sigma = np.dot(np.dot(V, D), np.linalg.inv(V))
    k = gm_blur_kernel(mean=[0, 0], cov=Sigma, size=ksize)

    return k

def fspecial_gaussian(hsize, sigma):
    hsize = [hsize, hsize]
    siz = [(hsize[0]-1.0)/2.0, (hsize[1]-1.0)/2.0]
    std = sigma
    [x, y] = np.meshgrid(np.arange(-siz[1], siz[1]+1), np.arange(-siz[0], siz[0]+1))
    arg = -(x*x + y*y)/(2*std*std)
    h = np.exp(arg)
    h[h < scipy.finfo(float).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h = h/sumh
    return h

def fspecial(filter_type, *args, **kwargs):
    '''
    python code from:
    https://github.com/ronaldosena/imagens-medicas-2/blob/40171a6c259edec7827a6693a93955de2bd39e76/Aulas/aula_2_-_uniform_filter/matlab_fspecial.py
    '''
    if filter_type == 'gaussian':
        return fspecial_gaussian(*args, **kwargs)
    # if filter_type == 'laplacian':
    #     return fspecial_laplacian(*args, **kwargs)

def add_blur(img, sf=4):
    wd2 = 4.0 + sf
    wd = 2.0 + 0.2*sf
    if random.random() < 0.5:
        l1 = wd2*random.random()
        l2 = wd2*random.random()
        k = anisotropic_Gaussian(ksize=2*random.randint(2,11)+3, theta=random.random()*np.pi, l1=l1, l2=l2)
    else:
        k = fspecial('gaussian', 2*random.randint(2,11)+3, wd*random.random())
    img = ndimage.filters.convolve(img, np.expand_dims(k, axis=2), mode='mirror')

    return img

def add_blur_fixed(img,params):

    wd2 = 4.0 + 4
    wd = 2.0 + 0.2*4

    if params['type_value'] < 0.5:
        l1 = wd2 * float(np.clip(params['l1_value']+(random.random()-0.5)/10.,1e-8,1-1e-8))
        l2 = wd2 * float(np.clip(params['l2_value']+(random.random()-0.5)/10.,1e-8,1-1e-8))
        k = anisotropic_Gaussian(ksize=2*params['shape_value']+3, theta=float(np.clip(params['angle_value']+(random.random()-0.5)/5.,1e-8,1-1e-8))*np.pi, l1=l1, l2=l2)
    else:
        k = fspecial('gaussian', 2*params['shape_value']+3, wd * float(np.clip(params['l1_value']+(random.random()-0.5)/10.,1e-8,1-1e-8)))
    img = ndimage.filters.convolve(img, np.expand_dims(k, axis=2), mode='mirror')

    return img

# def add_blur_fixed_v2(img,params):

#     wd2 = 4.0 + 4
#     wd = 2.0 + 0.2*4

#     if params['type_value'] < 0.5:
#         l1 = wd2 * (params['l1_value']+(random.random()-0.5)/100.)
#         l2 = wd2 * (params['l2_value']+(random.random()-0.5)/100.)
#         try:
#             k = anisotropic_Gaussian(ksize=2*params['shape_value']+3, theta=(params['angle_value'])*np.pi, l1=l1, l2=l2)
#         except:
#             print((l1,l2))
#             print((params['angle_value'])*np.pi)
#     else:
#         k = fspecial('gaussian', 2*params['shape_value']+3, wd * (params['l1_value']+(random.random()-0.5)/100.))
#     img = ndimage.filters.convolve(img, np.expand_dims(k, axis=2), mode='mirror')

#     return img

def gaussian_noise_artifact(image,std_l,std_r,verbose=False):

    ## Give PIL, return the noisy PIL

    img_pil=pil_to_np(image)

    mean=0
    std=random.uniform(std_l/255.,std_r/255.)
    gauss=np.random.normal(loc=mean,scale=std,size=img_pil.shape)
    noisy=img_pil+gauss
    noisy=np.clip(noisy,0,1).astype(np.float32)

    if verbose:
        print("Gaussian noise std =", std)

    return np_to_pil(noisy)

def gaussian_noise_artifact_v2(image,std,verbose=False):

    ## Give PIL, return the noisy PIL

    img_pil=pil_to_np(image)

    mean=0
    # std=random.uniform(std_l/255.,std_r/255.)
    std_variance=random.uniform(-0.5, 0.5)
    new_std = np.clip(std + std_variance/255. , 5.0/255., 10.0/255.)

    gauss=np.random.normal(loc=mean,scale=new_std,size=img_pil.shape)
    noisy=img_pil+gauss
    noisy=np.clip(noisy,0,1).astype(np.float32)

    if verbose:
        print("Gaussian noise std =", new_std)

    return np_to_pil(noisy)

def speckle_noise_artifact(image,std_l,std_r,verbose=False):

    ## Give PIL, return the noisy PIL

    img_pil=pil_to_np(image)

    mean=0
    std=random.uniform(std_l/255.,std_r/255.)
    gauss=np.random.normal(loc=mean,scale=std,size=img_pil.shape)
    noisy=img_pil+gauss*img_pil
    noisy=np.clip(noisy,0,1).astype(np.float32)

    if verbose:
        print("Speckle noise std =", std)

    return np_to_pil(noisy)

def speckle_noise_artifact_v2(image,std,verbose=False):

    ## Give PIL, return the noisy PIL

    img_pil=pil_to_np(image)

    mean=0
    # std=random.uniform(std_l/255.,std_r/255.)
    std_variance=random.uniform(-0.5, 0.5)
    new_std = np.clip(std + std_variance/255. , 5.0/255., 10.0/255.)

    gauss=np.random.normal(loc=mean,scale=new_std,size=img_pil.shape)
    noisy=img_pil+gauss*img_pil
    noisy=np.clip(noisy,0,1).astype(np.float32)

    if verbose:
        print("Speckle noise std =", new_std)

    return np_to_pil(noisy)

def degradation(image, distortion_probability=0.7, verbose=False):   ## TODO: add fractal noise, contrast/saturation/

    distortion_types = ['blur', 'noise', 'jpeg', 'downsample']
    distortion_sequence = np.random.permutation(len(distortion_types))
    for distortion_index in distortion_sequence:
        distortion = distortion_types[distortion_index]
        if random.uniform(0, 1) < distortion_probability:
            if distortion == 'blur':
                image = blur_artifact(image, verbose)
            elif distortion == 'noise':
                noise_type = random.choice([1, 2])
                if noise_type==1:
                    image = gaussian_noise_artifact(image, 5, 10, verbose)
                elif noise_type==2:
                    image = speckle_noise_artifact(image, 5, 10, verbose)
            elif distortion == 'jpeg': 
                image = jpeg_artifact(image, verbose)
            elif distortion == 'downsample':
                image = downsampling_artifact(image, verbose)
    
    return image

def degradation_v2(image, distortion_sequence, distortion_degree, distortion_probability, verbose=False):

    P=0.7
    distortion_types = ['blur', 'noise', 'jpeg', 'downsample']
    for i,distortion_index in enumerate(distortion_sequence):
        distortion = distortion_types[distortion_index]
        if distortion == 'blur' and distortion_probability[i] < P:
            image = blur_artifact_v2(image,distortion_degree['blur_kernel'],distortion_degree['blur_std'], verbose)
        elif distortion == 'noise' and distortion_probability[i] < P:
            noise_type = random.choice([1, 2])
            if noise_type==1:
                image = gaussian_noise_artifact_v2(image, distortion_degree['noise_std'], verbose)
            elif noise_type==2:
                image = speckle_noise_artifact_v2(image, distortion_degree['noise_std'], verbose)
        elif distortion == 'jpeg' and distortion_probability[i] < P: 
            image = jpeg_artifact_v2(image, distortion_degree['jpeg_quality'] , verbose)
        elif distortion == 'downsample':
            image = downsampling_artifact_v2(image, verbose)
    
    return image       

def degradation_v2_old(image, distortion_sequence, distortion_degree, distortion_probability, verbose=False):

    P=0.7
    distortion_types = ['blur', 'noise', 'jpeg', 'downsample']
    for i,distortion_index in enumerate(distortion_sequence):
        distortion = distortion_types[distortion_index]
        if distortion_probability[i] < P :
            if distortion == 'blur':
                image = blur_artifact_v2(image,distortion_degree['blur_kernel'],distortion_degree['blur_std'], verbose)
            elif distortion == 'noise':
                noise_type = random.choice([1, 2])
                if noise_type==1:
                    image = gaussian_noise_artifact_v2(image, distortion_degree['noise_std'], verbose)
                elif noise_type==2:
                    image = speckle_noise_artifact_v2(image, distortion_degree['noise_std'], verbose)
            elif distortion == 'jpeg': 
                image = jpeg_artifact_v2(image, distortion_degree['jpeg_quality'] , verbose)
            elif distortion == 'downsample':
                image = downsampling_artifact_v2(image, verbose)
    
    return image        

def degradation_v3(image, distortion_sequence, distortion_degree, distortion_probability, verbose=False):

    P=1.0
    x,y = image.size
    distortion_types = ['blur', 'downsample', 'noise', 'jpeg']
    for i,distortion_index in enumerate(distortion_sequence):
        distortion = distortion_types[i]
        if distortion == 'blur' and distortion_probability[i] < P:
            temp_cv2 = transfer_2(image)
            # image = blur_artifact_v2(image,distortion_degree['blur_kernel'],distortion_degree['blur_std'], verbose)
            temp_cv2 = add_blur_fixed(temp_cv2,distortion_degree)
            # temp_cv2 = add_blur(temp_cv2)
            image = transfer_1(temp_cv2)
        elif distortion == 'noise' and distortion_probability[i] < P:
            noise_type = random.choice([1, 2])
            if noise_type==1:
                image = gaussian_noise_artifact_v2(image, distortion_degree['noise_std'], verbose)
            elif noise_type==2:
                image = speckle_noise_artifact_v2(image, distortion_degree['noise_std'], verbose)
        elif distortion == 'jpeg' and distortion_probability[i] < P: 
            image = jpeg_artifact_v2(image, distortion_degree['jpeg_quality'] , verbose)
        elif distortion == 'downsample':
            image = downsampling_artifact_v3_fixed(image,distortion_degree,verbose=False)

    image = random_scaling(image,x,y)
    return image


def degradation_simple_debug(image, distortion_sequence, distortion_degree, distortion_probability, verbose=False):

    P=1.0
    x,y = image.size
    distortion_types = ['downsample']
    for i,distortion_index in enumerate(distortion_sequence):
        distortion = distortion_types[i]
        if distortion == 'blur' and distortion_probability[i] < P:
            temp_cv2 = transfer_2(image)
            # image = blur_artifact_v2(image,distortion_degree['blur_kernel'],distortion_degree['blur_std'], verbose)
            temp_cv2 = add_blur_fixed(temp_cv2,distortion_degree)
            # temp_cv2 = add_blur(temp_cv2)
            image = transfer_1(temp_cv2)
        elif distortion == 'noise' and distortion_probability[i] < P:
            noise_type = random.choice([1, 2])
            if noise_type==1:
                image = gaussian_noise_artifact_v2(image, distortion_degree['noise_std'], verbose)
            elif noise_type==2:
                image = speckle_noise_artifact_v2(image, distortion_degree['noise_std'], verbose)
        elif distortion == 'jpeg' and distortion_probability[i] < P: 
            image = jpeg_artifact_v2(image, distortion_degree['jpeg_quality'] , verbose)
        elif distortion == 'downsample':
            image = downsampling_artifact_v3_fixed(image,distortion_degree,verbose=False)

    image = random_scaling(image,x,y)
    return image

def color_jitter(image, verbose=False):

    transform=A.ColorJitter(brightness=[0.8,1.2], contrast=[0.9,1.0], saturation=[1.0,1.0], hue=0.0, always_apply=True, p=0.5)
    image=np.array(image)
    jittered=transform(image=image)['image']

    x=Image.fromarray(jittered.astype('uint8')).convert('L')
    return x


def transfer_1(img_np):

    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(ar, cv2.COLOR_BGR2RGB)).convert("L")

def transfer_2(img_pil):

    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img.astype(np.float32) / 255.


def degradation_video_list(video_list,texture_url='./noise_data'):

    texture_templates=getfilelist(texture_url)
    # print(texture_templates)
    degraded=[]
    gt_L=[]

    for x in video_list:
        
        frame_pil=transfer_1(x)
        # frame_pil=Image.fromarray(x.astype('uint8')).convert("L") 
        selected_texture_url=random.sample(texture_templates,1)[0]
        texture_pil=Image.open(selected_texture_url).convert("L")
        
        x1=texture_blending(frame_pil,texture_pil)
        x2=degradation(x1)
        x3=color_jitter(x2)

        x3=x3.convert("RGB")

        degraded.append(transfer_2(x3))

        frame_pil=frame_pil.convert("RGB")
        gt_L.append(transfer_2(frame_pil))

    return degraded,gt_L

def degradation_video_list_2(video_list,texture_url='./noise_data'):

    texture_templates=getfilelist(texture_url)
    # print(texture_templates)
    degraded=[]
    gt_L=[]

    ## For each video, fix the core degradation pattern, then disturb among frames
    distortion_types = ['blur', 'noise', 'jpeg', 'downsample']
    distortion_sequence = np.random.permutation(len(distortion_types))
    distortion_degree = {'blur_kernel':random.sample([(3,3),(5,5),(7,7)],1)[0], 'blur_std': random.uniform(1.,5.),'noise_std': random.uniform(5.0/255.,10.0/255.), 'jpeg_quality': random.randint(40, 100)}
    distortion_probability = [random.uniform(0, 1) for i in range(4)]
    ##


    for x in video_list:
        
        frame_pil=transfer_1(x)
        # frame_pil=Image.fromarray(x.astype('uint8')).convert("L") 
        selected_texture_url=random.sample(texture_templates,1)[0]

        folder_name = selected_texture_url.split('/')[-2]
        texture_pil=Image.open(selected_texture_url).convert("L")
        
        x1=texture_blending(frame_pil,texture_pil,folder_name)
        x2=degradation_v2_old(x1, distortion_sequence, distortion_degree, distortion_probability)
        x3=color_jitter(x2)

        x3=x3.convert("RGB")

        degraded.append(transfer_2(x3))

        frame_pil=frame_pil.convert("RGB")
        gt_L.append(transfer_2(frame_pil))

    return degraded,gt_L


def add_sharpening(img, weight=0.5, radius=50, threshold=10):
    """USM sharpening. borrowed from real-ESRGAN
    Input image: I; Blurry image: B.
    1. K = I + weight * (I - B)
    2. Mask = 1 if abs(I - B) > threshold, else: 0
    3. Blur mask:
    4. Out = Mask * K + (1 - Mask) * I
    Args:
        img (Numpy array): Input image, HWC, BGR; float32, [0, 1].
        weight (float): Sharp weight. Default: 1.
        radius (float): Kernel size of Gaussian blur. Default: 50.
        threshold (int):
    """
    if radius % 2 == 0:
        radius += 1
    blur = cv2.GaussianBlur(img, (radius, radius), 0)
    residual = img - blur
    mask = np.abs(residual) * 255 > threshold
    mask = mask.astype('float32')
    soft_mask = cv2.GaussianBlur(mask, (radius, radius), 0)

    K = img + weight * residual
    K = np.clip(K, 0, 1)
    return soft_mask * K + (1 - soft_mask) * img


def degradation_video_list_3(video_list,texture_url='./noise_data'):

    texture_templates=getfilelist(texture_url)
    # print(texture_templates)
    degraded=[]
    gt_L=[]

    ## For each video, fix the core degradation pattern, then disturb among frames
    distortion_types = ['blur', 'noise', 'jpeg', 'downsample']
    distortion_sequence = np.random.permutation(len(distortion_types))
    distortion_degree = {'type_value': random.random(), 'l1_value': random.random(), 'l2_value': random.random(),'angle_value': random.random(), 'shape_value': random.randint(2,11), 'noise_std': random.uniform(5.0/255.,10.0/255.), 'jpeg_quality': random.randint(40, 100), 'rnum': np.random.rand(), 'up_scale': random.uniform(1, 2), 'down_scale': random.uniform(0.5/4, 1)}
    distortion_probability = [random.uniform(0, 1) for i in range(4)]
    ##

    # params['type_value']
    # params['l1_value']
    # params['l2_value']
    # params['angle_value']
    # params['shape_value']

    for x in video_list:
        
        frame_pil = transfer_1(x)
        frame_cv2 = transfer_2(frame_pil.convert("RGB"))
        GT_cv2 = add_sharpening(frame_cv2)

        # frame_pil=Image.fromarray(x.astype('uint8')).convert("L") 
        selected_texture_url=random.sample(texture_templates,1)[0]

        folder_name = selected_texture_url.split('/')[-2]
        texture_pil=Image.open(selected_texture_url).convert("L")
        
        x1=texture_blending(frame_pil,texture_pil,folder_name)
        x2=degradation_v3(x1, distortion_sequence, distortion_degree, distortion_probability)
        x3=color_jitter(x2)

        x3=x3.convert("RGB")
        degraded.append(transfer_2(x3))

        gt_L.append(GT_cv2)

    return degraded,gt_L


def degradation_video_list_4(video_list,texture_url='./noise_data'): ## Add moving lines

    texture_templates=getfilelist(texture_url)
    # print(texture_templates)
    degraded=[]
    gt_L=[]

    ## For each video, fix the core degradation pattern, then disturb among frames
    distortion_types = ['blur', 'noise', 'jpeg', 'downsample']
    distortion_sequence = np.random.permutation(len(distortion_types))
    distortion_degree = {'type_value': random.random(), 'l1_value': random.random(), 'l2_value': random.random(),'angle_value': random.random(), 'shape_value': random.randint(2,11), 'noise_std': random.uniform(5.0/255.,10.0/255.), 'jpeg_quality': random.randint(40, 100), 'rnum': np.random.rand(), 'up_scale': random.uniform(1, 2), 'down_scale': random.uniform(0.5/4, 1)}
    distortion_probability = [random.uniform(0, 1) for i in range(4)]
    ##

    moving_line_flag = False
    if random.uniform(0, 1)<0.2:
        moving_line_flag = True  
        texture_templates=getfilelist_001(texture_url)
        selected_texture_url=random.sample(texture_templates,1)[0]
        texture_pil=Image.open(selected_texture_url).convert("L") ##  Initial line texture
        mode = random.randint(0, 2) ## pre-define the blending mode
        # print("Yes, Moving Line is training")
        last_texture=None

    for x in video_list:
        
        frame_pil = transfer_1(x)
        frame_cv2 = transfer_2(frame_pil.convert("RGB"))
        GT_cv2 = add_sharpening(frame_cv2)

        # frame_pil=Image.fromarray(x.astype('uint8')).convert("L")
        if not moving_line_flag: 
            selected_texture_url=random.sample(texture_templates,1)[0]
            folder_name = selected_texture_url.split('/')[-2]
            texture_pil=Image.open(selected_texture_url).convert("L")
            x1=texture_blending(frame_pil,texture_pil,folder_name)
        else:
            x1, last_texture=moving_line_texture_blending(frame_pil, texture_pil, last_texture,'001', mode)


        x2=degradation_v3(x1, distortion_sequence, distortion_degree, distortion_probability)
        x3=color_jitter(x2)

        x3=x3.convert("RGB")
        degraded.append(transfer_2(x3))

        gt_L.append(GT_cv2)

    return degraded,gt_L




def getfilelist(path):
    rlist=[]
    for dir,folder,file_name in os.walk(path):
        for i in file_name:
            t = "%s/%s"%(dir,i)
            rlist.append(t)
    return rlist

def getfilelist_001(path):
    rlist=[]
    for dir,folder,file_name in os.walk(path):
        for i in file_name:
            t = "%s/%s"%(dir,i)
            if dir.endswith("001"):
                rlist.append(t)
    return rlist


def degradation_video_list_simple_debug(video_list,texture_url='./noise_data'):

    texture_templates=getfilelist(texture_url)
    # print(texture_templates)
    degraded=[]
    gt_L=[]

    ## For each video, fix the core degradation pattern, then disturb among frames
    distortion_types = ['downsample']
    distortion_sequence = np.random.permutation(len(distortion_types))
    distortion_degree = {'type_value': random.random(), 'l1_value': random.random(), 'l2_value': random.random(),'angle_value': random.random(), 'shape_value': random.randint(2,11), 'noise_std': random.uniform(5.0/255.,10.0/255.), 'jpeg_quality': random.randint(40, 100), 'rnum': np.random.rand(), 'up_scale': random.uniform(1, 2), 'down_scale': random.uniform(0.5/4, 1)}
    distortion_probability = [random.uniform(0, 1) for i in range(4)]
    ##

    # params['type_value']
    # params['l1_value']
    # params['l2_value']
    # params['angle_value']
    # params['shape_value']

    for x in video_list:
        
        frame_pil = transfer_1(x)
        frame_cv2 = transfer_2(frame_pil.convert("RGB"))
        GT_cv2 = add_sharpening(frame_cv2)

        # frame_pil=Image.fromarray(x.astype('uint8')).convert("L") 
        selected_texture_url=random.sample(texture_templates,1)[0]

        folder_name = selected_texture_url.split('/')[-2]
        texture_pil=Image.open(selected_texture_url).convert("L")
        
        x1=frame_pil
        x2=degradation_simple_debug(x1, distortion_sequence, distortion_degree, distortion_probability)
        x3=x2

        x3=x3.convert("RGB")
        degraded.append(transfer_2(x3))

        gt_L.append(GT_cv2)

    return degraded,gt_L

def getfilelist(path):
    rlist=[]
    for dir,folder,file_name in os.walk(path):
        for i in file_name:
            t = "%s/%s"%(dir,i)
            rlist.append(t)
    return rlist


def degradation_video_url(frame_url,texture_url,save_gt_url,save_degradation_url):

    os.makedirs(save_degradation_url,exist_ok=True)
    os.makedirs(save_gt_url,exist_ok=True)

    texture_templates=getfilelist(texture_url)

    print("The # of Texture Templates is %d"%(len(texture_templates)))
    frame_list=sorted(os.listdir(frame_url))


    count=0
    for frame_name in frame_list:
        
        count+=1
        c_frame_url=os.path.join(frame_url,frame_name)
        selected_texture_url=random.sample(texture_templates,1)[0]

        frame_pil=Image.open(c_frame_url).convert("L")

        ### Resize to 128
        frame_pil=frame_pil.resize((128,128))
        frame_pil.save(os.path.join(save_gt_url,frame_name))

        texture_pil=Image.open(selected_texture_url).convert("L")
        
        x1=texture_blending(frame_pil,texture_pil)
        x2=degradation(x1)
        x3=color_jitter(x2)
        x3.save(os.path.join(save_degradation_url,frame_name))
        
        print("Finish Frame %d"%(count))

def degradation_video_url_v2(frame_url,texture_url,save_degradation_url):

    os.makedirs(save_degradation_url,exist_ok=True)
    # os.makedirs(save_gt_url,exist_ok=True)

    texture_templates=getfilelist(texture_url)

    print("The # of Texture Templates is %d"%(len(texture_templates)))
    frame_list=sorted(os.listdir(frame_url))

    ## For each video, fix the core degradation pattern, then disturb among frames
    distortion_types = ['blur', 'noise', 'jpeg', 'downsample']
    distortion_sequence = np.random.permutation(len(distortion_types))
    distortion_degree = {'blur_kernel':random.sample([(3,3),(5,5),(7,7)],1)[0], 'blur_std': random.uniform(1.,5.),'noise_std': random.uniform(5.0/255.,10.0/255.), 'jpeg_quality': random.randint(40, 100)}
    distortion_probability = [random.uniform(0, 1) for i in range(4)]
    ##

    count=0
    for frame_name in frame_list:
        
        count+=1
        c_frame_url=os.path.join(frame_url,frame_name)
        selected_texture_url=random.sample(texture_templates,1)[0]
        print("Texture_Url: %s"%(selected_texture_url))
        frame_pil=Image.open(c_frame_url).convert("L")


        ### Resize to 128
        frame_pil=frame_pil.resize((640,360),Image.LANCZOS)
        # frame_pil.save(os.path.join(save_gt_url,frame_name))

        folder_name = selected_texture_url.split('/')[-2]
        texture_pil=Image.open(selected_texture_url).convert("L")
        
        x1=texture_blending(frame_pil,texture_pil,folder_name)
        # x1=frame_pil
        x2=degradation_v2(x1, distortion_sequence, distortion_degree, distortion_probability, verbose=False)
        # x2=texture_blending(x2,texture_pil,folder_name)
        x3=color_jitter(x2)
        x3.save(os.path.join(save_degradation_url,frame_name))
        
        print("Finish Frame %d"%(count))

def degradation_video_url_v3(frame_url,texture_url,save_degradation_url,save_gt_url):

    os.makedirs(save_degradation_url,exist_ok=True)
    # os.makedirs(save_gt_url,exist_ok=True)

    texture_templates=getfilelist(texture_url)

    print("The # of Texture Templates is %d"%(len(texture_templates)))
    frame_list=sorted(os.listdir(frame_url))

    ## For each video, fix the core degradation pattern, then disturb among frames
    distortion_types = ['blur', 'noise', 'jpeg', 'downsample']
    distortion_sequence = np.random.permutation(len(distortion_types))
    distortion_degree = {'type_value': random.random(), 'l1_value': random.random(), 'l2_value': random.random(),'angle_value': random.random(), 'shape_value': random.randint(2,11), 'noise_std': random.uniform(5.0/255.,10.0/255.), 'jpeg_quality': random.randint(40, 100), 'rnum': np.random.rand(), 'up_scale': random.uniform(1, 2), 'down_scale': random.uniform(0.5/4, 1)}
    distortion_probability = [random.uniform(0, 1) for i in range(4)]
    ##

    count=0
    for frame_name in frame_list:
        
        count+=1
        c_frame_url=os.path.join(frame_url,frame_name)
        selected_texture_url=random.sample(texture_templates,1)[0]
        # print("Texture_Url: %s"%(selected_texture_url))
        frame_pil=Image.open(c_frame_url).convert("L")


        ### Resize to 128
        frame_pil=frame_pil.resize((640,368),Image.LANCZOS)
        frame_cv2 = transfer_2(frame_pil.convert("RGB"))
        GT_cv2 = add_sharpening(frame_cv2)
        GT_pil = transfer_1(GT_cv2)
        GT_pil.save(os.path.join(save_gt_url,frame_name))

        folder_name = selected_texture_url.split('/')[-2]
        texture_pil=Image.open(selected_texture_url).convert("L")
        
        x1=texture_blending_generate_validation(frame_pil,texture_pil,folder_name)
        # x1=frame_pil
        x2=degradation_v3(x1, distortion_sequence, distortion_degree, distortion_probability, verbose=False)
        # x2=texture_blending(x2,texture_pil,folder_name)
        x3=color_jitter(x2)
        x3.save(os.path.join(save_degradation_url,frame_name))
        
        print("Finish Frame %d"%(count))



def degradation_video_colorization(video_list): ## RGB level degradation


    degraded=[]
    gt_L=[]

    Random_color_index = random.randint(0,len(video_list)-1)


    for id,x in enumerate(video_list):
        
        if id!=Random_color_index:
            frame_pil = transfer_1(x)
            frame_cv2 = transfer_2(frame_pil.convert("RGB"))
        else:
            frame_cv2 = x

        # Remove sharpening
        # GT_cv2 = add_sharpening(x)
        GT_cv2 = x

        # # frame_pil=Image.fromarray(x.astype('uint8')).convert("L") 
        # selected_texture_url=random.sample(texture_templates,1)[0]

        # folder_name = selected_texture_url.split('/')[-2]
        # texture_pil=Image.open(selected_texture_url).convert("L")
        
        # x1=texture_blending(frame_pil,texture_pil,folder_name)
        # x2=degradation_v3(x1, distortion_sequence, distortion_degree, distortion_probability)
        # x3=color_jitter(x2)

        # x3=x3.convert("RGB")
        degraded.append(frame_cv2)
        gt_L.append(GT_cv2)

    return degraded,gt_L


def degradation_video_colorization_v2(video_list): ## LAB level degradation


    degraded=[]
    gt_L=[]

    Random_color_index = random.randint(0,len(video_list)-1)


    for id,x in enumerate(video_list):
        
        LAB_array=x
        input_array = np.copy(LAB_array)
        if id!=Random_color_index:
            input_array[:,:,1:3] = 0.

        # Remove sharpening
        # GT_cv2 = add_sharpening(x)
        # print(input_array-LAB_array)
        degraded.append(input_array)
        gt_L.append(LAB_array)

    return degraded,gt_L


def degradation_video_colorization_v3(video_list): ## LAB level degradation | the last one don't drop color(reference) | random all gray input


    degraded=[]
    gt_L=[]
    end_index = len(video_list)-2
    Random_color_index = random.randint(0,end_index)
    all_gray_flag=False
    if random.uniform(0.0,1.0)<0.5:
        all_gray_flag=True

    for id,x in enumerate(video_list):
        
        LAB_array=x
        input_array = np.copy(LAB_array)
        if (id!=Random_color_index or all_gray_flag) and id<end_index+1:
            input_array[:,:,1:3] = 0.

        # Remove sharpening
        # GT_cv2 = add_sharpening(x)
        # print(input_array-LAB_array)
        degraded.append(input_array)
        gt_L.append(LAB_array)

    return degraded,gt_L


def degradation_video_colorization_v4(video_list): ## LAB level degradation | the last one don't drop color(reference) | the rest are all gray


    degraded=[]
    gt_L=[]
    end_index = len(video_list)-2
    Random_color_index = random.randint(0,end_index)
    all_gray_flag=True

    for id,x in enumerate(video_list):
        
        LAB_array=x
        input_array = np.copy(LAB_array)
        if (id!=Random_color_index or all_gray_flag) and id<end_index+1:
            input_array[:,:,1:3] = 0.

        # Remove sharpening
        # GT_cv2 = add_sharpening(x)
        # print(input_array-LAB_array)
        degraded.append(input_array)
        gt_L.append(LAB_array)

    return degraded,gt_L




if __name__=='__main__':


    img_num=10000
    save_gt_url = "./datasets/old_film_gt"
    save_lq_url = "./datasets/old_film_lq"
    os.makedirs(save_gt_url,exist_ok=True)
    os.makedirs(save_lq_url,exist_ok=True)

    for i in range(img_num):
        lq,gt=render_esrgan_data()
        lq=lq.convert("RGB")
        gt=gt.convert("RGB")
        lq.save(os.path.join(save_lq_url,str(i)+'.png'))
        gt.save(os.path.join(save_gt_url,str(i)+'.png'))
        print(i)
