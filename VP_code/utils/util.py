import os
import cv2
import io
import sys
import glob
import time
import zipfile
import subprocess
import logging
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision
import torch.distributed as dist

# set random seed 
def set_seed(seed, base=0, is_set=True):
    seed += base
    assert seed >=0, '{} >= {}'.format(seed, 0)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

### Don't use this now
def worker_set_seed(seed, base=0, num_workers=0, rank=0, is_set=True):
    seed += base
    seed += num_workers*rank
    assert seed >=0, '{} >= {}'.format(seed, 0)
    np.random.seed(seed)
    random.seed(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_device(args):
    if torch.cuda.is_available():
        if isinstance(args, list):
            return (item.cuda() for item in args)
        else:
            return args.cuda()
    return args


def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())


def mkdir_and_rename(path):
    """mkdirs. If path exists, rename it with timestamp and create a new one.
    Args:
        path (str): Folder path.
    """
    if os.path.exists(path):
        new_name = path + '_archived_' + get_time_str()
        print(f'Path already exists. Rename it to {new_name}', flush=True)
        os.rename(path, new_name)
    os.makedirs(path, exist_ok=True)


def get_root_logger(logger_name='Video_Process', log_level=logging.INFO, log_file=None, rank=0):

    logger = logging.getLogger(logger_name)
    # if the logger has been initialized, just return it
    if logger.hasHandlers():
        return logger

    format_str = '%(asctime)s %(levelname)s: %(message)s'
    logging.basicConfig(format=format_str, level=log_level)

    if rank != 0:
        logger.setLevel('ERROR')
    elif log_file is not None:
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

    return logger


def init_loggers(config, opts):
    log_file = os.path.join(config['path']['log'],
                        f"train_{opts.name}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='Video_Process', log_level=logging.INFO, log_file=log_file, rank=opts.global_rank)
    logger.info(get_env_info())
    logger.info(dict2str(config))

    return logger


def get_env_info():
    """Get environment information.
    Currently, only log the software version.
    """
    import torch
    import torchvision

    msg = ""

    msg += ('\nVersion Information: '
            f'\n\tPyTorch: {torch.__version__}'
            f'\n\tTorchVision: {torchvision.__version__}')
    return msg


def dict2str(opt, indent_level=1):
    """dict to string for printing options.
    Args:
        opt (dict): Option dict.
        indent_level (int): Indent level. Default: 1.
    Return:
        (str): Option string for printing.
    """
    msg = '\n'
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_level * 2) + k + ':['
            msg += dict2str(v, indent_level + 1)
            msg += ' ' * (indent_level * 2) + ']\n'
        else:
            msg += ' ' * (indent_level * 2) + k + ': ' + str(v) + '\n'
    return msg

## TODO: adaptive size while converting to videos [√]
def get_frames_input(path, width, height):

    img_list=os.listdir(path)
    img_list=sorted(img_list)
    img=[]
    for i in range(len(img_list)):
        temp = cv2.imread(os.path.join(path,img_list[i]))
        temp = cv2.resize(temp, (width,height), interpolation = cv2.INTER_AREA)
        img.append(temp)

    return img

def get_frames_output(path):

    img_list=os.listdir(path)
    img_list=sorted(img_list)
    img=[]
    for i in range(len(img_list)):
        temp = cv2.imread(os.path.join(path,img_list[i]))
        # temp = cv2.resize(temp, (640,360), interpolation = cv2.INTER_AREA)
        img.append(temp)

    height, width= img[0].shape[:2]

    return img, height, width

def frame_to_video(input_frame_url, restored_frame_url, save_place):

    output_frames, height, width = get_frames_output(restored_frame_url)
    input_frames = get_frames_input(input_frame_url, width, height)
    fourcc=cv2.VideoWriter_fourcc(*'MJPG')
    video=cv2.VideoWriter(save_place,fourcc,15,(width,height*2))

    for j in range(min(len(input_frames),len(output_frames))):
        temp=cv2.vconcat([input_frames[j],output_frames[j]])
        video.write(temp)

    video.release()