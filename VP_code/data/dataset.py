import random
import torch
from torch.utils import data as data
import os
import numpy as np
import cv2
import operator
from PIL import Image
from skimage.color import rgb2lab

import torchvision.transforms as transforms
from VP_code.utils.util import get_root_logger
from VP_code.utils.data_util import img2tensor, paired_random_crop, augment
from VP_code.data.Data_Degradation.util import degradation_video_list, degradation_video_list_2, degradation_video_list_3, degradation_video_list_4, degradation_video_list_simple_debug, degradation_video_colorization, transfer_1, transfer_2, degradation_video_colorization_v2, degradation_video_colorization_v3, degradation_video_colorization_v4
from VP_code.utils.LAB_util import to_mytensor, Normalize_LAB

def getfilelist(file_path):
    all_file = []
    for dir,folder,file in os.walk(file_path):
        for i in file:
            t = "%s/%s"%(dir,i)
            all_file.append(t)
    all_file = sorted(all_file)
    return all_file

def getfilelist_with_length(file_path):
    all_file = []
    for dir,folder,file in os.walk(file_path):
        for i in file:
            t = "%s/%s"%(dir,i)
            all_file.append((t,len(os.listdir(dir))))

    all_file.sort(key = operator.itemgetter(0))
    return all_file

def getfolderlist(file_path):

    all_folder = []
    for dir,folder,file in os.walk(file_path):
        if len(file)==0:
            continue
        rerank = sorted(file)
        t = "%s/%s"%(dir,rerank[0])
        if t.endswith('.avi'):
            continue
        all_folder.append((t,len(file)))
    
    all_folder.sort(key = operator.itemgetter(0))
    # all_folder = sorted(all_folder)
    return all_folder


def convert_to_L(img):

    frame_pil = transfer_1(img)
    frame_cv2 = transfer_2(frame_pil.convert("RGB"))

    return frame_cv2


def resize_256_short_side(img):
    width, height = img.size

    if width<height:
        new_height =  int (256 * height / width)
        new_width = 256
    else:
        new_width =  int (256 * width / height)
        new_height = 256
    
    return img.resize((new_width,new_height),resample=Image.BILINEAR)


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

# def getfolderlist_with_length(file_path):
#     all_folder = []
#     for dir,folder,file in os.walk(file_path):
#         for i in folder:
#             t = "%s/%s"%(dir,i)
#             all_folder.append((t,len(os.listdir(t))))
#     all_folder.sort(key = operator.itemgetter(0))
#     return all_folder


class Film_dataset_1(data.Dataset): ## 1 for REDS dataset

    def __init__(self, data_config):
        super(Film_dataset_1, self).__init__()
        
        self.data_config = data_config
        
        self.scale = data_config['scale']
        self.gt_root, self.lq_root = data_config['dataroot_gt'], data_config['dataroot_lq']
        self.is_train = data_config.get('is_train', False)
        
        ## TODO: dynamic frame num for different video clips
        self.num_frame = data_config['num_frame']
        self.num_half_frames = data_config['num_frame'] // 2

        if self.is_train:

            self.lq_frames = getfilelist_with_length(self.lq_root)
            self.gt_frames = getfilelist_with_length(self.gt_root)
        
        else:
            ## Now: Append the first frame name, then load all frames based on the clip length
            self.lq_frames = getfolderlist(self.lq_root)
            self.gt_frames = getfolderlist(self.gt_root)
            # self.lq_frames = []
            # self.gt_frames = []
            # for i in range(len(self.lq_folders))
            #     val_frame_list_this = sorted(os.listdir(self.lq_folders[i]))
            #     first_frame_name = val_frame_list_this[0]
            #     clip_length = len(val_frame_list_this)
            #     self.lq_frames.append((os.path.join(self.lq_folders[i],f'{first_frame_name:08d}.png'),clip_length))
            #     self.gt_frames.append((os.path.join(self.gt_folders[i],f'{first_frame_name:08d}.png'),clip_length))

        # temporal augmentation configs
        self.interval_list = data_config['interval_list']
        self.random_reverse = data_config['random_reverse']
        interval_str = ','.join(str(x) for x in data_config['interval_list'])
        logger = get_root_logger()
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'Random reverse is {self.random_reverse}.')

    def __getitem__(self, index):

        gt_size = self.data_config.get('gt_size', None)
        gt_size_w = gt_size[0]
        gt_size_h = gt_size[1]

        key = self.gt_frames[index][0]
        current_len = self.gt_frames[index][1]

        ## Fetch the parent directory of clip name
        current_gt_root = os.path.dirname(os.path.dirname(self.gt_frames[index][0]))
        current_lq_root = os.path.dirname(os.path.dirname(self.lq_frames[index][0]))

        clip_name, frame_name = key.split('/')[-2:]  # key example: 000/00000000
        key = clip_name + "/" + frame_name
        center_frame_idx = int(frame_name[:-4])

        new_clip_sequence = sorted(os.listdir(os.path.join(current_gt_root, clip_name)))
        if self.is_train:
            # determine the frameing frames
            interval = random.choice(self.interval_list)

            # ensure not exceeding the borders
            start_frame_idx = center_frame_idx - self.num_half_frames * interval
            end_frame_idx = start_frame_idx + (self.num_frame - 1) * interval

            # each clip has 100 frames starting from 0 to 99. TODO: if the training clip is not 100 frames [√]
            # Training start frames should be 0
            while (start_frame_idx < 0) or (end_frame_idx > current_len-1):
                center_frame_idx = random.randint(self.num_half_frames * interval, current_len - self.num_half_frames *interval)
                start_frame_idx = (center_frame_idx - self.num_half_frames * interval)
                end_frame_idx = start_frame_idx + (self.num_frame - 1) * interval
            
            # frame_name = f'{center_frame_idx:08d}'
            frame_list = list(range(start_frame_idx, end_frame_idx + 1, interval))
            # Sample number should equal to the numer we set
            assert len(frame_list) == self.num_frame, (f'Wrong length of frame list: {len(frame_list)}')
        else:

            frame_list = list(range(center_frame_idx, center_frame_idx + current_len))
            # Sample number should equal to the all frames number in on folder
            assert len(frame_list) == current_len, (f'Wrong length of frame list: {len(frame_list)}')

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            frame_list.reverse()


        # get the GT frame (as the center frame)
        img_gts = []
        img_lqs = []
        for tmp_id, frame in enumerate(frame_list):

            # img_gt_path = os.path.join(current_gt_root, clip_name, f'{frame:05d}.png')
            img_gt_path = os.path.join(current_gt_root, clip_name, new_clip_sequence[tmp_id])
            img_gt = cv2.imread(img_gt_path)
            img_gt = img_gt.astype(np.float32) / 255.
            img_gts.append(img_gt)

            if not self.is_train:
                img_lq_path = os.path.join(current_lq_root, clip_name, new_clip_sequence[tmp_id])
                img_lq = cv2.imread(img_lq_path)
                img_lq = img_lq.astype(np.float32) / 255.
                img_lqs.append(img_lq)

        if self.is_train:
            img_lqs = img_gts
            img_gts, img_lqs = paired_random_crop(img_gts, img_lqs, gt_size_w, gt_size_h, self.scale, clip_name)
            img_lqs, img_gts = degradation_video_list_4(img_gts, texture_url=self.data_config['texture_template'])
        else:
            for i in range(len(img_gts)):
                # img_gts[i] = cv2.resize(img_gts[i], (gt_size_w, gt_size_h), interpolation = cv2.INTER_AREA)
                # img_lqs[i] = cv2.resize(img_lqs[i], (gt_size_w, gt_size_h), interpolation = cv2.INTER_AREA)
                img_gts[i] = resize_368_short_side(img_gts[i])
                img_lqs[i] = resize_368_short_side(img_lqs[i])
                if self.data_config['name']=='colorization':
                    img_gts[i] = convert_to_L(img_gts[i])
                    img_lqs[i] = convert_to_L(img_lqs[i])

        # augmentation - flip, rotate
        img_lqs.extend(img_gts)
        if self.is_train:
            img_lqs = augment(img_lqs, self.data_config['use_flip'], self.data_config['use_rot'])

        img_results = img2tensor(img_lqs) ## List of tensor

        if self.data_config['normalizing']:
            transform_normalize=transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
            for i in range(len(img_results)):
                img_results[i]=transform_normalize(img_results[i])

        if self.is_train:
            img_lqs = torch.stack(img_results[:self.num_frame], dim=0)
            img_gts = torch.stack(img_results[self.num_frame:], dim=0)
        else:
            img_lqs = torch.stack(img_results[:current_len], dim=0)
            img_gts = torch.stack(img_results[current_len:], dim=0)           

        # img_lqs: (t, c, h, w)
        # img_gt: (t, c, h, w)
        # key: str
        return {'lq': img_lqs, 'gt': img_gts, 'key': key, 'frame_list': frame_list, 'video_name': os.path.basename(current_lq_root), 'name_list': new_clip_sequence}

    def __len__(self):
        return len(self.lq_frames)


class Film_dataset_2(data.Dataset): ## 2 for Vimeo dataset

    def __init__(self, data_config):
        super(Film_dataset_2, self).__init__()
        
        self.data_config = data_config
        
        self.scale = data_config['scale']
        self.gt_root, self.lq_root = data_config['dataroot_gt'], data_config['dataroot_lq']
        self.is_train = data_config.get('is_train', False)
        
        ## TODO: dynamic frame num for different video clips
        self.num_frame = data_config['num_frame']
        self.num_half_frames = data_config['num_frame'] // 2


        self.lq_frames = getfolderlist(self.lq_root)
        self.gt_frames = getfolderlist(self.gt_root)

        # temporal augmentation configs
        self.interval_list = data_config['interval_list']
        self.random_reverse = data_config['random_reverse']
        interval_str = ','.join(str(x) for x in data_config['interval_list'])
        logger = get_root_logger()
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'Random reverse is {self.random_reverse}.')

    def __getitem__(self, index):

        gt_size = self.data_config.get('gt_size', None)
        gt_size_w = gt_size[0]
        gt_size_h = gt_size[1]

        key = self.gt_frames[index][0]
        current_len = self.gt_frames[index][1]

        ## Fetch the parent directory of clip name
        current_gt_root = os.path.dirname(os.path.dirname(self.gt_frames[index][0]))
        current_lq_root = os.path.dirname(os.path.dirname(self.lq_frames[index][0]))

        clip_name, frame_name = key.split('/')[-2:]  # key example: 000/00000000
        key = clip_name + "/" + frame_name
        center_frame_idx = 1


        frame_list = list(range(center_frame_idx, center_frame_idx + current_len))
        # Sample number should equal to the all frames number in on folder
        assert len(frame_list) == current_len, (f'Wrong length of frame list: {len(frame_list)}')

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            frame_list.reverse()


        # get the GT frame (as the center frame)
        img_gts = []
        img_lqs = []
        for frame in frame_list:

            img_gt_path = os.path.join(current_gt_root, clip_name, 'im%d.png'%(frame))
            img_gt = cv2.imread(img_gt_path)
            img_gt = img_gt.astype(np.float32) / 255.
            img_gts.append(img_gt)

            if not self.is_train:
                img_lq_path = os.path.join(current_lq_root, clip_name, f'{frame:08d}.png')
                img_lq = cv2.imread(img_lq_path)
                img_lq = img_lq.astype(np.float32) / 255.
                img_lqs.append(img_lq)

        if self.is_train:
            img_lqs = img_gts
            img_gts, img_lqs = paired_random_crop(img_gts, img_lqs, gt_size_w, gt_size_h, self.scale, clip_name)
            img_lqs, img_gts = degradation_video_list_3(img_gts, texture_url=self.data_config['texture_template'])
        else:
            for i in range(len(img_gts)):
                img_gts[i] = cv2.resize(img_gts[i], (gt_size_w, gt_size_h), interpolation = cv2.INTER_AREA)
                img_lqs[i] = cv2.resize(img_lqs[i], (gt_size_w, gt_size_h), interpolation = cv2.INTER_AREA)

        # augmentation - flip, rotate
        img_lqs.extend(img_gts)
        if self.is_train:
            img_lqs = augment(img_lqs, self.data_config['use_flip'], self.data_config['use_rot'])

        img_results = img2tensor(img_lqs) ## List of tensor

        if self.data_config['normalizing']:
            transform_normalize=transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
            for i in range(len(img_results)):
                img_results[i]=transform_normalize(img_results[i])

        if self.is_train:
            img_lqs = torch.stack(img_results[:self.num_frame], dim=0)
            img_gts = torch.stack(img_results[self.num_frame:], dim=0)
        else:
            img_lqs = torch.stack(img_results[:current_len], dim=0)
            img_gts = torch.stack(img_results[current_len:], dim=0)           

        # img_lqs: (t, c, h, w)
        # img_gt: (t, c, h, w)
        # key: str
        return {'lq': img_lqs, 'gt': img_gts, 'key': key, 'frame_list': frame_list, 'video_name': os.path.basename(current_lq_root)}

    def __len__(self):
        return len(self.lq_frames)

class Film_dataset_3(data.Dataset): ## 3 for Vimeo dataset + Colorization

    def __init__(self, data_config):
        super(Film_dataset_3, self).__init__()
        
        self.data_config = data_config
        
        self.scale = data_config['scale']
        self.gt_root, self.lq_root = data_config['dataroot_gt'], data_config['dataroot_lq']
        self.is_train = data_config.get('is_train', False)
        
        ## TODO: dynamic frame num for different video clips
        self.num_frame = data_config['num_frame']
        self.num_half_frames = data_config['num_frame'] // 2


        self.lq_frames = getfolderlist(self.lq_root)
        self.gt_frames = getfolderlist(self.gt_root)

        # temporal augmentation configs
        self.interval_list = data_config['interval_list']
        self.random_reverse = data_config['random_reverse']
        interval_str = ','.join(str(x) for x in data_config['interval_list'])
        logger = get_root_logger()
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'Random reverse is {self.random_reverse}.')

    def __getitem__(self, index):

        gt_size = self.data_config.get('gt_size', None)
        gt_size_w = gt_size[0]
        gt_size_h = gt_size[1]

        key = self.gt_frames[index][0]
        current_len = self.gt_frames[index][1]

        ## Fetch the parent directory of clip name
        current_gt_root = os.path.dirname(os.path.dirname(self.gt_frames[index][0]))
        current_lq_root = os.path.dirname(os.path.dirname(self.lq_frames[index][0]))

        clip_name, frame_name = key.split('/')[-2:]  # key example: 000/00000000
        key = clip_name + "/" + frame_name
        center_frame_idx = 1


        frame_list = list(range(center_frame_idx, center_frame_idx + current_len))
        # Sample number should equal to the all frames number in on folder
        assert len(frame_list) == current_len, (f'Wrong length of frame list: {len(frame_list)}')

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            frame_list.reverse()


        # get the GT frame (as the center frame)
        img_gts = []
        img_lqs = []
        for frame in frame_list:

            img_gt_path = os.path.join(current_gt_root, clip_name, 'im%d.png'%(frame))
            img_gt = cv2.imread(img_gt_path)
            img_gt = img_gt.astype(np.float32) / 255.
            img_gts.append(img_gt)

            if not self.is_train:
                img_lq_path = os.path.join(current_lq_root, clip_name, f'{frame:08d}.png')
                img_lq = cv2.imread(img_lq_path)
                img_lq = img_lq.astype(np.float32) / 255.
                img_lqs.append(img_lq)

        if self.is_train:
            img_lqs = img_gts
            img_gts, img_lqs = paired_random_crop(img_gts, img_lqs, gt_size_w, gt_size_h, self.scale, clip_name)
            img_lqs, img_gts = degradation_video_colorization(img_gts)
        else:
            for i in range(len(img_gts)):
                img_gts[i] = cv2.resize(img_gts[i], (gt_size_w, gt_size_h), interpolation = cv2.INTER_AREA)
                img_lqs[i] = cv2.resize(img_lqs[i], (gt_size_w, gt_size_h), interpolation = cv2.INTER_AREA)

        # augmentation - flip, rotate
        img_lqs.extend(img_gts)
        if self.is_train:
            img_lqs = augment(img_lqs, self.data_config['use_flip'], self.data_config['use_rot'])

        img_results = img2tensor(img_lqs) ## List of tensor

        if self.data_config['normalizing']:
            transform_normalize=transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
            for i in range(len(img_results)):
                img_results[i]=transform_normalize(img_results[i])

        if self.is_train:
            img_lqs = torch.stack(img_results[:self.num_frame], dim=0)
            img_gts = torch.stack(img_results[self.num_frame:], dim=0)
        else:
            img_lqs = torch.stack(img_results[:current_len], dim=0)
            img_gts = torch.stack(img_results[current_len:], dim=0)           

        # img_lqs: (t, c, h, w)
        # img_gt: (t, c, h, w)
        # key: str
        return {'lq': img_lqs, 'gt': img_gts, 'key': key, 'frame_list': frame_list, 'video_name': os.path.basename(current_lq_root)}

    def __len__(self):
        return len(self.lq_frames)


class Film_dataset_4(data.Dataset): ## 4 for REDS dataset + resize by 2

    def __init__(self, data_config):
        super(Film_dataset_4, self).__init__()
        
        self.data_config = data_config
        
        self.scale = data_config['scale']
        self.gt_root, self.lq_root = data_config['dataroot_gt'], data_config['dataroot_lq']
        self.is_train = data_config.get('is_train', False)
        
        ## TODO: dynamic frame num for different video clips
        self.num_frame = data_config['num_frame']
        self.num_half_frames = data_config['num_frame'] // 2

        if self.is_train:

            self.lq_frames = getfilelist_with_length(self.lq_root)
            self.gt_frames = getfilelist_with_length(self.gt_root)
        
        else:
            ## Now: Append the first frame name, then load all frames based on the clip length
            self.lq_frames = getfolderlist(self.lq_root)
            self.gt_frames = getfolderlist(self.gt_root)
            # self.lq_frames = []
            # self.gt_frames = []
            # for i in range(len(self.lq_folders))
            #     val_frame_list_this = sorted(os.listdir(self.lq_folders[i]))
            #     first_frame_name = val_frame_list_this[0]
            #     clip_length = len(val_frame_list_this)
            #     self.lq_frames.append((os.path.join(self.lq_folders[i],f'{first_frame_name:08d}.png'),clip_length))
            #     self.gt_frames.append((os.path.join(self.gt_folders[i],f'{first_frame_name:08d}.png'),clip_length))

        # temporal augmentation configs
        self.interval_list = data_config['interval_list']
        self.random_reverse = data_config['random_reverse']
        interval_str = ','.join(str(x) for x in data_config['interval_list'])
        logger = get_root_logger()
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'Random reverse is {self.random_reverse}.')

    def __getitem__(self, index):

        gt_size = self.data_config.get('gt_size', None)
        gt_size_w = gt_size[0]
        gt_size_h = gt_size[1]

        key = self.gt_frames[index][0]
        current_len = self.gt_frames[index][1]

        ## Fetch the parent directory of clip name
        current_gt_root = os.path.dirname(os.path.dirname(self.gt_frames[index][0]))
        current_lq_root = os.path.dirname(os.path.dirname(self.lq_frames[index][0]))

        clip_name, frame_name = key.split('/')[-2:]  # key example: 000/00000000
        key = clip_name + "/" + frame_name
        center_frame_idx = int(frame_name[:-4])

        if self.is_train:
            # determine the frameing frames
            interval = random.choice(self.interval_list)

            # ensure not exceeding the borders
            start_frame_idx = center_frame_idx - self.num_half_frames * interval
            end_frame_idx = start_frame_idx + (self.num_frame - 1) * interval

            # each clip has 100 frames starting from 0 to 99. TODO: if the training clip is not 100 frames [√]
            # Training start frames should be 0
            while (start_frame_idx < 0) or (end_frame_idx > current_len-1):
                center_frame_idx = random.randint(self.num_half_frames * interval, current_len - self.num_half_frames *interval)
                start_frame_idx = (center_frame_idx - self.num_half_frames * interval)
                end_frame_idx = start_frame_idx + (self.num_frame - 1) * interval
            
            # frame_name = f'{center_frame_idx:08d}'
            frame_list = list(range(start_frame_idx, end_frame_idx + 1, interval))
            # Sample number should equal to the numer we set
            assert len(frame_list) == self.num_frame, (f'Wrong length of frame list: {len(frame_list)}')
        else:

            frame_list = list(range(center_frame_idx, center_frame_idx + current_len))
            # Sample number should equal to the all frames number in on folder
            assert len(frame_list) == current_len, (f'Wrong length of frame list: {len(frame_list)}')

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            frame_list.reverse()


        # get the GT frame (as the center frame)
        img_gts = []
        img_lqs = []
        for frame in frame_list:

            img_gt_path = os.path.join(current_gt_root, clip_name, f'{frame:08d}.png')
            img_gt = cv2.imread(img_gt_path)
            img_gt = cv2.resize(img_gt, (640, 360), interpolation = cv2.INTER_LANCZOS4)
            img_gt = img_gt.astype(np.float32) / 255.
            img_gts.append(img_gt)

            if not self.is_train:
                img_lq_path = os.path.join(current_lq_root, clip_name, f'{frame:08d}.png')
                img_lq = cv2.imread(img_lq_path)
                img_lq = img_lq.astype(np.float32) / 255.
                img_lqs.append(img_lq)

        if self.is_train:
            img_lqs = img_gts
            img_gts, img_lqs = paired_random_crop(img_gts, img_lqs, gt_size_w, gt_size_h, self.scale, clip_name)
            img_lqs, img_gts = degradation_video_list_4(img_gts, texture_url=self.data_config['texture_template'])
        else:
            for i in range(len(img_gts)):
                img_gts[i] = cv2.resize(img_gts[i], (gt_size_w, gt_size_h), interpolation = cv2.INTER_AREA)
                img_lqs[i] = cv2.resize(img_lqs[i], (gt_size_w, gt_size_h), interpolation = cv2.INTER_AREA)

        # augmentation - flip, rotate
        img_lqs.extend(img_gts)
        if self.is_train:
            img_lqs = augment(img_lqs, self.data_config['use_flip'], self.data_config['use_rot'])

        img_results = img2tensor(img_lqs) ## List of tensor

        if self.data_config['normalizing']:
            transform_normalize=transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
            for i in range(len(img_results)):
                img_results[i]=transform_normalize(img_results[i])

        if self.is_train:
            img_lqs = torch.stack(img_results[:self.num_frame], dim=0)
            img_gts = torch.stack(img_results[self.num_frame:], dim=0)
        else:
            img_lqs = torch.stack(img_results[:current_len], dim=0)
            img_gts = torch.stack(img_results[current_len:], dim=0)           

        # img_lqs: (t, c, h, w)
        # img_gt: (t, c, h, w)
        # key: str
        return {'lq': img_lqs, 'gt': img_gts, 'key': key, 'frame_list': frame_list, 'video_name': os.path.basename(current_lq_root)}

    def __len__(self):
        return len(self.lq_frames)


class Film_dataset_5(data.Dataset): ## 3 for Vimeo dataset + Colorization + Convert_to_LAB

    def __init__(self, data_config):
        super(Film_dataset_5, self).__init__()
        
        self.data_config = data_config
        
        self.scale = data_config['scale']
        self.gt_root, self.lq_root = data_config['dataroot_gt'], data_config['dataroot_lq']
        self.is_train = data_config.get('is_train', False)
        
        ## TODO: dynamic frame num for different video clips
        self.num_frame = data_config['num_frame']
        self.num_half_frames = data_config['num_frame'] // 2


        self.lq_frames = getfolderlist(self.lq_root)
        self.gt_frames = getfolderlist(self.gt_root)

        # temporal augmentation configs
        self.interval_list = data_config['interval_list']
        self.random_reverse = data_config['random_reverse']
        interval_str = ','.join(str(x) for x in data_config['interval_list'])
        logger = get_root_logger()
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'Random reverse is {self.random_reverse}.')

    def __getitem__(self, index):

        gt_size = self.data_config.get('gt_size', None)
        gt_size_w = gt_size[0]
        gt_size_h = gt_size[1]

        key = self.gt_frames[index][0]
        current_len = self.gt_frames[index][1]

        ## Fetch the parent directory of clip name
        current_gt_root = os.path.dirname(os.path.dirname(self.gt_frames[index][0]))
        current_lq_root = os.path.dirname(os.path.dirname(self.lq_frames[index][0]))

        clip_name, frame_name = key.split('/')[-2:]  # key example: 000/00000000
        key = clip_name + "/" + frame_name
        
        new_clip_sequence = sorted(os.listdir(os.path.join(current_gt_root, clip_name)))
        
        if self.is_train:
            center_frame_idx = 1
        else:
            center_frame_idx = int(frame_name[:-4])

        frame_list = list(range(center_frame_idx, center_frame_idx + current_len))
        # Sample number should equal to the all frames number in on folder
        assert len(frame_list) == current_len, (f'Wrong length of frame list: {len(frame_list)}')

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            frame_list.reverse()


        # get the GT frame (as the center frame)
        img_gts = []
        img_lqs = []
        for tmp_id,frame in enumerate(frame_list):

            if self.is_train:
                img_gt_path = os.path.join(current_gt_root, clip_name, 'im%d.png'%(frame))
            else:
                img_gt_path = os.path.join(current_gt_root, clip_name, new_clip_sequence[tmp_id])

            img_gt=rgb2lab(Image.open(img_gt_path).convert("RGB"))
            img_gts.append(np.array(img_gt))
            # img_gt = cv2.imread(img_gt_path)
            # img_gt = img_gt.astype(np.float32) / 255.
            # img_gts.append(img_gt)

            if not self.is_train:
                img_lq_path = os.path.join(current_lq_root, clip_name, new_clip_sequence[tmp_id])

                img_lq=rgb2lab(Image.open(img_lq_path).convert("RGB"))
                img_lqs.append(np.array(img_lq))
                # img_lq = cv2.imread(img_lq_path)
                # img_lq = img_lq.astype(np.float32) / 255.
                # img_lqs.append(img_lq)

        if self.is_train:
            img_lqs = img_gts
            img_gts, img_lqs = paired_random_crop(img_gts, img_lqs, gt_size_w, gt_size_h, self.scale, clip_name)
            img_lqs, img_gts = degradation_video_colorization_v2(img_gts) ## LAB now
        else:
            # for i in range(len(img_gts)): ##TODO: inference stage: set the non-reference AB channel to 0 [√]
            #     img_gts[i] = cv2.resize(img_gts[i], (gt_size_w, gt_size_h), interpolation = cv2.INTER_AREA)
            #     img_lqs[i] = cv2.resize(img_lqs[i], (gt_size_w, gt_size_h), interpolation = cv2.INTER_AREA)
            img_gts = img_gts
            img_lqs = img_lqs 

        # augmentation - flip, rotate
        img_lqs.extend(img_gts)
        if self.is_train:
            img_lqs = augment(img_lqs, self.data_config['use_flip'], self.data_config['use_rot'])

        img_results = []
        for x in img_lqs:
            img_results.append(to_mytensor(x))

        if self.data_config['normalizing']:
            # transform_normalize=transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
            for i in range(len(img_results)): 
                img_results[i]=Normalize_LAB(img_results[i])

        if self.is_train:
            img_lqs = torch.stack(img_results[:self.num_frame], dim=0)
            img_gts = torch.stack(img_results[self.num_frame:], dim=0)
        else:
            img_lqs = torch.stack(img_results[:current_len], dim=0)
            img_gts = torch.stack(img_results[current_len:], dim=0)           

        # img_lqs: (t, c, h, w)
        # img_gt: (t, c, h, w)
        # key: str
        return {'lq': img_lqs, 'gt': img_gts, 'key': key, 'frame_list': frame_list, 'video_name': os.path.basename(current_lq_root)}

    def __len__(self):
        return len(self.lq_frames)



class Film_dataset_6(data.Dataset): ## 6 for DAVIS&YoutubeVOS dataset + Colorization + Convert_to_LAB

    def __init__(self, data_config):
        super(Film_dataset_6, self).__init__()
        
        self.data_config = data_config
        
        self.scale = data_config['scale']
        self.gt_root, self.lq_root = data_config['dataroot_gt'], data_config['dataroot_lq']
        self.is_train = data_config.get('is_train', False)
        

        self.num_frame = data_config['num_frame']
        self.num_half_frames = data_config['num_frame'] // 2


        self.lq_frames = getfolderlist(self.lq_root)
        self.gt_frames = getfolderlist(self.gt_root)

        # temporal augmentation configs
        self.interval_list = data_config['interval_list']
        self.random_reverse = data_config['random_reverse']
        interval_str = ','.join(str(x) for x in data_config['interval_list'])
        logger = get_root_logger()
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'Random reverse is {self.random_reverse}.')

    def __getitem__(self, index):

        gt_size = self.data_config.get('gt_size', None)
        gt_size_w = gt_size[0]
        gt_size_h = gt_size[1]

        key = self.gt_frames[index][0]
        current_len = self.gt_frames[index][1]


        ## Fetch the parent directory of clip name
        current_gt_root = os.path.dirname(os.path.dirname(self.gt_frames[index][0]))
        current_lq_root = os.path.dirname(os.path.dirname(self.lq_frames[index][0]))

        clip_name, frame_name = key.split('/')[-2:]  # key example: 000/00000000
        key = clip_name + "/" + frame_name
        start_id = int(frame_name[:-4])

        if self.is_train:
            # determine the frameing frames
            interval = random.choice(self.interval_list)

            center_frame_idx = random.randint(self.num_half_frames * interval, current_len - self.num_half_frames *interval - 1)
            start_frame_idx = (center_frame_idx - self.num_half_frames * interval)
            end_frame_idx = start_frame_idx + (self.num_frame - 1) * interval
            # frame_name = f'{center_frame_idx:08d}'
            frame_list = list(range(start_frame_idx, end_frame_idx + 1, interval))
            # print(frame_list)
            # Sample number should equal to the numer we set
            new_clip_sequence = sorted(os.listdir(os.path.join(current_gt_root, clip_name)))
            assert len(frame_list) == self.num_frame, (f'Wrong length of frame list: {len(frame_list)}')

        else:
            frame_list = list(range(start_id, start_id + current_len))
            # Sample number should equal to the all frames number in on folder
            assert len(frame_list) == current_len, (f'Wrong length of frame list: {len(frame_list)}')

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            frame_list.reverse()


        # get the GT frame (as the center frame)
        img_gts = []
        img_lqs = []
        for frame in frame_list:

            if self.is_train:
                # img_gt_path = os.path.join(current_gt_root, clip_name, f'{frame:05d}.jpg') ## Adaptive for DAVIS and YoutubeVOS
                img_gt_path = os.path.join(current_gt_root, clip_name, new_clip_sequence[frame])
                current_frame = Image.open(img_gt_path).convert("RGB")
                current_frame = resize_256_short_side(current_frame)
                img_gt=rgb2lab(current_frame)
                img_gts.append(np.array(img_gt))
            else:
                img_gt_path = os.path.join(current_gt_root, clip_name, f'{frame:08d}.png')
                img_gt=rgb2lab(Image.open(img_gt_path).convert("RGB"))
                img_gts.append(np.array(img_gt))
            # img_gt = cv2.imread(img_gt_path)
            # img_gt = img_gt.astype(np.float32) / 255.
            # img_gts.append(img_gt)

            if not self.is_train:
                img_lq_path = os.path.join(current_lq_root, clip_name, f'{frame:08d}.png')

                img_lq=rgb2lab(Image.open(img_lq_path).convert("RGB"))
                img_lqs.append(np.array(img_lq))
                # img_lq = cv2.imread(img_lq_path)
                # img_lq = img_lq.astype(np.float32) / 255.
                # img_lqs.append(img_lq)

        if self.is_train:
            img_lqs = img_gts
            img_gts, img_lqs = paired_random_crop(img_gts, img_lqs, gt_size_w, gt_size_h, self.scale, clip_name)
            img_lqs, img_gts = degradation_video_colorization_v2(img_gts) ## LAB now
        else:
            for i in range(len(img_gts)): ##TODO: inference stage: set the non-reference AB channel to 0 [√]
                img_gts[i] = cv2.resize(img_gts[i], (gt_size_w, gt_size_h), interpolation = cv2.INTER_AREA)
                img_lqs[i] = cv2.resize(img_lqs[i], (gt_size_w, gt_size_h), interpolation = cv2.INTER_AREA)

        # augmentation - flip, rotate
        img_lqs.extend(img_gts)
        if self.is_train:
            img_lqs = augment(img_lqs, self.data_config['use_flip'], self.data_config['use_rot'])

        img_results = []
        for x in img_lqs:
            img_results.append(to_mytensor(x))

        if self.data_config['normalizing']:
            # transform_normalize=transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
            for i in range(len(img_results)): 
                img_results[i]=Normalize_LAB(img_results[i])

        if self.is_train:
            img_lqs = torch.stack(img_results[:self.num_frame], dim=0)
            img_gts = torch.stack(img_results[self.num_frame:], dim=0)
        else:
            img_lqs = torch.stack(img_results[:current_len], dim=0)
            img_gts = torch.stack(img_results[current_len:], dim=0)           

        # img_lqs: (t, c, h, w)
        # img_gt: (t, c, h, w)
        # key: str
        return {'lq': img_lqs, 'gt': img_gts, 'key': key, 'frame_list': frame_list, 'video_name': os.path.basename(current_lq_root), 'clip_name': clip_name}

    def __len__(self):
        return len(self.lq_frames)

class Film_dataset_7(data.Dataset): ## 7 for DAVIS&YoutubeVOS dataset + Colorization + Convert_to_LAB, add extra long-term references (the final element)

    def __init__(self, data_config):
        super(Film_dataset_7, self).__init__()
        
        self.data_config = data_config
        
        self.scale = data_config['scale']
        self.gt_root, self.lq_root = data_config['dataroot_gt'], data_config['dataroot_lq']
        self.is_train = data_config.get('is_train', False)
        

        self.num_frame = data_config['num_frame']
        self.num_half_frames = data_config['num_frame'] // 2


        self.lq_frames = getfolderlist(self.lq_root)
        self.gt_frames = getfolderlist(self.gt_root)

        # temporal augmentation configs
        self.interval_list = data_config['interval_list']
        self.random_reverse = data_config['random_reverse']
        interval_str = ','.join(str(x) for x in data_config['interval_list'])
        logger = get_root_logger()
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'Random reverse is {self.random_reverse}.')

    def __getitem__(self, index):

        gt_size = self.data_config.get('gt_size', None)
        gt_size_w = gt_size[0]
        gt_size_h = gt_size[1]

        key = self.gt_frames[index][0]
        current_len = self.gt_frames[index][1]


        ## Fetch the parent directory of clip name
        current_gt_root = os.path.dirname(os.path.dirname(self.gt_frames[index][0]))
        current_lq_root = os.path.dirname(os.path.dirname(self.lq_frames[index][0]))

        clip_name, frame_name = key.split('/')[-2:]  # key example: 000/00000000
        key = clip_name + "/" + frame_name
        start_id = int(frame_name[:-4])

        if self.is_train:
            # determine the frameing frames
            interval = random.choice(self.interval_list)

            center_frame_idx = random.randint(self.num_half_frames * interval, current_len - self.num_half_frames *interval - 1)
            start_frame_idx = (center_frame_idx - self.num_half_frames * interval)
            end_frame_idx = start_frame_idx + (self.num_frame - 1) * interval
            # frame_name = f'{center_frame_idx:08d}'
            frame_list = list(range(start_frame_idx, end_frame_idx + 1, interval))
            # print(frame_list)
            # Sample number should equal to the numer we set
            new_clip_sequence = sorted(os.listdir(os.path.join(current_gt_root, clip_name)))
            assert len(frame_list) == self.num_frame, (f'Wrong length of frame list: {len(frame_list)}')


        else:
            frame_list = list(range(start_id, start_id + current_len))
            # Sample number should equal to the all frames number in on folder
            assert len(frame_list) == current_len, (f'Wrong length of frame list: {len(frame_list)}')

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            frame_list.reverse()

        if self.is_train:
            # Random for the reference frame
            frame_list.append(random.randint(0,current_len-1))            


        # get the GT frame (as the center frame)
        img_gts = []
        img_lqs = []
        for frame in frame_list:

            if self.is_train:
                # img_gt_path = os.path.join(current_gt_root, clip_name, f'{frame:05d}.jpg') ## Adaptive for DAVIS and YoutubeVOS
                img_gt_path = os.path.join(current_gt_root, clip_name, new_clip_sequence[frame])
                current_frame = Image.open(img_gt_path).convert("RGB")
                current_frame = resize_256_short_side(current_frame)
                img_gt=rgb2lab(current_frame)
                img_gts.append(np.array(img_gt))
            else:
                img_gt_path = os.path.join(current_gt_root, clip_name, f'{frame:08d}.png')
                img_gt=rgb2lab(Image.open(img_gt_path).convert("RGB"))
                img_gts.append(np.array(img_gt))
            # img_gt = cv2.imread(img_gt_path)
            # img_gt = img_gt.astype(np.float32) / 255.
            # img_gts.append(img_gt)

            if not self.is_train:
                img_lq_path = os.path.join(current_lq_root, clip_name, f'{frame:08d}.png')

                img_lq=rgb2lab(Image.open(img_lq_path).convert("RGB"))
                img_lqs.append(np.array(img_lq))
                # img_lq = cv2.imread(img_lq_path)
                # img_lq = img_lq.astype(np.float32) / 255.
                # img_lqs.append(img_lq)

        if self.is_train:
            img_lqs = img_gts
            img_gts, img_lqs = paired_random_crop(img_gts, img_lqs, gt_size_w, gt_size_h, self.scale, clip_name)
            img_lqs, img_gts = degradation_video_colorization_v3(img_gts) ## LAB now
        else:
            for i in range(len(img_gts)): ##TODO: inference stage: set the non-reference AB channel to 0 [√]
                img_gts[i] = cv2.resize(img_gts[i], (gt_size_w, gt_size_h), interpolation = cv2.INTER_AREA)
                img_lqs[i] = cv2.resize(img_lqs[i], (gt_size_w, gt_size_h), interpolation = cv2.INTER_AREA)

        # augmentation - flip, rotate
        img_lqs.extend(img_gts)
        if self.is_train:
            img_lqs = augment(img_lqs, self.data_config['use_flip'], self.data_config['use_rot'])

        img_results = []
        for x in img_lqs:
            img_results.append(to_mytensor(x))

        if self.data_config['normalizing']:
            # transform_normalize=transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
            for i in range(len(img_results)): 
                img_results[i]=Normalize_LAB(img_results[i])

        if self.is_train:
            img_lqs = torch.stack(img_results[:self.num_frame+1], dim=0)
            img_gts = torch.stack(img_results[self.num_frame+1:], dim=0)
        else:
            img_lqs = torch.stack(img_results[:current_len], dim=0)
            img_gts = torch.stack(img_results[current_len:], dim=0)           

        # img_lqs: (t, c, h, w)
        # img_gt: (t, c, h, w)
        # key: str
        return {'lq': img_lqs, 'gt': img_gts, 'key': key, 'frame_list': frame_list, 'video_name': os.path.basename(current_lq_root), 'clip_name': clip_name}

    def __len__(self):
        return len(self.lq_frames)



class Film_dataset_8(data.Dataset): ## 7 for DAVIS&YoutubeVOS dataset + Colorization + Convert_to_LAB, add extra long-term references (the final element), degradation_video_colorization_v4

    def __init__(self, data_config):
        super(Film_dataset_8, self).__init__()
        
        self.data_config = data_config
        
        self.scale = data_config['scale']
        self.gt_root, self.lq_root = data_config['dataroot_gt'], data_config['dataroot_lq']
        self.is_train = data_config.get('is_train', False)
        

        self.num_frame = data_config['num_frame']
        self.num_half_frames = data_config['num_frame'] // 2


        self.lq_frames = getfolderlist(self.lq_root)
        self.gt_frames = getfolderlist(self.gt_root)

        # temporal augmentation configs
        self.interval_list = data_config['interval_list']
        self.random_reverse = data_config['random_reverse']
        interval_str = ','.join(str(x) for x in data_config['interval_list'])
        logger = get_root_logger()
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'Random reverse is {self.random_reverse}.')

    def __getitem__(self, index):

        gt_size = self.data_config.get('gt_size', None)
        gt_size_w = gt_size[0]
        gt_size_h = gt_size[1]

        key = self.gt_frames[index][0]
        current_len = self.gt_frames[index][1]


        ## Fetch the parent directory of clip name
        current_gt_root = os.path.dirname(os.path.dirname(self.gt_frames[index][0]))
        current_lq_root = os.path.dirname(os.path.dirname(self.lq_frames[index][0]))

        clip_name, frame_name = key.split('/')[-2:]  # key example: 000/00000000
        key = clip_name + "/" + frame_name
        start_id = int(frame_name[:-4])

        if self.is_train:
            # determine the frameing frames
            interval = random.choice(self.interval_list)

            center_frame_idx = random.randint(self.num_half_frames * interval, current_len - self.num_half_frames *interval - 1)
            start_frame_idx = (center_frame_idx - self.num_half_frames * interval)
            end_frame_idx = start_frame_idx + (self.num_frame - 1) * interval
            # frame_name = f'{center_frame_idx:08d}'
            frame_list = list(range(start_frame_idx, end_frame_idx + 1, interval))
            # print(frame_list)
            # Sample number should equal to the numer we set
            new_clip_sequence = sorted(os.listdir(os.path.join(current_gt_root, clip_name)))
            assert len(frame_list) == self.num_frame, (f'Wrong length of frame list: {len(frame_list)}')


        else:
            frame_list = list(range(start_id, start_id + current_len))
            # Sample number should equal to the all frames number in on folder
            assert len(frame_list) == current_len, (f'Wrong length of frame list: {len(frame_list)}')

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            frame_list.reverse()

        if self.is_train:
            # Random for the reference frame
            frame_list.append(random.randint(0,current_len-1))            


        # get the GT frame (as the center frame)
        img_gts = []
        img_lqs = []
        for frame in frame_list:

            if self.is_train:
                # img_gt_path = os.path.join(current_gt_root, clip_name, f'{frame:05d}.jpg') ## Adaptive for DAVIS and YoutubeVOS
                img_gt_path = os.path.join(current_gt_root, clip_name, new_clip_sequence[frame])
                current_frame = Image.open(img_gt_path).convert("RGB")
                current_frame = resize_256_short_side(current_frame)
                img_gt=rgb2lab(current_frame)
                img_gts.append(np.array(img_gt))
            else:
                img_gt_path = os.path.join(current_gt_root, clip_name, f'{frame:08d}.png')
                img_gt=rgb2lab(Image.open(img_gt_path).convert("RGB"))
                img_gts.append(np.array(img_gt))
            # img_gt = cv2.imread(img_gt_path)
            # img_gt = img_gt.astype(np.float32) / 255.
            # img_gts.append(img_gt)

            if not self.is_train:
                img_lq_path = os.path.join(current_lq_root, clip_name, f'{frame:08d}.png')

                img_lq=rgb2lab(Image.open(img_lq_path).convert("RGB"))
                img_lqs.append(np.array(img_lq))
                # img_lq = cv2.imread(img_lq_path)
                # img_lq = img_lq.astype(np.float32) / 255.
                # img_lqs.append(img_lq)

        if self.is_train:
            img_lqs = img_gts
            img_gts, img_lqs = paired_random_crop(img_gts, img_lqs, gt_size_w, gt_size_h, self.scale, clip_name)
            img_lqs, img_gts = degradation_video_colorization_v4(img_gts) ## LAB now
        else:
            for i in range(len(img_gts)): ##TODO: inference stage: set the non-reference AB channel to 0 [√]
                img_gts[i] = cv2.resize(img_gts[i], (gt_size_w, gt_size_h), interpolation = cv2.INTER_AREA)
                img_lqs[i] = cv2.resize(img_lqs[i], (gt_size_w, gt_size_h), interpolation = cv2.INTER_AREA)

        # augmentation - flip, rotate
        img_lqs.extend(img_gts)
        if self.is_train:
            img_lqs = augment(img_lqs, self.data_config['use_flip'], self.data_config['use_rot'])

        img_results = []
        for x in img_lqs:
            img_results.append(to_mytensor(x))

        if self.data_config['normalizing']:
            # transform_normalize=transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
            for i in range(len(img_results)): 
                img_results[i]=Normalize_LAB(img_results[i])

        if self.is_train:
            img_lqs = torch.stack(img_results[:self.num_frame+1], dim=0)
            img_gts = torch.stack(img_results[self.num_frame+1:], dim=0)
        else:
            img_lqs = torch.stack(img_results[:current_len], dim=0)
            img_gts = torch.stack(img_results[current_len:], dim=0)           

        # img_lqs: (t, c, h, w)
        # img_gt: (t, c, h, w)
        # key: str
        return {'lq': img_lqs, 'gt': img_gts, 'key': key, 'frame_list': frame_list, 'video_name': os.path.basename(current_lq_root), 'clip_name': clip_name}

    def __len__(self):
        return len(self.lq_frames)

