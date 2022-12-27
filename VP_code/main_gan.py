import time
from collections import OrderedDict

import os
import numpy as np
import torch
import torchvision.utils as vutils
import argparse
import yaml
from trainer_gan import Trainer
import torch.multiprocessing as mp
import random
from VP_code.utils.util import mkdir_and_rename, set_seed, init_loggers



def prepare(config, opts):

    config['path']['root'] = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))
    config['path']['experiments_root'] = os.path.join(config['path']['root'], 'OUTPUT', opts.name)
    config['path']['models'] = os.path.join(config['path']['experiments_root'], 'models')
    config['path']['log'] = config['path']['experiments_root'] 
    config['path']['visualization'] = os.path.join(config['path']['experiments_root'], 'visualization')
    ## Creat experimental folder
    mkdir_and_rename(config['path']['experiments_root'])
    os.makedirs(config['path']['models'],exist_ok=True)
    ## TODO: creat other folders
    return config

def main_worker(gpu, config, opts):

    opts.local_rank = gpu
    opts.global_rank = opts.node_rank*opts.gpus + gpu


    if config['distributed']:
        torch.cuda.set_device(opts.local_rank)
        print('using GPU {} for training'.format(opts.local_rank))

        torch.distributed.init_process_group(
            backend = 'nccl', 
            init_method = opts.dist_url,
            world_size = opts.world_size, 
            rank = opts.global_rank,
            group_name='mtorch')
  
    set_seed(config['seed'])
    logger=init_loggers(config, opts)



    trainer = Trainer(config, opts, logger, debug=False)
    trainer.train()

if __name__=='__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--name',type=str,default='',help='The name of this experiment')
    parser.add_argument('--model_name',type=str,default='',help='The name of adopted model')
    parser.add_argument('--discriminator_name',type=str,default='discriminator',help='The name of adopted discriminator model')
    parser.add_argument('--epoch',type=int,default=100,help='training epoch number')
    parser.add_argument('--fix_iter',type=int,default=5000,help='The iteration number of fixed flow-estimation network')
    parser.add_argument('--fix_flow_estimator',action='store_true',help='Do not finetune the flow network')
    parser.add_argument('--which_gan',type=str,default='hinge',help=' nsgan | lsgan | hinge')
    parser.add_argument('--flow_path',type=str,default='pretrained_models/network-sintel-final.pytorch',help='the pretrainde model path of flow estimation')    

    ## DDP
    parser.add_argument('--nodes',type=int,default=1,help='how many machines')
    parser.add_argument('--gpus',type=int,default=1,help='how many GPUs in one node')
    parser.add_argument('--node_rank',type=int,default=0,help='the id of this machine (default: only one machine with id 0)')
    parser.add_argument('--dist_url',type=str,default="",help='Port Address')


    opts = parser.parse_args()
    opts.isTrain = True
    opts.world_size= opts.nodes*opts.gpus
    
    if opts.fix_flow_estimator:
        opts.fix_iter = float("inf")

    with open(os.path.join('./configs',opts.name+'.yaml'), 'r') as stream:
        config_dict = yaml.safe_load(stream)

    ## Used for the communication of multi-host
    port_num = str(random.randint(20000,30000))
    opts.dist_url = 'tcp://127.0.0.1:' + port_num
    
    config_dict=prepare(config_dict,opts)

    mp.spawn(main_worker, nprocs=opts.gpus, args=(config_dict, opts,))
