'''
* visibility에 따른 eval 수행
partial, noisy, clean.

'''
import warnings
warnings.filterwarnings("ignore")
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
import torch.nn as nn
import torch.distributed as dist
from datetime import timedelta

import argparse

def init_dist_pytorch():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',default = 0 , type=int)
    parser.add_argument('--nproc_per_node',default = 0 , type=int)
    parser.add_argument('--nnode',default = 0 , type=int)
    parser.add_argument('--local_rank', default = -1 ,type=int)
    parser.add_argument('--tag', default = 'none',type=str)
    parser.add_argument('--cfg_file', default = 'default',type=str)
    args = parser.parse_args()
    if args.local_rank == -1:
        return -1, args.cfg_file, args.tag
    else:
        
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(args.local_rank % num_gpus)
        torch.cuda.empty_cache()
        dist.init_process_group(
            backend="nccl",
            rank=args.local_rank,
            world_size=num_gpus
        )
        rank = dist.get_rank()
        return rank, args.cfg_file, args.tag

if __name__ == '__main__':
    clean = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22]
    noisy = [30, 37, 40, 47, 51, 52, 53, 57, 58]
    
    rank, PATH_CONFIG, tag = init_dist_pytorch()
    # model_name = PATH_CONFIG.split('/')[2].replace('.yml', '') # hj
    
    from pipelines.pipeline_detection_v1_1 import PipelineDetection_v1_1
    pline = PipelineDetection_v1_1(path_cfg=PATH_CONFIG, mode='test', rank = rank, tag = tag)

    model_name = pline.cfg.GENERAL.NAME
    
    # epoch = 10
    for epoch in range(19, -1, -1):
    # for epoch in [29, 34]:
        PATH_MODEL = '/home/user/heejun/L4DR/logs/' + model_name + '/'+ tag + '/models/model_'+str(epoch)+'.pt'
        pline.load_dict_model(PATH_MODEL)
        print('* Start resume, path_state_dict =  ', PATH_MODEL)
        pline.network.eval()
        pline.validate_kitti_visibility(epoch = epoch, list_conf_thr=[0.1, 0.2, 0.3], is_subset=False, is_print_memory=False, visibility_clean=clean, visibility_noisy=noisy) # visibility eval
        # pline.validate_kitti_visibility(epoch = epoch, list_conf_thr=[0.1, 0.2, 0.3], is_subset=True, is_print_memory=False, subset_num=100, visibility_clean=clean, visibility_noisy=noisy) # subset visibility eval
