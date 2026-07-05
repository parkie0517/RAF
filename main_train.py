'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
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
import logging
nb_logger = logging.getLogger('numba')
nb_logger.setLevel(logging.ERROR)  # only show error
# Silence noisy OpenMMLab loggers during model init.
class _OpenMMLabFilter(logging.Filter):
    def filter(self, record):
        if record.name.startswith(("mmcv", "mmdet", "mmdet3d")):
            return record.levelno >= logging.WARNING
        return True

_openmmlab_filter = _OpenMMLabFilter()
for _name in ("mmcv", "mmdet", "mmcv.runner", "mmcv.cnn", "mmdet3d"):
    _logger = logging.getLogger(_name)
    _logger.setLevel(logging.WARNING)
    _logger.addFilter(_openmmlab_filter)

import argparse

def init_dist_pytorch():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',default = 0 , type=int)
    parser.add_argument('--nproc_per_node',default = 0 , type=int)
    parser.add_argument('--nnode',default = 0 , type=int)
    parser.add_argument('--local_rank', default = -1 ,type=int)
    parser.add_argument('--cfg_file', default = 'none',type=str)
    parser.add_argument('--tag', default = 'none',type=str)
    args = parser.parse_args()
    PATH_CONFIG = args.cfg_file
    local_rank = args.local_rank
    if local_rank == -1:
        env_local_rank = os.environ.get("LOCAL_RANK", None)
        if env_local_rank is not None:
            local_rank = int(env_local_rank)
    if local_rank == -1:
        return -1, args.cfg_file, args.tag

    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(local_rank % num_gpus)
    torch.cuda.empty_cache()
    world_size = int(os.environ.get("WORLD_SIZE", num_gpus))
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        timeout=timedelta(minutes=30)
    )
    rank = dist.get_rank()
    return rank, args.cfg_file, args.tag

if __name__ == '__main__':
    rank, PATH_CONFIG, tag = init_dist_pytorch()
    from pipelines.pipeline_detection_v1_0 import PipelineDetection_v1_0
    pline = PipelineDetection_v1_0(path_cfg=PATH_CONFIG, mode='train', rank = rank, tag = tag)
    pline.train_network()
