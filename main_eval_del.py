'''
* eval한 이후에 용량 save위해 결과 파일 지워주는 거까지 수행하는 코드
'''
import warnings
warnings.filterwarnings("ignore")
import logging
import os
import gc
import torch
import shutil
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
import torch.nn as nn
import torch.distributed as dist
from datetime import timedelta

import argparse

# Silence noisy OpenMMLab loggers during model init/eval.
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
    rank, PATH_CONFIG, tag = init_dist_pytorch()
    # model_name = PATH_CONFIG.split('/')[2].replace('.yml', '') # hj
    
    from pipelines.pipeline_detection_v1_0 import PipelineDetection_v1_0
    pline = PipelineDetection_v1_0(path_cfg=PATH_CONFIG, mode='test', rank = rank, tag = tag)
    max_epoch = 0

    model_name = pline.cfg.GENERAL.NAME
    
    for epoch in range(100,0,-1):
        PATH_MODEL = '/home/user/heejun/L4DR/logs/' + model_name + '/'+ tag + '/models/model_'+str(epoch)+'.pt'
        if os.path.exists(PATH_MODEL):
            max_epoch = epoch
            break
    
    for epoch in range(max_epoch, -1, -1):
    # for epoch in range(17, -1, -1): # (0~17)
    # for epoch in [34]: # 18~35
    # for epoch in range(1, 35):
        PATH_MODEL = '/home/user/heejun/L4DR/logs/' + model_name + '/'+ tag + '/models/model_'+str(epoch)+'.pt'
        pline.load_dict_model(PATH_MODEL)
        print('* Start resume, path_state_dict =  ', PATH_MODEL)
        pline.network.eval()
        pline.validate_kitti_conditional(epoch = epoch, list_conf_thr=[0.1,0.2,0.3], is_subset=False, is_print_memory=False)
        # pline.validate_kitti_conditional(epoch = epoch, list_conf_thr=[0.1], is_subset=False, is_print_memory=False)
        
        # metric 추출 후 complete_results.txt만 남기고 나머지 폴더/파일 정리
        base_threshold_dir = f'/home/user/heejun/L4DR/logs/{model_name}/test_{tag}/test_kitti/epoch_{epoch}_total'
        if os.path.isdir(base_threshold_dir):
            for thr_name in os.listdir(base_threshold_dir):
                thr_path = os.path.join(base_threshold_dir, thr_name)
                if not os.path.isdir(thr_path):
                    continue
                for item in os.listdir(thr_path):
                    item_path = os.path.join(thr_path, item)
                    if item == 'complete_results.txt':
                        continue
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path, ignore_errors=True)
                    else:
                        try:
                            os.remove(item_path)
                        except FileNotFoundError:
                            pass

        # Memory cleanup after each epoch evaluation
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[Epoch {epoch}] Memory cleaned up")
