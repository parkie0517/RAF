'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr
'''

import torch
import numpy as np
import open3d as o3d
import os
from tqdm import tqdm
import shutil
from torch.utils.data import Subset
import pickle
# Ingnore numba warning
from numba.core.errors import NumbaWarning
import warnings
import logging
warnings.simplefilter('ignore', category=NumbaWarning)
warnings.filterwarnings("ignore")
import os
import torch
from .vis import save_frame_vis
from torch.utils.tensorboard import SummaryWriter
from utils.util_pipeline import *
from utils.util_point_cloud import *
from utils.util_config import cfg, cfg_from_yaml_file
from utils.util_point_cloud import Object3D
import utils.kitti_eval.kitti_common as kitti
from utils.kitti_eval.eval import get_official_eval_result
from utils.util_optim import clip_grad_norm_
import matplotlib.pyplot as plt
import cv2
import time
from torch.profiler import profile, record_function, ProfilerActivity

from easydict import EasyDict
from utils.util_calib import *


class PipelineDetection_v1_1():
    def __init__(self, path_cfg=None, mode='train', rank = 0, tag = 'default'):
        '''
        * mode in ['train', 'test', 'vis']
        *   'train' denotes both train & test
        *   'test'  denotes mode for inference
        '''
        self.cfg = cfg_from_yaml_file(path_cfg, cfg)
        self.mode = mode
        self.tag = tag
        self.is_logging = False
        self.update_cfg_regarding_mode()
        self.rank = rank
        self.dist = rank != -1
        self.local_rank = int(os.environ.get("LOCAL_RANK", self.rank))
        if not self.dist: # go in 
            self.rank = 0
        if self.cfg.GENERAL.SEED is not None: # go in
            try:
                set_random_seed(cfg.GENERAL.SEED, cfg.GENERAL.IS_CUDA_SEED, cfg.GENERAL.IS_DETERMINISTIC)
            except:
                print('* Exception error: check cfg.GENERAL for seed')
                set_random_seed(cfg.GENERAL.SEED)
        if self.rank == 0:
            print('* K-Radar dataset is being loaded.')
        self.dataset_train = build_dataset(self, split='train') if self.mode == 'train' else None
        self.dataset_test = build_dataset(self, split='test')
        if self.dist:
            if mode == 'train':
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset_train)
            self.test_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset_test, shuffle=False)
        if self.rank == 0: # go in
            print('* The dataset is loaded.')
        if mode == 'train': # for setting scheduler
            self.cfg.DATASET.NUM = len(self.dataset_train)
        elif mode in ['test', 'vis']:
            self.cfg.DATASET.NUM = len(self.dataset_test)
        # print(self.cfg.DATASET.CLASS_INFO.NUM_CLS) # check if it is updated
        
        if self.rank == 0:
            print('* NUM done.')
        self.network = build_network(self).cuda()
        if self.dist:
            device_id = self.local_rank % torch.cuda.device_count()
            self.network = torch.nn.parallel.DistributedDataParallel(
                self.network,
                device_ids=[device_id],
                output_device=device_id,
                find_unused_parameters=True
            )
        if self.rank == 0:
            print('* network done.')
        self.optimizer = build_optimizer(self, self.network)
        if self.rank == 0:
            print('* optimizer done.')
        self.scheduler = build_scheduler(self, self.optimizer)
        if self.rank == 0:
            print('* scheduler done.')
        self.epoch_start = 0
        
        # Logging
        if self.cfg.GENERAL.LOGGING.IS_LOGGING and self.rank == 0:
            self.set_logging(path_cfg)

        # Validation
        if self.cfg.VAL.IS_VALIDATE or mode != 'train':
            self.set_validate()
        else:
            self.is_validate = False
        
        # Resume from checkpoint
        if self.cfg.GENERAL.RESUME.IS_RESUME:
            self.resume_network()
        
        # load checkpoints and freeze modules
        uem_cfg = self.cfg.MODEL.get('UEM_BACKBONE_3D', None)
        uem_pretrained = bool(uem_cfg and uem_cfg.get('PRETRAINED', False))
        if self.cfg.GENERAL.FINETUNE.IS_FINETUNE or uem_pretrained:
            if self.mode == 'train':
                self.loadCheckpoint()

        self.cfg_dataset_ver2 = self.cfg.get('cfg_dataset_ver2', False)
        self.get_loss_from = self.cfg.get('get_loss_from', 'head')
        self.optim_fastai = True \
            if self.cfg.OPTIMIZER.NAME in ['adam_onecycle', 'adam_cosineanneal'] else False
        self.grad_norm_clip = self.cfg.OPTIMIZER.get('GRAD_NORM_CLIP', -1)

        # Vis
        #self.set_vis()
        
        # self.show_pline_description()

    def update_cfg_regarding_mode(self):
        '''
        * You don't have to update values in cfg changed in dataset
        * They are related in pointer
        * e.g., check print(self.cfg.DATASET.CLASS_INFO.NUM_CLS) after dataset initialization
        '''
        if self.mode == 'train':
            pass
        elif self.mode == 'test':
            pass
        elif self.mode == 'vis':
            self.cfg.GET_ITEM = {
                'rdr_sparse_cube'   : True,
                'rdr_tesseract'     : False,
                'rdr_cube'          : True,
                'rdr_cube_doppler'  : False,
                'ldr_pc_64'         : True,
                'cam_front_img'     : True,
            }
        else:
            print('* Exception error (Pipeline): check modify_cfg')
        return

    def set_validate(self):
        self.is_validate = True
        self.is_consider_subset = self.cfg.VAL.IS_CONSIDER_VAL_SUBSET
        self.val_per_epoch_subset = self.cfg.VAL.VAL_PER_EPOCH_SUBSET
        self.val_num_subset = self.cfg.VAL.NUM_SUBSET
        self.val_per_epoch_full = self.cfg.VAL.VAL_PER_EPOCH_FULL

        self.val_keyword = self.cfg.VAL.CLASS_VAL_KEYWORD # for kitti_eval
        list_val_keyword_keys = list(self.val_keyword.keys()) # same order as VAL.CLASS_VAL_KEYWORD.keys()
        self.list_val_care_idx = []

        # index matching with kitti_eval
        for cls_name in self.cfg.VAL.LIST_CARE_VAL:
            idx_val_cls = list_val_keyword_keys.index(cls_name)
            self.list_val_care_idx.append(idx_val_cls)
        # print(self.list_val_care_idx)

        ### Consider output of network and dataset ###
        if self.cfg.VAL.REGARDING == 'anchor':
            self.val_regarding = 0 # anchor
            self.list_val_conf_thr = self.cfg.VAL.LIST_VAL_CONF_THR
        else:
            print('* Exception error: check VAL.REGARDING')
        ### Consider output of network and dataset ###

    def set_vis(self):
        if self.cfg_dataset_ver2:
            pass # TODO
        else:
            self.dict_cls_name_to_id = self.cfg.DATASET.CLASS_INFO.CLASS_ID
            self.dict_cls_id_to_name = dict()
            for k, v in self.dict_cls_name_to_id.items():
                if v != -1:
                    self.dict_cls_id_to_name[v] = k
            self.dict_cls_name_to_bgr = self.cfg.VIS.CLASS_BGR
            self.dict_cls_name_to_rgb = self.cfg.VIS.CLASS_RGB
    
    def show_pline_description(self):
        print('* newtork (description start) -------')
        print(self.network)
        print('* newtork (description end) ---------')
        print('* optimizer (description start) -----')
        print(self.optimizer)
        print('* optimizer (description end) -------')
        print(f'* mode = {self.mode}')
        len_data = self.cfg.DATASET.NUM
        print(f'* dataset length = {len_data}')
    
    def set_logging(self, path_cfg, is_print_where=True):
        self.is_logging = True
        str_local_time = get_local_time_str()
        str_exp = self.mode +'_'+ self.tag
        self.path_log = os.path.join(self.cfg.GENERAL.LOGGING.PATH_LOGGING, self.cfg.GENERAL.NAME, str_exp)
        if is_print_where:
            print(f'* Start logging in {str_exp}')
        if not (os.path.exists(self.path_log)):
            os.makedirs(self.path_log)
        else:
            str_exp = self.mode +'_'+ self.tag + str_local_time
            self.path_log = os.path.join(self.cfg.GENERAL.LOGGING.PATH_LOGGING, self.cfg.GENERAL.NAME, str_exp)
            os.makedirs(self.path_log)

        self.log_train_iter = SummaryWriter(os.path.join(self.path_log, 'train_iter'), comment='iteration')
        self.log_train_epoch = SummaryWriter(os.path.join(self.path_log, 'train_epoch'), comment='epoch')
        self.log_test = SummaryWriter(os.path.join(self.path_log, 'test'), comment='test')
        self.log_iter_start = None

        self.is_save_model = self.cfg.GENERAL.LOGGING.IS_SAVE_MODEL
        try:
            self.interval_epoch_model = self.cfg.GENERAL.LOGGING.INTERVAL_EPOCH_MODEL
            self.interval_epoch_util = self.cfg.GENERAL.LOGGING.INTERVAL_EPOCH_UTIL
        except:
            self.interval_epoch_model = 1
            self.interval_epoch_util = 5
            print('* Exception error (Pipeline): check LOGGING.INTERVAL_EPOCH_MODEL/UTIL')
        if self.is_save_model:
            os.makedirs(os.path.join(self.path_log, 'models'))
            os.makedirs(os.path.join(self.path_log, 'utils'))

        if self.cfg.MODEL.get('UEM', False):
            if self.cfg.MODEL.UEM.get('VIS', False):
                self.save_confmap_path = os.path.join(self.path_log, 'vis_confmap')
                os.makedirs(self.save_confmap_path)
                
                
        
        # cfg backup (same files, just for identification)
        name_file_origin = path_cfg.split('/')[-1] # original cfg file name
        name_file_cfg = 'config.yml'
        shutil.copy2(path_cfg, os.path.join(self.path_log, name_file_origin))
        shutil.copy2(path_cfg, os.path.join(self.path_log, name_file_cfg))

        # code backup (TBD)

    def resume_network(self):
        path_exp = self.cfg.GENERAL.RESUME.PATH_EXP
        path_state_dict = os.path.join(path_exp, 'utils')
        epoch = self.cfg.GENERAL.RESUME.START_EP
        list_epochs = sorted(list(map(lambda x: int(x.split('.')[0].split('_')[1]), os.listdir(path_state_dict))))
        epoch = list_epochs[-1] if epoch is None else epoch

        path_state_dict = os.path.join(path_state_dict, f'util_{epoch}.pt')
        print('* Start resume, path_state_dict =  ', path_state_dict)
        state_dict = torch.load(path_state_dict)

        try:
            self.epoch_start = epoch + 1
            target_net = self.network.module if self.dist else self.network
            target_net.load_state_dict(state_dict['model_state_dict']) # TODO 요기서 애러 발생
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            self.log_iter_start = state_dict['idx_log_iter']
            print(f'* Network & Optimizer are loaded / Resume epoch is {epoch} / Start from {self.epoch_start} ...')
        # except:
            # print('* Exception error (Pipeline): check resume network')
            # exit()
        except Exception as e:
            print(f"* Exception error (Pipeline): {e}")
            traceback.print_exc()
            exit() # 오류 확인 후 프로그램 종료

        if ('scheduler_state_dict' in state_dict.keys()) and (not (self.scheduler is None)):
            self.scheduler.load_state_dict(state_dict['scheduler_state_dict'])
            print('* Scheduler is loaded')
        else:
            print('* Scheduler is started from vanilla')

        ### Copy logging folder ###
        list_copy_dirs = ['train_epoch', 'train_iter', 'test', 'test_kitti']
        if (self.cfg.GENERAL.RESUME.IS_COPY_LOGS) and (self.is_logging):
            for copy_dir in list_copy_dirs:
                shutil.copytree(os.path.join(path_exp, copy_dir), \
                    os.path.join(self.path_log, copy_dir), dirs_exist_ok=True)
        ### Copy logging folder ###

        return

    def loadCheckpoint(self):
        """
        * created: Heejun Park
        * Description: Function for loading pretrained checkpoints & freezing desired modules
        """
        
        target_net = self.network.module if self.dist else self.network
        current_model_state_dict = target_net.state_dict()
        filtered_state_dict = {}
        
        
        # l4dr 불러 올 거면
        if self.cfg.GENERAL.FINETUNE.IS_FINETUNE:
            
            # l4dr 체크포인트 로드하기 (models 폴더 우선, 없으면 utils 백업)
            path_exp = self.cfg.GENERAL.FINETUNE.PATH_EXP
            path_models = os.path.join(path_exp, 'models')
            path_utils = os.path.join(path_exp, 'utils')
            target_dir = path_models if os.path.isdir(path_models) else path_utils
            prefix = 'model' if target_dir == path_models else 'util'

            if not os.path.isdir(target_dir):
                raise FileNotFoundError(f"No checkpoint folder found under {path_models} or {path_utils}")

            epoch = self.cfg.GENERAL.FINETUNE.START_EP # 지정된 값 또는 latest
            if epoch == None:
                list_epochs = sorted([
                    int(fname.split('.')[0].split('_')[1])
                    for fname in os.listdir(target_dir)
                    if fname.startswith(f'{prefix}_') and fname.endswith('.pt')
                ])
                epoch = list_epochs[-1] if epoch is None else epoch # 최대값 뽑기

            path_state_dict = os.path.join(target_dir, f'{prefix}_{epoch}.pt')
            state_dict = torch.load(path_state_dict, map_location='cpu')

            # models 폴더는 state_dict만 저장되어 있어서 키가 없을 수 있음
            if isinstance(state_dict, dict) and ('model_state_dict' in state_dict):
                pretrained_model_state_dict = state_dict['model_state_dict']
            elif isinstance(state_dict, dict) and ('state_dict' in state_dict):
                pretrained_model_state_dict = state_dict['state_dict']
            else:
                pretrained_model_state_dict = state_dict

            # 필요한 weight 추출
            for k, v in pretrained_model_state_dict.items():
                if k.split(".")[0] in ['backbone_3d', 'point_head', 'vfe'] and k in current_model_state_dict and v.shape == current_model_state_dict[k].shape:
                    filtered_state_dict[k] = v
      
        
        # uem인코더 weight 부르기
        uem_cfg = self.cfg.MODEL.get('UEM_BACKBONE_3D', None)
        if uem_cfg and uem_cfg.get('PRETRAINED', False):
            
            # simple voxel encoder 로딩
            checkpoint_path = uem_cfg.PRETRAINED.PATH
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # 필요한 weight 추출
            for k, v in checkpoint.items():
                if k.split('.')[0] == 'uem_encoder' and k in current_model_state_dict and v.shape == current_model_state_dict[k].shape:
                    filtered_state_dict[k] = v
        
            
        # weight update 수행
        current_model_state_dict.update(filtered_state_dict)
        target_net.load_state_dict(current_model_state_dict, strict=False)

        # 얼리고 싶은 parameter들 얼리기
        if self.cfg.MODEL.FREEZE.FREEZE_MODULES:
            for name, param in self.network.named_parameters():
                if name.split(".")[0] in self.cfg.MODEL.FREEZE.FREEZE_MODULES:
                    param.requires_grad = False        
        
        ### Copy logging folder ###
        list_copy_dirs = ['train_epoch', 'train_iter', 'test']
        if (self.cfg.GENERAL.RESUME.IS_COPY_LOGS) and (self.is_logging):
            for copy_dir in list_copy_dirs:
                shutil.copytree(os.path.join(path_exp, copy_dir), \
                    os.path.join(self.path_log, copy_dir), dirs_exist_ok=True)
        ### Copy logging folder ###

        return

    def toEval(self):
        """
        Freeze된 모듈들을 eval 모드로 변경하는 함수
        """
        for name, module in self.network.named_modules():
            if name.split('.')[0] in self.cfg.MODEL.FREEZE.FREEZE_MODULES:
                module.eval()
    
    def train_network(self, is_shuffle=True):
        self.network.train()
        # print("Before eval_frozen_modules:")
        # for name, module in self.network.named_modules():
        #     if name.split('.')[0] in self.cfg.MODEL.FREEZE.FREEZE_MODULES:
        #         print(f"  {name}: training = {module.training}")  
        if self.cfg.GENERAL.FINETUNE.IS_FINETUNE:
            self.toEval()
        # print("After eval_frozen_modules:")
        # for name, module in self.network.named_modules():
        #     if name.split('.')[0] in self.cfg.MODEL.FREEZE.FREEZE_MODULES:
        #         print(f"  {name}: training = {module.training}")
        
        if self.dist:
            data_loader_train = torch.utils.data.DataLoader(self.dataset_train, \
                batch_size = self.cfg.OPTIMIZER.BATCH_SIZE, shuffle = False, \
                collate_fn = self.dataset_train.collate_fn,
                num_workers = self.cfg.OPTIMIZER.NUM_WORKERS, pin_memory=True, drop_last = True, sampler=self.train_sampler)
        else:
            data_loader_train = torch.utils.data.DataLoader(self.dataset_train, \
                batch_size = self.cfg.OPTIMIZER.BATCH_SIZE, shuffle = is_shuffle, \
                collate_fn = self.dataset_train.collate_fn,
                num_workers = self.cfg.OPTIMIZER.NUM_WORKERS, pin_memory=True, drop_last = True)
        epoch_start = self.epoch_start
        epoch_end = self.cfg.OPTIMIZER.MAX_EPOCH

        if self.is_logging:
            idx_log_iter = 0 if self.log_iter_start is None else self.log_iter_start

        # if self.optim_fastai: # pass
        #     accumulated_iter = 0
        #     cfg_optim = self.cfg.OPTIMIZER
        #     use_amp = cfg_optim.get('USE_AMP', False)
        #     scaler = torch.cuda.amp.GradScaler(enabled=use_amp, init_scale=cfg_optim.get('LOSS_SCALE_FP16', 2.0**16))

        for epoch in range(epoch_start, epoch_end):
            if self.dist:
                self.train_sampler.set_epoch(epoch)
            if self.rank==0:
                print(f'* Training epoch = {epoch}/{epoch_end-1}')
            if self.rank==0 and self.is_logging :
                print(f'* Logging path = {self.path_log}')
            
            self.network.train()
            if self.cfg.GENERAL.FINETUNE.IS_FINETUNE:
                self.toEval()
            self.network.training = True
            
            avg_loss = []
            # for idx_iter, dict_datum in enumerate(tqdm(data_loader_train)):
            for idx_iter, dict_datum in enumerate(tqdm(data_loader_train, ncols=60, dynamic_ncols=False)):
                # if self.optim_fastai:
                #     self.scheduler.step(accumulated_iter, epoch)
                #torch.cuda.empty_cache()
                dict_net = self.network(dict_datum)
                dict_net['epoch'] = epoch+1
                str_exp = self.mode +'_'+ self.tag
                self.path_log = os.path.join(self.cfg.GENERAL.LOGGING.PATH_LOGGING, self.cfg.GENERAL.NAME, str_exp)
                dict_net['path'] = os.path.join(self.path_log,'train_loss.txt')
                # if self.dist: # 멀티 gpu학습이면
                #     if self.get_loss_from == 'head':
                #         loss = self.network.module.head.loss(dict_net)
                #     elif self.get_loss_from == 'detector':
                #         loss = self.network.module.loss(dict_net)
                # else:
                    
                target_net = self.network.module if self.dist else self.network
                if self.get_loss_from == 'head':
                    loss = target_net.head.loss(dict_net)
                elif self.get_loss_from == 'detector': # 요기 goin
                    loss = target_net.loss(dict_net)
                try:
                    log_avg_loss = loss.cpu().detach().item()
                except:
                    log_avg_loss = loss
                avg_loss.append(log_avg_loss)

                if self.optim_fastai:
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    clip_grad_norm_(self.network.parameters(), cfg_optim.GRAD_NORM_CLIP)
                    scaler.step(self.optimizer)
                    scaler.update()
                    accumulated_iter += 1
                else:
                    if loss == 0.:
                        pass
                        # print('loss is 0.') # No objs for all samples
                    elif torch.isfinite(loss):
                        loss.backward()
                    else:
                        print('* Exception error (pipeline): nan or inf loss happend')
                        print('* Meta: ', dict_datum['meta'])

                    self.optimizer.step()
                    if not (self.scheduler is None):
                        self.scheduler.step()
                
                self.optimizer.zero_grad()

                if self.is_logging:
                    dict_logging = dict_net['logging']
                    idx_log_iter +=1
                    for k, v in dict_logging.items():
                        self.log_train_iter.add_scalar(f'train/{k}', v, idx_log_iter)
                    if not (self.scheduler is None):
                        if self.optim_fastai:
                            lr = float(self.optimizer.lr)
                            self.log_train_iter.add_scalar(f'train/learning_rate', lr, idx_log_iter)
                        else:
                            lr = self.scheduler.get_last_lr()
                            self.log_train_iter.add_scalar(f'train/learning_rate', lr[0], idx_log_iter)

                # free memory (Killed error, checked with htop)
                if 'pointer' in dict_datum.keys():
                    for dict_item in dict_datum['pointer']:
                        for k in dict_item.keys():
                            if k != 'meta':
                                dict_item[k] = None
                for temp_key in dict_datum.keys():
                    dict_datum[temp_key] = None
            # end of for loop for training
            
            if self.rank==0 and self.is_save_model :
                # epoch: indexing from 0
                path_dict_model = os.path.join(self.path_log, 'models', f'model_{epoch}.pt')
                path_dict_util = os.path.join(self.path_log, 'utils', f'util_{epoch}.pt')
                if self.dist:
                    if (epoch+1) % self.interval_epoch_model == 0:
                        torch.save(self.network.module.state_dict(), path_dict_model)
                    if (epoch+1) % self.interval_epoch_util == 0:
                        dict_util = {
                            'epoch': epoch,
                            'model_state_dict': self.network.module.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'idx_log_iter': idx_log_iter, 
                        }
                        if self.optim_fastai:
                            dict_util.update({'it': accumulated_iter})
                        else:
                            if not (self.scheduler is None):
                                dict_util.update({'scheduler_state_dict': self.scheduler.state_dict()})
                        torch.save(dict_util, path_dict_util)
                else:
                    if (epoch+1) % self.interval_epoch_model == 0:
                        torch.save(self.network.state_dict(), path_dict_model)
                    if (epoch+1) % self.interval_epoch_util == 0:
                        dict_util = {
                            'epoch': epoch,
                            'model_state_dict': self.network.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'idx_log_iter': idx_log_iter, 
                        }
                        if self.optim_fastai:
                            dict_util.update({'it': accumulated_iter})
                        else:
                            if not (self.scheduler is None):
                                dict_util.update({'scheduler_state_dict': self.scheduler.state_dict()})
                        torch.save(dict_util, path_dict_util)

            if self.is_logging:
                self.log_train_epoch.add_scalar(f'train/avg_loss', np.mean(avg_loss), epoch)

            if self.is_validate:
                if self.is_consider_subset:
                    if ((epoch + 1) % self.val_per_epoch_subset) == 0:
                        self.validate_kitti(epoch, list_conf_thr=self.list_val_conf_thr, is_subset=True)
                if ((epoch + 1) % self.val_per_epoch_full) == 0:
                    self.validate_kitti(epoch, list_conf_thr=self.list_val_conf_thr)

    def load_dict_model(self, path_dict_model, is_strict=False):
        x = torch.load(path_dict_model)
        target_net = self.network.module if self.dist else self.network
        target_net.load_state_dict(x, False)


    # V2
    def validate_kitti(self, epoch=None, list_conf_thr=None, is_subset=False):
        self.network.training=False
        self.network.eval()

        eval_ver2 = self.cfg.get('cfg_eval_ver2', False)
        if eval_ver2:
            class_names = []
            dict_label = self.dataset_test.label.copy()
            list_for_pop = ['calib', 'onlyR', 'Label', 'consider_cls', 'consider_roi', 'remove_0_obj']
            for temp_key in list_for_pop:
                dict_label.pop(temp_key)
            for k, v in dict_label.items():
                _, logit_idx, _, _ = v
                if logit_idx > 0:
                    class_names.append(k)
            self.dict_cls_id_to_name = dict()
            for idx_cls, cls_name in enumerate(class_names):
                self.dict_cls_id_to_name[(idx_cls+1)] = cls_name # 1 for Background
        

        ### Check is_validate with small dataset ###
        if is_subset:
            is_shuffle = False
            tqdm_bar = tqdm(total=self.val_num_subset, desc='* Test (Subset): ')
            log_header = 'val_sub'
        else:
            is_shuffle = False
            tqdm_bar = tqdm(total=len(self.dataset_test), desc='* Test (Total): ')
            log_header = 'val_tot'
        if self.dist:
            data_loader = torch.utils.data.DataLoader(self.dataset_test,
                    batch_size = 1, shuffle = False,
                    collate_fn = self.dataset_test.collate_fn,
                    num_workers = self.cfg.OPTIMIZER.NUM_WORKERS, sampler=self.test_sampler, pin_memory=True)
        else:
            data_loader = torch.utils.data.DataLoader(self.dataset_test,
                    batch_size = 1, shuffle = False,
                    collate_fn = self.dataset_test.collate_fn,
                    num_workers = self.cfg.OPTIMIZER.NUM_WORKERS,
                    pin_memory=True
                    )
        
        if epoch is None:
            dir_epoch = 'none'
        else:
            dir_epoch = f'epoch_{epoch}_subset' if is_subset else f'epoch_{epoch}_total'

        # initialize via VAL.LIST_VAL_CONF_THR
        if self.rank == 0 :
            path_dir = os.path.join(self.path_log, 'test_kitti', dir_epoch)
        # print(path_dir)
        for conf_thr in list_conf_thr:
            os.makedirs(os.path.join(path_dir, f'{conf_thr}'), exist_ok=True)
            with open(path_dir + f'/{conf_thr}/' + 'val.txt', 'w') as f:
                f.write('')
            f.close()

        for idx_datum, dict_datum in enumerate(data_loader):
            if is_subset & (idx_datum >= self.val_num_subset):
                break
            
            try:
                dict_out = self.network(dict_datum) # inference
                is_feature_inferenced = True
            except:
                print('* Exception error (Pipeline): error during inferencing a sample -> empty prediction')
                print('* Meta info: ', dict_out['meta'])
                is_feature_inferenced = False

            idx_name = str(idx_datum).zfill(6)

            ### for every conf in list_conf_thr ###
            for conf_thr in list_conf_thr:
                preds_dir = os.path.join(path_dir, f'{conf_thr}', 'pred')
                labels_dir = os.path.join(path_dir, f'{conf_thr}', 'gt')
                desc_dir = os.path.join(path_dir, f'{conf_thr}', 'desc')
                list_dir = [preds_dir, labels_dir, desc_dir]
                split_path = path_dir + f'/{conf_thr}/' + 'val.txt'
                for temp_dir in list_dir:
                    os.makedirs(temp_dir, exist_ok=True)

                if is_feature_inferenced:
                    if eval_ver2:
                        pred_dicts = dict_out['pred_dicts'][0]
                        pred_boxes = pred_dicts['pred_boxes'].detach().cpu().numpy()
                        pred_scores = pred_dicts['pred_scores'].detach().cpu().numpy()
                        pred_labels = pred_dicts['pred_labels'].detach().cpu().numpy()
                        list_pp_bbox = []
                        list_pp_cls = []

                        for idx_pred in range(len(pred_labels)):
                            x, y, z, l, w, h, th = pred_boxes[idx_pred]
                            score = pred_scores[idx_pred]
                            
                            if score > conf_thr:
                                cls_idx = int(np.round(pred_labels[idx_pred]))
                                cls_name = class_names[cls_idx-1]
                                list_pp_bbox.append([score, x, y, z, l, w, h, th])
                                list_pp_cls.append(cls_idx)
                            else:
                                continue
                        pp_num_bbox = len(list_pp_cls)
                        dict_out_current = dict_out
                        dict_out_current.update({
                            'pp_bbox': list_pp_bbox,
                            'pp_cls': list_pp_cls,
                            'pp_num_bbox': pp_num_bbox,
                            'pp_desc': dict_out['meta'][0]['desc']
                        })
                    else:
                        dict_out_current = self.network.list_modules[-1].get_nms_pred_boxes_for_single_sample(dict_out, conf_thr, is_nms=True)
                else:
                    dict_out_current = update_dict_feat_not_inferenced(dict_out) # mostly sleet for lpc (e.g. no measurement)                
                if dict_out is None:
                    print('* Exception error (Pipeline): dict_item is None in validation')
                    continue

                dict_out = dict_datum_to_kitti(self, dict_out)

                if len(dict_out['kitti_gt']) == 0: # no eval for emptry obj label
                    pass
                else:
                    ### Gt ###
                    for idx_label, label in enumerate(dict_out['kitti_gt']):
                        open_mode = 'w' if idx_label == 0 else 'a'
                        with open(labels_dir + '/' + idx_name + '.txt', open_mode) as f:
                            f.write(label+'\n')
                    ### Gt ###

                    ### Process description ###
                    with open(desc_dir + '/' + idx_name + '.txt', 'w') as f:
                        f.write(dict_out['kitti_desc'])
                    ### Process description ###

                    ### Pred: do not care len 0 with if else: already care as dummy ###
                    for idx_pred, pred in enumerate(dict_out['kitti_pred']):
                        open_mode = 'w' if idx_pred == 0 else 'a'
                        with open(preds_dir + '/' + idx_name + '.txt', open_mode) as f:
                            f.write(pred+'\n')
                    ### Pred: do not care len 0 with if else: already care as dummy ###

                    str_log = idx_name + '\n'
                    with open(split_path, 'a') as f:
                        f.write(str_log)
            
            # free memory (Killed error, checked with htop)
            if 'pointer' in dict_datum.keys():
                for dict_item in dict_datum['pointer']:
                    for k in dict_item.keys():
                        if k != 'meta':
                            dict_item[k] = None
            for temp_key in dict_datum.keys():
                dict_datum[temp_key] = None
            tqdm_bar.update(1)
        tqdm_bar.close()

        ### Validate per conf ###
        for conf_thr in list_conf_thr:
            preds_dir = os.path.join(path_dir, f'{conf_thr}', 'pred')
            labels_dir = os.path.join(path_dir, f'{conf_thr}', 'gt')
            desc_dir = os.path.join(path_dir, f'{conf_thr}', 'desc')
            split_path = path_dir + f'/{conf_thr}/' + 'val.txt'

            dt_annos = kitti.get_label_annos(preds_dir)
            val_ids = read_imageset_file(split_path)
            gt_annos = kitti.get_label_annos(labels_dir, val_ids)

            list_metrics = []
            for idx_cls_val in self.list_val_care_idx:
                dict_metrics, result_log = get_official_eval_result(gt_annos, dt_annos, idx_cls_val, is_return_with_dict=True)
                print(f'-----conf{conf_thr}-----')
                print(result_log)
                list_metrics.append(dict_metrics)

            for dict_metrics in list_metrics:
                cls_name = dict_metrics['cls']
                ious = dict_metrics['iou']
                bevs = dict_metrics['bev']
                ap3ds = dict_metrics['3d']
                self.log_test.add_scalars(f'{log_header}/BEV_conf_thr_{conf_thr}', {
                    f'iou_{ious[0]}_{cls_name}': bevs[0],
                    f'iou_{ious[1]}_{cls_name}': bevs[1],
                    f'iou_{ious[2]}_{cls_name}': bevs[2],
                }, epoch)
                self.log_test.add_scalars(f'{log_header}/3D_conf_thr_{conf_thr}', {
                    f'iou_{ious[0]}_{cls_name}': ap3ds[0],
                    f'iou_{ious[1]}_{cls_name}': ap3ds[1],
                    f'iou_{ious[2]}_{cls_name}': ap3ds[2],
                }, epoch)
        ### Validate per conf ###


    def validate_kitti_visibility(self, epoch=None, list_conf_thr=None, is_subset=False, is_print_memory=False, savevis = False, subset_num=None, visibility_clean=None, visibility_noisy=None):
        self.network.eval()
        self.network.training=False
        eval_ver2 = self.cfg.get('cfg_eval_ver2', False)
        if eval_ver2:
            class_names = []
            dict_label = self.dataset_test.label.copy()
            list_for_pop = ['calib', 'onlyR', 'Label', 'consider_cls', 'consider_roi', 'remove_0_obj']
            for temp_key in list_for_pop:
                dict_label.pop(temp_key)
            for k, v in dict_label.items():
                _, logit_idx, _, _ = v
                if logit_idx > 0:
                    class_names.append(k)
            self.dict_cls_id_to_name = dict()
            for idx_cls, cls_name in enumerate(class_names):
                self.dict_cls_id_to_name[(idx_cls+1)] = cls_name # 1 for Background
        
        use_visibility = (visibility_clean is not None) or (visibility_noisy is not None)
        if use_visibility:
            visibility_clean_set = set(visibility_clean or [])
            visibility_noisy_set = set(visibility_noisy or [])
            condition_list = ['clean', 'noisy', 'partial']

            def _parse_seq(value):
                if value is None:
                    return None
                try:
                    return int(value)
                except (TypeError, ValueError):
                    if isinstance(value, str):
                        digits = ''.join(ch for ch in value if ch.isdigit())
                        if digits:
                            return int(digits)
                return None

            def get_visibility_tag(seq_value):
                seq_int = _parse_seq(seq_value)
                if seq_int in visibility_clean_set:
                    return 'clean'
                if seq_int in visibility_noisy_set:
                    return 'noisy'
                return 'partial'
        else:
            condition_list = ['close', 'mid', 'far']
            dist_close_thr = 24.0
            dist_mid_thr = 48.0

            def get_distance_tag(xc, yc, zc):
                dist = float(np.sqrt(xc ** 2 + yc ** 2 + zc ** 2))
                if dist < dist_close_thr:
                    return 'close'
                elif dist <= dist_mid_thr:
                    return 'mid'
                return 'far'

        def has_valid_boxes(annos):
            if not annos:
                return False
            for anno in annos:
                bbox = anno.get('bbox', None)
                if bbox is not None and len(bbox) > 0:
                    return True
            return False

        # Check is_validate with small dataset
        max_subset_samples = subset_num if subset_num is not None else self.val_num_subset
        if is_subset:
            is_shuffle = False
            # tqdm_bar = tqdm(total=self.val_num_subset, desc='Test (Subset): ')
        else:
            is_shuffle = False
            # tqdm_bar = tqdm(total=len(self.dataset_test), desc='Test (Total): ')
        
        if self.dist:
            data_loader = torch.utils.data.DataLoader(self.dataset_test,
                    batch_size = 1, shuffle = False,
                    collate_fn = self.dataset_test.collate_fn,
                    num_workers = self.cfg.OPTIMIZER.NUM_WORKERS, sampler=self.test_sampler, pin_memory=False)
        else:
            data_loader = torch.utils.data.DataLoader(self.dataset_test,
                    batch_size = 1, shuffle = False,
                    collate_fn = self.dataset_test.collate_fn,
                    num_workers = self.cfg.OPTIMIZER.NUM_WORKERS)
        
        if epoch is None:
            dir_epoch = 'none'
        else:
            dir_epoch = f'epoch_{epoch}_subset' if is_subset else f'epoch_{epoch}_total'
        vis = []
        # initialize via VAL.LIST_VAL_CONF_THR
        if self.rank == 0 :
            path_dir = os.path.join(self.path_log, 'test_kitti', dir_epoch)
            condition_names = ['all'] + condition_list
            dir_cache = dict()

            def write_kitti_lines(path_file, lines):
                with open(path_file, 'w') as f:
                    if len(lines) != 0:
                        for line in lines:
                            f.write(line+'\n')

            for conf_thr in list_conf_thr:
                dir_cache[conf_thr] = dict()
                os.makedirs(os.path.join(path_dir, f'{conf_thr}'), exist_ok=True)
                for condition in condition_names:
                    cond_dir = os.path.join(path_dir, f'{conf_thr}', condition)
                    os.makedirs(cond_dir, exist_ok=True)
                    dir_cache[conf_thr][condition] = {
                        'preds': os.path.join(cond_dir, 'preds'),
                        'gts': os.path.join(cond_dir, 'gts'),
                        'desc': os.path.join(cond_dir, 'desc'),
                        'split': os.path.join(cond_dir, 'val.txt')
                    }
                    for leaf in ['preds', 'gts', 'desc']:
                        os.makedirs(dir_cache[conf_thr][condition][leaf], exist_ok=True)
                    with open(dir_cache[conf_thr][condition]['split'], 'w') as f:
                        f.write('')

            with torch.no_grad():
                # Creating gts and preds txt files for evaluation
                for idx_datum, dict_datum in enumerate(tqdm(data_loader, ncols=60, dynamic_ncols=False)):
                    if is_subset and (idx_datum >= max_subset_samples):
                        break
                    dict_out = self.network(dict_datum) # inference
                    if savevis:
                        vis.append(save_frame_vis(dict_out['pred_dicts'][0], dict_datum))
                    is_feature_inferenced = True

                    if is_print_memory:
                        print('max_memory: ', torch.cuda.max_memory_allocated(device='cuda'))
                    idx_name = str(idx_datum).zfill(6)
                    seq_value = dict_out['meta'][0].get('seq', None)
                    
                    for conf_thr in list_conf_thr:
                        dirs_by_condition = dir_cache[conf_thr]
                        preds_dir = dirs_by_condition['all']['preds']
                        labels_dir = dirs_by_condition['all']['gts']
                        desc_dir = dirs_by_condition['all']['desc']
                        split_path = dirs_by_condition['all']['split']

                        if is_feature_inferenced:
                            if eval_ver2:
                                pred_dicts = dict_out['pred_dicts'][0]
                                pred_boxes = pred_dicts['pred_boxes'].detach().cpu().numpy()
                                pred_scores = pred_dicts['pred_scores'].detach().cpu().numpy()
                                pred_labels = pred_dicts['pred_labels'].detach().cpu().numpy()
                                list_pp_bbox = []
                                list_pp_cls = []

                                for idx_pred in range(len(pred_labels)):
                                    x, y, z, l, w, h, th = pred_boxes[idx_pred]
                                    score = pred_scores[idx_pred]
                                    
                                    if score > conf_thr:
                                        cls_idx = int(np.round(pred_labels[idx_pred]))
                                        cls_name = class_names[cls_idx-1]
                                        list_pp_bbox.append([score, x, y, z, l, w, h, th])
                                        list_pp_cls.append(cls_idx)
                                    else:
                                        continue
                                pp_num_bbox = len(list_pp_cls)
                                dict_out_current = dict_out
                                dict_out_current.update({
                                    'pp_bbox': list_pp_bbox,
                                    'pp_cls': list_pp_cls,
                                    'pp_num_bbox': pp_num_bbox,
                                    'pp_desc': dict_out['meta'][0]['desc']
                                })
                            else:
                                dict_out_current = self.network.list_modules[-1].get_nms_pred_boxes_for_single_sample(dict_out, conf_thr, is_nms=True)
                        else:
                            dict_out_current = update_dict_feat_not_inferenced(dict_out) # mostly sleet for lpc (e.g. no measurement)

                        if dict_out_current is None:
                            print('* Exception error (Pipeline): dict_item is None in validation')
                            continue

                        dict_out_current = dict_datum_to_kitti(self, dict_out_current)

                        write_kitti_lines(os.path.join(labels_dir, idx_name + '.txt'), dict_out_current['kitti_gt'])
                        write_kitti_lines(os.path.join(preds_dir, idx_name + '.txt'), dict_out_current['kitti_pred'])
                        with open(os.path.join(desc_dir, idx_name + '.txt'), 'w') as f:
                            f.write(dict_out_current['kitti_desc'])
                        with open(split_path, 'a') as f:
                            f.write(idx_name + '\n')

                        if use_visibility:
                            visibility_tag = get_visibility_tag(seq_value)
                            cond_dirs = dirs_by_condition[visibility_tag]
                            write_kitti_lines(os.path.join(cond_dirs['gts'], idx_name + '.txt'), dict_out_current['kitti_gt'])
                            write_kitti_lines(os.path.join(cond_dirs['preds'], idx_name + '.txt'), dict_out_current['kitti_pred'])
                            with open(os.path.join(cond_dirs['desc'], idx_name + '.txt'), 'w') as f:
                                f.write(dict_out_current['kitti_desc'])
                            with open(cond_dirs['split'], 'a') as f:
                                f.write(idx_name + '\n')
                        else:
                            gt_distance_map = {cond: [] for cond in condition_list}
                            pred_distance_map = {cond: [] for cond in condition_list}

                            if len(dict_out_current['kitti_gt']) != 0:
                                for label_item, label in zip(dict_out_current['label'][0], dict_out_current['kitti_gt']):
                                    xc, yc, zc = label_item[2][:3]
                                    dist_tag = get_distance_tag(xc, yc, zc)
                                    gt_distance_map[dist_tag].append(label)

                            if dict_out_current.get('pp_num_bbox', 0) > 0 and dict_out_current.get('pp_bbox') is not None:
                                for pred_box, pred in zip(dict_out_current['pp_bbox'], dict_out_current['kitti_pred']):
                                    _, x, y, z, _, _, _, _ = pred_box
                                    dist_tag = get_distance_tag(x, y, z)
                                    pred_distance_map[dist_tag].append(pred)

                            for distance_cond in condition_list:
                                gt_entries = gt_distance_map[distance_cond]
                                pred_entries = pred_distance_map[distance_cond]
                                if len(gt_entries) == 0 and len(pred_entries) == 0:
                                    continue
                                cond_dirs = dirs_by_condition[distance_cond]
                                write_kitti_lines(os.path.join(cond_dirs['gts'], idx_name + '.txt'), gt_entries)
                                write_kitti_lines(os.path.join(cond_dirs['preds'], idx_name + '.txt'), pred_entries)
                                with open(os.path.join(cond_dirs['desc'], idx_name + '.txt'), 'w') as f:
                                    f.write(dict_out_current['kitti_desc'])
                                with open(cond_dirs['split'], 'a') as f:
                                    f.write(idx_name + '\n')
                                
                    # free memory (Killed error, checked with htop)
                    if 'pointer' in dict_datum.keys():
                        for dict_item in dict_datum['pointer']:
                            for k in dict_item.keys():
                                if k != 'meta':
                                    dict_item[k] = None
                    for temp_key in dict_datum.keys():
                        dict_datum[temp_key] = None
            if savevis:
                with open(path_dir +  '/lr_vis.pkl', 'wb') as f:
                        pickle.dump(vis, f)
                print("vis save in ",(path_dir + 'lr_vis.pkl'))
            ### Validate per conf ###
            all_condition_list = ['all'] + condition_list
            for conf_thr in list_conf_thr:
                for condition in all_condition_list:
                    preds_dir = os.path.join(path_dir, f'{conf_thr}', condition, 'preds')
                    labels_dir = os.path.join(path_dir, f'{conf_thr}', condition, 'gts')
                    desc_dir = os.path.join(path_dir, f'{conf_thr}', condition, 'desc')
                    split_path = path_dir + f'/{conf_thr}/' + condition + '/val.txt'

                    val_ids = read_imageset_file(split_path)
                    if len(val_ids) == 0:
                        print(f'Skip condition "{condition}" (conf {conf_thr}) because no samples were logged.')
                        continue

                    dt_annos = kitti.get_label_annos(preds_dir)
                    gt_annos = kitti.get_label_annos(labels_dir, val_ids)
                    if len(gt_annos) == 0 and len(dt_annos) == 0:
                        print(f'Skip condition "{condition}" (conf {conf_thr}) because both GT and preds are empty.')
                        continue
                    if not has_valid_boxes(gt_annos):
                        print(f'Skip condition "{condition}" (conf {conf_thr}) because GT boxes are empty.')
                        continue

                    list_metrics = []
                    list_results = []
                    for idx_cls_val in self.list_val_care_idx:
                        dict_metrics, result = get_official_eval_result(gt_annos, dt_annos, idx_cls_val, is_return_with_dict=True)
                        list_metrics.append(dict_metrics)
                        list_results.append(result)
                    print('Conf thr: ', str(conf_thr), ', Condition: ', condition)
                    with open(os.path.join(path_dir, f'{conf_thr}', 'complete_results.txt'), 'a') as f:
                        for dic_metric in list_metrics:
                            print('='*50)
                            print('Cls: ', dic_metric['cls'])
                            print('IoU:', dic_metric['iou'])
                            print('BEV: ', dic_metric['bev'])
                            print('3D: ', dic_metric['3d'])
                            print('-'*50)
                            
                            f.write('Conf thr: ' + str(conf_thr) +  ', Condition: ' + condition + '\n')
                            f.write('cls: ' + dic_metric['cls'] + '\n')
                            f.write('iou: ')
                            for iou in dic_metric['iou']:
                                f.write(str(iou) + ' ')
                            f.write('\n')
                            f.write('bev: ')
                            for bev in dic_metric['bev']:
                                f.write(str(bev) + ' ')
                            f.write('\n')
                            f.write('3d  :')
                            for det3d in dic_metric['3d']:
                                f.write(str(det3d) + ' ')
                            f.write('\n\n')
                    print('\n')
            path_check = os.path.join(path_dir, 'Conf_thr', 'complete_results.txt')
            print(f'* Check {path_check}')
        ### Validate per conf ###
