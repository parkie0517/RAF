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
from utils.util_optim import clip_grad_norm_
import matplotlib.pyplot as plt
import cv2
import time
from torch.profiler import profile, record_function, ProfilerActivity

from easydict import EasyDict
from utils.util_calib import *
def drawBBox(img, intrinsics, T_ldr2cam, gt_bbox, pred_bbox, pred_scores, calib, threshold=0.2, intrinsics_v1=None, T_ldr2cam_v1=None):
    img_out = img.copy()

    # ----- Helper: Compute 8 corners of 3D box -----
    def get_3d_corners(x, y, z, yaw, L, W, H):
        # L: length (x 방향)
        # W: width  (y 방향)
        # H: height (z 방향)
        # yaw: z축 기준 회전

        # 3D box corners in object coordinate
        x_c = L / 2
        y_c = W / 2
        z_c = H / 2

        corners = np.array([
            [ x_c,  y_c, -z_c],
            [ x_c, -y_c, -z_c],
            [-x_c, -y_c, -z_c],
            [-x_c,  y_c, -z_c],
            [ x_c,  y_c,  z_c],
            [ x_c, -y_c,  z_c],
            [-x_c, -y_c,  z_c],
            [-x_c,  y_c,  z_c]
        ])  # (8,3)

        # rotation
        R = np.array([
            [ np.cos(yaw), -np.sin(yaw), 0],
            [ np.sin(yaw),  np.cos(yaw), 0],
            [ 0, 0, 1]
        ])

        rotated = corners @ R.T
        translated = rotated + np.array([x, y, z])

        return translated  # (8,3)


    # ----- Helper: LiDAR → Camera → Image projection -----
    def project_to_image(pts_3d, intrinsics, T_ldr2cam):
        pts_h = np.concatenate([pts_3d, np.ones((pts_3d.shape[0], 1))], axis=1)  # (8,4)

        pts_cam = (T_ldr2cam @ pts_h.T).T  # (8,3)

        # must be in front of camera
        if np.any(pts_cam[:,2] <= 0):
            return None, False
        pts_img = (intrinsics @ pts_cam.T).T  # (8,3)
        pts_img = pts_img[:, :2] / pts_img[:, 2:3]

        return pts_img, True


    # ----- Helper: draw polygon lines -----
    def draw_box(img, corners_2d, color):
        corners_2d = corners_2d.astype(int)

        edges = [
            (0,1), (1,2), (2,3), (3,0),
            (4,5), (5,6), (6,7), (7,4),
            (0,4), (1,5), (2,6), (3,7)
        ]

        for s, e in edges:
            cv2.line(img, tuple(corners_2d[s]), tuple(corners_2d[e]), color, 2)

        return img


    # =====================================================
    # 1. Draw GT BBoxes (blue) — already center based
    # =====================================================
    for obj in gt_bbox:
        _, params, _, _ = obj
        x, y, z, yaw, L, W, H = params

        # calib shift
        x -= calib[0]
        y -= calib[1]
        z -= calib[2]

        corners_3d = get_3d_corners(x, y, z, yaw, L, W, H)
        pts_2d, valid = project_to_image(corners_3d, intrinsics, T_ldr2cam)
        if valid:
            img_out = draw_box(img_out, pts_2d, (255, 0, 0))  # blue


    # =====================================================
    # 2. Draw Pred BBoxes (yellow) — FIXED bottom center issue!!!
    # =====================================================
    for i in range(pred_bbox.shape[0]):
        if pred_scores[i] < threshold:
            continue

        x, y, z, L, W, H, yaw = pred_bbox[i]

        # calib shift
        x -= calib[0]
        y -= calib[1]
        z -= calib[2]

        # 🚨 bottom-center → box center 변환 (핵심 수정)
        # z = z + H / 2.0

        corners_3d = get_3d_corners(x, y, z, yaw, L, W, H)
        if intrinsics_v1 is not None and T_ldr2cam is not None:
            pts_2d, valid = project_to_image(corners_3d, intrinsics_v1, T_ldr2cam_v1)
        else:
            pts_2d, valid = project_to_image(corners_3d, intrinsics, T_ldr2cam)
        if valid:
            img_out = draw_box(img_out, pts_2d, (0, 255, 255))  # yellow

    return img_out


def show_projected_point_cloud_yml_Qual1(img, pcd, list_params, undistort=True):
    img_size = list_params[0]
    intrinsics = list_params[1].copy()
    distortion = list_params[2].copy()
    T_ldr2cam = list_params[3].copy()    
    
    img_process = img

    # ------------------------
    # 1) Undistortion (Optional)
    # ------------------------
    if undistort:
        ncm, _ = cv2.getOptimalNewCameraMatrix(intrinsics, distortion, img_size, alpha=0.0)
        
        for j in range(3):
            for i in range(3):
                intrinsics[j,i] = ncm[j, i]

        map_x, map_y = cv2.initUndistortRectifyMap(
            intrinsics, distortion, None, ncm, img_size, cv2.CV_32FC1
        )
        
        img_process = cv2.remap(img_process, map_x, map_y, cv2.INTER_LINEAR)

    # ------------------------
    # 2) Projection Matrix
    # ------------------------
    T_cam2pix = np.insert(np.insert(intrinsics, 3, [0,0,0], axis=1), 3, [0,0,0,1], axis=0)
    T_ldr2cam = np.insert(T_ldr2cam, 3, [0,0,0,1], axis=0)
    T_ldr2pix = T_cam2pix @ T_ldr2cam

    # ------------------------
    # 3) LiDAR → Pixel transform
    # ------------------------
    pc_ldr = (np.insert(pcd[:,:3], 3, [1], axis=1)).T
    pc_cam = T_ldr2pix @ pc_ldr
    pc_cam[:2,:] /= pc_cam[2,:]

    img_h, img_w, _ = img_process.shape

    pc_cam = (pc_cam.T)[:,:3]
    pc_cam = pc_cam[np.where(
        (pc_cam[:,0]>=0) & (pc_cam[:,0]<img_w) &
        (pc_cam[:,1]>=0) & (pc_cam[:,1]<img_h) &
        (pc_cam[:,2]>3)
    )]

    # ------------------------
    # 4) Overlay 준비
    # ------------------------
    overlay = np.zeros_like(img_process, dtype=np.uint8)
    alpha_mask = np.zeros((img_h, img_w), dtype=np.float32)

    if pc_cam.size > 0:
        pts = np.round(pc_cam[:, :2]).astype(np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, img_w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, img_h - 1)

        depth_inv = 1.0 / np.clip(pc_cam[:, 2], 1e-3, None)
        depth_norm = cv2.normalize(depth_inv, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        colors = cv2.applyColorMap(depth_norm, cv2.COLORMAP_RAINBOW).reshape(-1, 3)
        radius = 1
        alpha_val = 0.3
        for (x, y), color in zip(pts, colors):
            cv2.circle(overlay, (x, y), radius, color.tolist(), thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(alpha_mask, (x, y), radius, alpha_val, thickness=-1, lineType=cv2.LINE_AA)

    alpha_mask = np.clip(alpha_mask[..., None], 0.0, 1.0)

    # ------------------------
    # 5) Blend
    # ------------------------
    img_blended = (
        overlay.astype(np.float32) * alpha_mask +
        img_process.astype(np.float32) * (1.0 - alpha_mask)
    ).astype(np.uint8)

    # ------------------------
    # 6) numpy 형태로 반환
    # ------------------------
    return img_blended, img_process


class PipelineDetection_v1_0():
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
        self.init_img_backbone_pretrained()

        # Load checkpoints and freeze modules BEFORE DDP wrapping
        # This ensures frozen parameters are set before DDP sees them
        uem_cfg = self.cfg.MODEL.get('UEM_BACKBONE_3D', None)
        uem_pretrained = bool(uem_cfg and uem_cfg.get('PRETRAINED', False))

        if self.cfg.GENERAL.FINETUNE.IS_FINETUNE or uem_pretrained:
            if self.mode == 'train':
                self.loadCheckpoint()

        # Now wrap with DDP after freezing parameters
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

        self.cfg_dataset_ver2 = self.cfg.get('cfg_dataset_ver2', False)
        self.get_loss_from = self.cfg.get('get_loss_from', 'head')
        self.optim_fastai = True \
            if self.cfg.OPTIMIZER.NAME in ['adam_onecycle', 'adam_cosineanneal'] else False
        self.grad_norm_clip = self.cfg.OPTIMIZER.get('GRAD_NORM_CLIP', -1)

        # Vis
        #self.set_vis()
        
        # self.show_pline_description()

    def init_img_backbone_pretrained(self):
        """
        Initialize image backbone pretrained weights only in train mode.
        This is needed because custom skeletons are plain nn.Module and do not
        automatically trigger MMDet-style init hooks.
        """
        if self.mode != 'train':
            return

        img_backbone = getattr(self.network, 'img_backbone', None)
        if img_backbone is None or (not hasattr(img_backbone, 'init_weights')):
            if self.rank == 0:
                print('* img_backbone init skipped: no init_weights()')
            return

        with torch.no_grad():
            before = None
            for _, param in img_backbone.named_parameters():
                before = param.detach().float().sum().item()
                break

            img_backbone.init_weights()

            after = None
            for _, param in img_backbone.named_parameters():
                after = param.detach().float().sum().item()
                break

        if self.rank == 0:
            changed = (before is not None) and (after is not None) and (before != after)
            print(f'* img_backbone pretrained init done (changed={changed}, before={before}, after={after})')

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
        if self.rank == 0:
            print('* Start resume, path_state_dict =  ', path_state_dict)
        # Load to CPU first for proper multi-GPU handling
        state_dict = torch.load(path_state_dict, map_location='cpu')

        try:
            self.epoch_start = epoch + 1
            target_net = self.network.module if self.dist else self.network
            target_net.load_state_dict(state_dict['model_state_dict'])
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            self.log_iter_start = state_dict['idx_log_iter']
            if self.rank == 0:
                print(f'* Network & Optimizer are loaded / Resume epoch is {epoch} / Start from {self.epoch_start} ...')
        except Exception as e:
            print(f"* Exception error (Pipeline) on rank {self.rank}: {e}")
            traceback.print_exc()
            exit()

        if ('scheduler_state_dict' in state_dict.keys()) and (not (self.scheduler is None)):
            self.scheduler.load_state_dict(state_dict['scheduler_state_dict'])
            if self.rank == 0:
                print('* Scheduler is loaded')
        else:
            if self.rank == 0:
                print('* Scheduler is started from vanilla')

        ### Copy logging folder ###
        # list_copy_dirs = ['train_epoch', 'train_iter', 'test', 'test_kitti']
        list_copy_dirs = ['train_epoch', 'train_iter'] # we have nothing to test
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
        * Note: This function is called BEFORE DDP wrapping, so self.network is not wrapped yet
        """

        # Note: loadCheckpoint is called before DDP wrapping, so access self.network directly
        target_net = self.network
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
    def vis_infer(self, sample_indices, conf_thr=0.7, is_nms=True, vis_mode=['lpc', 'spcube', 'cube'], is_train=False):
        '''
        * sample_indices: e.g. [0, 1, 2, 3, 4]
        * assume batch_size = 1 for convenience
        * vis_mode (TBD)
        '''
        
        self.network.eval()
        
        if is_train:
            dataset_loaded = self.dataset_train
        else:
            dataset_loaded = self.dataset_test
        subset = Subset(dataset_loaded, sample_indices)
        if self.dist:
            data_loader = torch.utils.data.DataLoader(subset,
                    batch_size = 1, shuffle = False,
                    collate_fn = self.dataset_test.collate_fn,
                    num_workers = self.cfg.OPTIMIZER.NUM_WORKERS, sampler=self.test_sampler)
        else:
            data_loader = torch.utils.data.DataLoader(subset,
                    batch_size = 1, shuffle = False,
                    collate_fn = self.dataset_test.collate_fn,
                    num_workers = self.cfg.OPTIMIZER.NUM_WORKERS)
        for dict_datum in data_loader:
            dict_out = self.network(dict_datum)
            dict_out = self.network.list_modules[-1].get_nms_pred_boxes_for_single_sample(dict_out, conf_thr, is_nms)

            ### Vis data ###
            pc_lidar = dict_datum['ldr64']
            # rdr_spcube = dict_datum['rdr_sparse_cube']
            # rdr_cube = dict_datum['rdr_cube']
            ### Vis data ###

            ### Labels ###
            labels = dict_out['label'][0]
            list_obj_label = []
            for label_obj in labels:
                cls_name, cls_id, (xc, yc, zc, rot, xl, yl, zl), obj_idx = label_obj
                obj = Object3D(xc, yc, zc, xl, yl, zl, rot)
                list_obj_label.append(obj)
            ### Labels ###

            ### Preds: post processing bbox ###
            list_obj_pred = []
            list_cls_pred = []
            if dict_datum['pp_num_bbox'] == 0:
                pass
            else:
                pp_cls = dict_datum['pp_cls']
                for idx_pred, pred_obj in enumerate(dict_datum['pp_bbox']):
                    conf_score, xc, yc, zc, xl, yl, zl, rot = pred_obj
                    obj = Object3D(xc, yc, zc, xl, yl, zl, rot)
                    list_obj_pred.append(obj)
                    list_cls_pred.append('Sedan')
                    # list_cls_pred.append(self.dict_cls_id_to_name[pp_cls[idx_pred]])
            ### Preds: post processing bbox ###

            ### Vis for open3d ###
            lines = [[0, 1], [1, 2], [2, 3], [0, 3],
                    [4, 5], [6, 7], #[5, 6],[4, 7],
                    [0, 4], [1, 5], [2, 6], [3, 7],
                    [0, 2], [1, 3], [4, 6], [5, 7]]
            colors_label = [[0, 0, 0] for _ in range(len(lines))]
            list_line_set_label = []
            list_line_set_pred = []
            for label_obj in list_obj_label:
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(label_obj.corners)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.colors = o3d.utility.Vector3dVector(colors_label)
                list_line_set_label.append(line_set)
            
            for idx_pred, pred_obj in enumerate(list_obj_pred):
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(pred_obj.corners)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                # colors_pred = [self.dict_cls_name_to_rgb[list_cls_pred[idx_pred]] for _ in range(len(lines))]
                colors_pred = [[1.,0.,0.] for _ in range(len(lines))]
                line_set.colors = o3d.utility.Vector3dVector(colors_pred)
                list_line_set_pred.append(line_set)
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc_lidar[:, :3])
            o3d.visualization.draw_geometries([pcd] + list_line_set_label + list_line_set_pred)
            ### Vis for open3d ###

        return list_obj_label, list_obj_pred

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
        from utils.kitti_eval.eval import get_official_eval_result

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

    def validate_kitti_conditional(self, epoch=None, list_conf_thr=None, is_subset=False, is_print_memory=False, savevis = False):
        
        def has_valid_boxes(annos):
            if not annos:
                return False
            for anno in annos:
                bbox = anno.get('bbox', None)
                if bbox is not None and len(bbox) > 0:
                    return True
            return False
        
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
        
        #road_cond_list = ['urban', 'highway', 'countryside', 'alleyway', 'parkinglots', 'shoulder', 'mountain', 'university']
        #time_cond_list = ['day', 'night']
        road_cond_list = []
        time_cond_list = []
        weather_cond_list = ['normal', 'overcast', 'fog', 'rain', 'sleet', 'lightsnow', 'heavysnow', 'unnormal']

        # Check is_validate with small dataset
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
            for conf_thr in list_conf_thr:
                os.makedirs(os.path.join(path_dir, f'{conf_thr}'), exist_ok=True)

                os.makedirs(os.path.join(path_dir, f'{conf_thr}', 'all'), exist_ok=True)
                with open(path_dir + f'/{conf_thr}/' + 'all/val.txt', 'w') as f:
                    f.write('')

                for road_cond in road_cond_list:
                    os.makedirs(os.path.join(path_dir, f'{conf_thr}', road_cond), exist_ok=True)
                    with open(path_dir + f'/{conf_thr}/' + road_cond + '/val.txt', 'w') as f:
                        f.write('')

                for time_cond in time_cond_list:
                    os.makedirs(os.path.join(path_dir, f'{conf_thr}', time_cond), exist_ok=True)
                    with open(path_dir + f'/{conf_thr}/' + time_cond + '/val.txt', 'w') as f:
                        f.write('')

                for weather_cond in weather_cond_list:
                    os.makedirs(os.path.join(path_dir, f'{conf_thr}', weather_cond), exist_ok=True)
                    with open(path_dir + f'/{conf_thr}/' + weather_cond + '/val.txt', 'w') as f:
                        f.write('')

                pred_dir_list = []
                label_dir_list = []
                desc_dir_list = []
                split_path_list = []

                ### For All Conditions ###
                preds_dir = os.path.join(path_dir, f'{conf_thr}', 'all', 'preds')
                labels_dir = os.path.join(path_dir, f'{conf_thr}', 'all', 'gts')
                desc_dir = os.path.join(path_dir, f'{conf_thr}', 'all', 'desc')
                list_dir = [preds_dir, labels_dir, desc_dir]
                split_path = path_dir + f'/{conf_thr}/' + 'all/val.txt'

                for temp_dir in list_dir:
                    os.makedirs(temp_dir, exist_ok=True)

                pred_dir_list.append(preds_dir)
                label_dir_list.append(labels_dir)
                desc_dir_list.append(desc_dir)
                split_path_list.append(split_path)
                                
                ### For Specific Conditions ###
                for road_cond in road_cond_list:
                    preds_dir = os.path.join(path_dir, f'{conf_thr}', road_cond, 'preds')
                    labels_dir = os.path.join(path_dir, f'{conf_thr}', road_cond, 'gts')
                    desc_dir = os.path.join(path_dir, f'{conf_thr}', road_cond, 'desc')
                    list_dir = [preds_dir, labels_dir, desc_dir]
                    split_path = path_dir + f'/{conf_thr}/' + road_cond +'/val.txt'
                    
                    for temp_dir in list_dir:
                        os.makedirs(temp_dir, exist_ok=True)
                    
                    pred_dir_list.append(preds_dir)
                    label_dir_list.append(labels_dir)
                    desc_dir_list.append(desc_dir)
                    split_path_list.append(split_path)
                
                for time_cond in time_cond_list:
                    preds_dir = os.path.join(path_dir, f'{conf_thr}', time_cond, 'preds')
                    labels_dir = os.path.join(path_dir, f'{conf_thr}', time_cond, 'gts')
                    desc_dir = os.path.join(path_dir, f'{conf_thr}', time_cond, 'desc')
                    list_dir = [preds_dir, labels_dir, desc_dir]
                    split_path = path_dir + f'/{conf_thr}/' + time_cond +'/val.txt'
                    
                    for temp_dir in list_dir:
                        os.makedirs(temp_dir, exist_ok=True)

                    pred_dir_list.append(preds_dir)
                    label_dir_list.append(labels_dir)
                    desc_dir_list.append(desc_dir)
                    split_path_list.append(split_path)
                
                for weather_cond in weather_cond_list:
                    preds_dir = os.path.join(path_dir, f'{conf_thr}', weather_cond, 'preds')
                    labels_dir = os.path.join(path_dir, f'{conf_thr}', weather_cond, 'gts')
                    desc_dir = os.path.join(path_dir, f'{conf_thr}', weather_cond, 'desc')
                    list_dir = [preds_dir, labels_dir, desc_dir]
                    split_path = path_dir + f'/{conf_thr}/' + weather_cond +'/val.txt'
                    
                    for temp_dir in list_dir:
                        os.makedirs(temp_dir, exist_ok=True)

                    pred_dir_list.append(preds_dir)
                    label_dir_list.append(labels_dir)
                    desc_dir_list.append(desc_dir)
                    split_path_list.append(split_path)
            with torch.no_grad():
                # Creating gts and preds txt files for evaluation
                # for idx_datum, dict_datum in enumerate(data_loader):
                for idx_datum, dict_datum in enumerate(tqdm(data_loader, ncols=60, dynamic_ncols=False)):
                    if is_subset & (idx_datum >= self.val_num_subset):
                        break
                    ###########################################################
                    ### For debugging, radar shape problem while evaluating ###
                    ###                                                     ###
                    

                    # print(f"\r{idx_datum}", end="")
                    # if idx_datum == 10:
                        
                    #     break
                    # breakpoint()
                    
                    # print('pause here')
                    # idx_datum가 8645일 때  애러 voxel_features.shape는 [64]임
                    
                    
                    ###                                                     ###
                    ### For debugging, radar shape problem while evaluating ###
                    ###########################################################
                    dict_out = self.network(dict_datum) # inference
                    if savevis:
                        vis.append(save_frame_vis(dict_out['pred_dicts'][0], dict_datum))
                    is_feature_inferenced = True

                    if is_print_memory:
                        print('max_memory: ', torch.cuda.max_memory_allocated(device='cuda'))
                    idx_name = str(idx_datum).zfill(6)
                    
                    road_cond_tag, time_cond_tag, weather_cond_tag = \
                        dict_out['meta'][0]['desc']['road_type'], dict_out['meta'][0]['desc']['capture_time'], dict_out['meta'][0]['desc']['climate']
                    # print(dict_out['desc'][0])
                    ### for every conf in list_conf_thr ###
                    for conf_thr in list_conf_thr:
                        ### For All Conditions ###
                        preds_dir = os.path.join(path_dir, f'{conf_thr}', 'all', 'preds')
                        labels_dir = os.path.join(path_dir, f'{conf_thr}', 'all', 'gts')
                        desc_dir = os.path.join(path_dir, f'{conf_thr}', 'all', 'desc')
                        list_dir = [preds_dir, labels_dir, desc_dir]
                        split_path = path_dir + f'/{conf_thr}/' + 'all/val.txt'

                        preds_dir_road = os.path.join(path_dir, f'{conf_thr}', road_cond_tag, 'preds')
                        labels_dir_road = os.path.join(path_dir, f'{conf_thr}', road_cond_tag, 'gts')
                        desc_dir_road = os.path.join(path_dir, f'{conf_thr}', road_cond_tag, 'desc')
                        split_path_road =path_dir + f'/{conf_thr}/' + road_cond_tag + '/val.txt'

                        preds_dir_time = os.path.join(path_dir, f'{conf_thr}', time_cond_tag, 'preds')
                        labels_dir_time = os.path.join(path_dir, f'{conf_thr}', time_cond_tag, 'gts')
                        desc_dir_time = os.path.join(path_dir, f'{conf_thr}', time_cond_tag, 'desc')
                        split_path_time = path_dir + f'/{conf_thr}/' + time_cond_tag + '/val.txt'

                        preds_dir_weather = os.path.join(path_dir, f'{conf_thr}', weather_cond_tag, 'preds')
                        labels_dir_weather = os.path.join(path_dir, f'{conf_thr}', weather_cond_tag, 'gts')
                        desc_dir_weather = os.path.join(path_dir, f'{conf_thr}', weather_cond_tag, 'desc')
                        split_path_weather = path_dir + f'/{conf_thr}/' + weather_cond_tag + '/val.txt'
                        
                        if weather_cond_tag != 'normal':
                            preds_dir_exweather = os.path.join(path_dir, f'{conf_thr}', 'unnormal', 'preds')
                            labels_dir_exweather = os.path.join(path_dir, f'{conf_thr}', 'unnormal', 'gts')
                            desc_dir_exweather = os.path.join(path_dir, f'{conf_thr}', 'unnormal', 'desc')
                            split_path_exweather = path_dir + f'/{conf_thr}/' + 'unnormal' + '/val.txt'
                            os.makedirs(desc_dir_exweather, exist_ok=True)
                            os.makedirs(preds_dir_exweather, exist_ok=True)
                            os.makedirs(labels_dir_exweather, exist_ok=True)

                        os.makedirs(labels_dir_road, exist_ok=True)
                        os.makedirs(labels_dir_time, exist_ok=True)
                        os.makedirs(labels_dir_weather, exist_ok=True)
                        
                        os.makedirs(desc_dir_road, exist_ok=True)
                        os.makedirs(desc_dir_time, exist_ok=True)
                        os.makedirs(desc_dir_weather, exist_ok=True)
                        
                        os.makedirs(preds_dir_road, exist_ok=True)
                        os.makedirs(preds_dir_time, exist_ok=True)
                        os.makedirs(preds_dir_weather, exist_ok=True)
                        

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

                        if len(dict_out_current['kitti_gt']) == 0: # not eval emptry label
                            pass
                        else:
                            ### Gt ###
                            for idx_label, label in enumerate(dict_out_current['kitti_gt']):
                                if idx_label == 0:
                                    mode = 'w'
                                else:
                                    mode = 'a'

                                with open(labels_dir + '/' + idx_name + '.txt', mode) as f:
                                    f.write(label+'\n')
                                with open(labels_dir_road + '/' + idx_name + '.txt', mode) as f:
                                    f.write(label+'\n')
                                with open(labels_dir_time + '/' + idx_name + '.txt', mode) as f:
                                    f.write(label+'\n')
                                with open(labels_dir_weather + '/' + idx_name + '.txt', mode) as f:
                                    f.write(label+'\n')
                                if weather_cond_tag != 'normal': 
                                    with open(labels_dir_exweather + '/' + idx_name + '.txt', mode) as f:
                                        f.write(label+'\n')

                            ### Process description ###
                            with open(desc_dir + '/' + idx_name + '.txt', 'w') as f:
                                f.write(dict_out_current['kitti_desc'])
                            with open(desc_dir_road + '/' + idx_name + '.txt', 'w') as f:
                                f.write(dict_out_current['kitti_desc'])
                            with open(desc_dir_time + '/' + idx_name + '.txt', 'w') as f:
                                f.write(dict_out_current['kitti_desc'])
                            with open(desc_dir_weather + '/' + idx_name + '.txt', 'w') as f:
                                f.write(dict_out_current['kitti_desc'])
                            if weather_cond_tag != 'normal': 
                                with open(desc_dir_exweather + '/' + idx_name + '.txt', 'w') as f:
                                    f.write(dict_out_current['kitti_desc'])

                            ### Process description ###
                            if len(dict_out_current['kitti_pred']) == 0:
                                with open(preds_dir + '/' + idx_name + '.txt', mode) as f:
                                    f.write('\n')
                                with open(preds_dir_road + '/' + idx_name + '.txt', mode) as f:
                                    f.write('\n')
                                with open(preds_dir_time + '/' + idx_name + '.txt', mode) as f:
                                    f.write('\n')
                                with open(preds_dir_weather + '/' + idx_name + '.txt', mode) as f:
                                    f.write('\n')
                                if weather_cond_tag != 'normal': 
                                    with open(preds_dir_exweather + '/' + idx_name + '.txt', mode) as f:
                                        f.write('\n')
                            else:
                                for idx_pred, pred in enumerate(dict_out_current['kitti_pred']):
                                    if idx_pred == 0:
                                        mode = 'w'
                                    else:
                                        mode = 'a'

                                    with open(preds_dir + '/' + idx_name + '.txt', mode) as f:
                                        f.write(pred+'\n')
                                    with open(preds_dir_road + '/' + idx_name + '.txt', mode) as f:
                                        f.write(pred+'\n')
                                    with open(preds_dir_time + '/' + idx_name + '.txt', mode) as f:
                                        f.write(pred+'\n')
                                    with open(preds_dir_weather + '/' + idx_name + '.txt', mode) as f:
                                        f.write(pred+'\n')
                                    if weather_cond_tag != 'normal': 
                                        with open(preds_dir_exweather + '/' + idx_name + '.txt', mode) as f:
                                            f.write(pred+'\n')
                            
                            str_log = idx_name + '\n'
                            with open(split_path, 'a') as f:
                                f.write(str_log)
                            with open(split_path_road, 'a') as f:
                                f.write(str_log)
                            with open(split_path_time, 'a') as f:
                                f.write(str_log)
                            with open(split_path_weather, 'a') as f:
                                f.write(str_log)
                            if weather_cond_tag != 'normal': 
                                with open(split_path_exweather, 'a') as f:
                                    f.write(str_log)
                                
                    # free memory (Killed error, checked with htop)
                    if 'pointer' in dict_datum.keys():
                        for dict_item in dict_datum['pointer']:
                            for k in dict_item.keys():
                                if k != 'meta':
                                    dict_item[k] = None
                    for temp_key in dict_datum.keys():
                        dict_datum[temp_key] = None
            #         tqdm_bar.update(1)
            # tqdm_bar.close()
            if savevis:
                with open(path_dir +  '/lr_vis.pkl', 'wb') as f:
                        pickle.dump(vis, f)
                print("vis save in ",(path_dir + 'lr_vis.pkl'))
            ### Validate per conf ###
            all_condition_list = ['all'] + road_cond_list + time_cond_list + weather_cond_list
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

                    from utils.kitti_eval.eval import get_official_eval_result

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



    def validate_kitti_distance(self, epoch=None, list_conf_thr=None, is_subset=False, is_print_memory=False, savevis = False, subset_num=None):
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
        
        distance_cond_list = ['close', 'mid', 'far']
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
            condition_names = ['all'] + distance_cond_list
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

                        gt_distance_map = {cond: [] for cond in distance_cond_list}
                        pred_distance_map = {cond: [] for cond in distance_cond_list}

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

                        write_kitti_lines(os.path.join(labels_dir, idx_name + '.txt'), dict_out_current['kitti_gt'])
                        write_kitti_lines(os.path.join(preds_dir, idx_name + '.txt'), dict_out_current['kitti_pred'])
                        with open(os.path.join(desc_dir, idx_name + '.txt'), 'w') as f:
                            f.write(dict_out_current['kitti_desc'])
                        with open(split_path, 'a') as f:
                            f.write(idx_name + '\n')

                        for distance_cond in distance_cond_list:
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
            all_condition_list = ['all'] + distance_cond_list
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

                    from utils.kitti_eval.eval import get_official_eval_result

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



    def validateComputation(self, epoch=None):
        """
        GFLOPS, FPS, # parameters, VRAM 측정
        """
        self.network.eval()
        self.network.training=False
        use_cuda = torch.cuda.is_available()
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

        # parameter stats
        num_params = sum(p.numel() for p in self.network.parameters())
        num_trainable_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)

        # fetch one sample for FLOPs measurement
        try:
            sample_batch = next(iter(data_loader))
        except StopIteration:
            print('* validateComputation: dataloader is empty.')
            return

        # measure GFLOPs using torch.profiler (per forward pass)
        gflops = None
        try:
            activities = [ProfilerActivity.CPU]
            if use_cuda:
                activities.append(ProfilerActivity.CUDA)
            with profile(activities=activities,
                         record_shapes=False,
                         profile_memory=False,
                         with_flops=True) as prof:
                with torch.no_grad():
                    with record_function('model_inference'):
                        self.network(sample_batch)
            total_flops = sum(evt.flops for evt in prof.key_averages() if evt.flops is not None)
            gflops = total_flops / 1e9
        except Exception as e:
            print(f'* GFLOPs profiling failed: {e}')
            gflops = None

        # FPS / latency / VRAM
        warmup_iters = 5
        measure_iters = 50
        total_time = 0.0
        counted = 0
        if use_cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        with torch.no_grad():
            for idx_datum, dict_datum in enumerate(tqdm(data_loader, ncols=60, dynamic_ncols=False)):
                if idx_datum < warmup_iters:
                    _ = self.network(dict_datum)
                    if use_cuda:
                        torch.cuda.synchronize()
                    continue
                if counted >= measure_iters:
                    break

                start_t = time.time()
                _ = self.network(dict_datum)
                if use_cuda:
                    torch.cuda.synchronize()
                total_time += (time.time() - start_t)
                counted += 1

        fps = (counted / total_time) if total_time > 0 else 0.0
        latency_ms = (total_time / counted * 1000.) if counted > 0 else 0.0
        peak_vram_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)) if use_cuda else 0.0

        # logging
        metrics_lines = [
            f'Total parameters: {num_params/1e6:.3f} M (trainable: {num_trainable_params/1e6:.3f} M)',
            f'GFLOPs / forward: {gflops:.3f}' if gflops is not None else 'GFLOPs / forward: N/A',
            f'FPS @batch1: {fps:.3f}',
            f'Latency @batch1 (ms): {latency_ms:.3f}',
            f'Peak VRAM (MB): {peak_vram_mb:.1f}',
            f'Warmup iters: {warmup_iters}, Measured iters: {counted}',
        ]

        print('='*20 + ' Computation Metrics ' + '='*20)
        for line in metrics_lines:
            print(line)

        path_log = getattr(self, 'path_log', None)
        if path_log is not None:
            os.makedirs(path_log, exist_ok=True)
            path_out = os.path.join(path_log, 'computation.txt')
            with open(path_out, 'w') as f:
                for line in metrics_lines:
                    f.write(line + '\n')
            print(f'* Metrics are saved to {path_out}')

    
    def visualizeConfMap(self, epoch=None, list_conf_thr=None, is_subset=False, is_print_memory=False, savevis = False):
        import torchvision.transforms.functional as TF
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        from PIL import Image
        # 0=red, 0.5=white, 1=blue
        red_white_blue_cmap = LinearSegmentedColormap.from_list(
            "RedWhiteBlue", ["red", "white", "blue"]
        )

        self.network.eval()
        self.network.training=False

        tqdm_bar = tqdm(total=len(self.dataset_test), desc='Test (Total): ')
        data_loader = torch.utils.data.DataLoader(self.dataset_test, batch_size = 1, shuffle = False, collate_fn = self.dataset_test.collate_fn, num_workers = self.cfg.OPTIMIZER.NUM_WORKERS)

        with torch.no_grad():
            for idx_datum, dict_datum in enumerate(data_loader):
                if idx_datum % 500 != 0:
                    tqdm_bar.update(1)
                    continue
                dict_out = self.network(dict_datum) # inference

                road_cond_tag, time_cond_tag, weather_cond_tag = dict_out['meta'][0]['desc']['road_type'], dict_out['meta'][0]['desc']['capture_time'], dict_out['meta'][0]['desc']['climate']
                conf_map = dict_datum['conf_map'][0, 0]   # [H, W]
                
                image_path = dict_datum['meta'][0]['path']['front']

                # 1) 원본 이미지 로드 (PIL -> tensor)
                image = Image.open(image_path).convert("RGB")
                image = TF.to_tensor(image)  # [C, 720, 2560]
                C, H_img, W_img = image.shape # [3, 720, 2560]
                image = image[:, 80:, :W_img//2]  # [3, 640, 1280]

                # 2) 이미지 downscale → (320, 640)
                image = TF.resize(image, [320, 640])  # [C, 320, 640]

                # 3) conf_map도 (320, 640)으로 upsample
                conf_map = conf_map.unsqueeze(0)  # [1, 40, 80]
                conf_map_up = TF.resize(conf_map, [320, 640])  # [1, 320, 640]
                conf_map_up = conf_map_up[0]  # [320, 640]

                # 4) numpy 변환
                resized_img = image.permute(1, 2, 0).cpu().numpy()  # [320, 640, 3]
                conf_map_np = conf_map_up.detach().cpu().numpy()

                # 5) 시각화
                fig, axs = plt.subplots(1, 2, figsize=(10, 5))

                # input image
                axs[0].imshow(resized_img)
                axs[0].set_title("Input Image")
                axs[0].axis("off")

                # confidence map (0=white, 1=red)
                # axs[1].imshow(conf_map_np, cmap="Reds", vmin=0, vmax=1)
                # axs[1].set_title("Confidence Map")
                # axs[1].axis("off")
                
                # confidence map (0=red, 0.5=white, 1=blue)
                axs[1].imshow(conf_map_np, cmap=red_white_blue_cmap, vmin=0, vmax=1)
                axs[1].set_title("Confidence Map")
                axs[1].axis("off")
                
                
                plt.tight_layout()

                # 6) 저장
                filename = f"{idx_datum:05d}_{road_cond_tag}_{time_cond_tag}_{weather_cond_tag}.jpg"
                save_file = os.path.join(self.save_confmap_path, filename)
                plt.savefig(save_file, dpi=150, bbox_inches="tight")
                plt.close()
                

                tqdm_bar.update(1)
        tqdm_bar.close()




    def visualizeConfMap_v4(self, epoch=None, list_conf_thr=None, is_subset=False, is_print_memory=False, savevis = False):
        import torchvision.transforms.functional as TF
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        from PIL import Image
        # 0=red, 1=white
        # red_white_cmap = LinearSegmentedColormap.from_list(
        #     "RedWhite", ["red", "white"]
        # )
        red_half_white_cmap = LinearSegmentedColormap.from_list(
            "RedHalfWhite",
            [(0.0, "red"),
            (0.5, "white"),
            (1.0, "white")]
        )

        self.network.eval()
        self.network.training=False
        
        
        save_file_folder = os.path.join(self.save_confmap_path, str(epoch))
        os.makedirs(save_file_folder)
        # tqdm_bar = tqdm(total=len(self.dataset_test), desc='Test (Total): ')
        # data_loader = torch.utils.data.DataLoader(self.dataset_test, batch_size = 1, shuffle = False, collate_fn = self.dataset_test.collate_fn, num_workers = self.cfg.OPTIMIZER.NUM_WORKERS)

        with torch.no_grad():
            # for idx_datum, dict_datum in enumerate(data_loader):
            for idx_datum in range(0, len(self.dataset_test), 100):
                # if idx_datum % 500 != 0:
                #     tqdm_bar.update(1)
                #     continue
                sample = self.dataset_test[idx_datum]
                dict_datum = self.dataset_test.collate_fn([sample])
                dict_out = self.network(dict_datum) # inference

                road_cond_tag, time_cond_tag, weather_cond_tag = dict_out['meta'][0]['desc']['road_type'], dict_out['meta'][0]['desc']['capture_time'], dict_out['meta'][0]['desc']['climate']
                conf_map = dict_datum['conf_map'][0, 0]   # [H, W]
                
                image_path = dict_datum['meta'][0]['path']['front']

                # 1) 원본 이미지 로드 (PIL -> tensor)
                image = Image.open(image_path).convert("RGB")
                image = TF.to_tensor(image)  # [C, 720, 2560]
                C, H_img, W_img = image.shape # [3, 720, 2560]
                image = image[:, 80:, :W_img//2]  # [3, 640, 1280]

                # 2) 이미지 downscale → (320, 640)
                image = TF.resize(image, [320, 640])  # [C, 320, 640]

                # 3) conf_map도 (320, 640)으로 upsample
                conf_map = conf_map.unsqueeze(0)  # [1, 40, 80]
                conf_map_up = TF.resize(conf_map, [320, 640])  # [1, 320, 640]
                conf_map_up = conf_map_up[0]  # [320, 640]

                # 4) numpy 변환
                resized_img = image.permute(1, 2, 0).cpu().numpy()  # [320, 640, 3]
                conf_map_np = conf_map_up.detach().cpu().numpy()

                # 5) 시각화
                fig, axs = plt.subplots(1, 2, figsize=(10, 5))

                # input image
                axs[0].imshow(resized_img)
                axs[0].set_title("Input Image")
                # axs[0].axis("off")
                axs[0].set_xticks([])  # 눈금 제거
                axs[0].set_yticks([])  # 눈금 제거
                for spine in axs[0].spines.values():
                    spine.set_edgecolor("black")
                    spine.set_linewidth(0.8)
                    
                    
                # confidence map (0=red, 0.5=white, 1=blue)
                # axs[1].imshow(conf_map_np, cmap=red_white_blue_cmap, vmin=0, vmax=1)
                axs[1].imshow(conf_map_np, cmap=red_half_white_cmap, vmin=0, vmax=1)
                axs[1].set_title("Confidence Map")
                # axs[1].axis("off")
                axs[1].set_xticks([])
                axs[1].set_yticks([])
                for spine in axs[1].spines.values():
                    spine.set_edgecolor("black")
                    spine.set_linewidth(0.8)
                
                plt.tight_layout()

                # 6) 저장
                filename = f"{idx_datum:05d}_{road_cond_tag}_{time_cond_tag}_{weather_cond_tag}.jpg"
                save_file = os.path.join(save_file_folder, filename)
                plt.savefig(save_file, dpi=150, bbox_inches="tight", pad_inches=0.05)
                plt.close()
                print(f'saved at: {save_file}')


    def visualizeSuppleVanilla(self, epoch=None, is_subset=False, viz_interval=100):
        import torchvision.transforms.functional as TF
        import torch.nn.functional as F
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        from PIL import Image
        import os.path as osp
        import yaml


        self.network.eval()
        self.network.training=False
        
        # 1. l4dr(vanilla) 모델 부르기
        l4dr_vanilla_config = './logs/L4DR_cam/scaled_size/L4DR_cam.yml'
        model_path = './logs/L4DR_cam/scaled_size/models/model_34.pt'
        if not hasattr(self, '_l4dr_model'):
            cfg_l4dr = EasyDict()
            cfg_l4dr.ROOT_DIR = getattr(self.cfg, 'ROOT_DIR', None)
            cfg_l4dr.LOCAL_RANK = getattr(self.cfg, 'LOCAL_RANK', 0)
            cfg_l4dr = cfg_from_yaml_file(l4dr_vanilla_config, cfg_l4dr)

            class _TmpPline:
                pass
            tmp_pline = _TmpPline()
            tmp_pline.cfg = cfg_l4dr

            self._l4dr_model = build_network(tmp_pline).cuda()
            state_dict = torch.load(model_path, map_location='cpu')
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            self._l4dr_model.load_state_dict(state_dict, strict=False)
            self._l4dr_model.eval()
            self._l4dr_model.training = False
        
        
        save_file_folder = os.path.join(self.save_confmap_path, str(epoch))
        os.makedirs(save_file_folder)
        # tqdm_bar = tqdm(total=len(self.dataset_test), desc='Test (Total): ')
        # data_loader = torch.utils.data.DataLoader(self.dataset_test, batch_size = 1, shuffle = False, collate_fn = self.dataset_test.collate_fn, num_workers = self.cfg.OPTIMIZER.NUM_WORKERS)
        self.cam_calib_sequence = "./resources/cam_calib/calib_seq_v2"
        with torch.no_grad():
            # for idx_datum, dict_datum in enumerate(data_loader):
            for idx_datum in range(0, len(self.dataset_test), viz_interval):
                # if idx_datum % 500 != 0:
                #     tqdm_bar.update(1)
                #     continue
                sample = self.dataset_test[idx_datum]
                dict_datum = self.dataset_test.collate_fn([sample])
                dict_out = self.network(dict_datum) # NOTE 일단 주석처리해서 빨리 image + lidar projected뽑기

                
                road_cond_tag = sample['meta']['desc']['road_type']
                weather_cond_tag =  sample['meta']['desc']['climate']
                
                
                # 1. 이미지에 lidar projection
                # 🛠️ camera parameters
                seq = sample['meta']['seq']
                dir_cam_calib = self.cam_calib_sequence + '/seq_'+str(int(seq)).zfill(2)
                dict_cam_calib = dict()
                list_yml = ['cam_1.yml']
                for yml_file_name in list_yml:
                    key_name = yml_file_name.split('.')[0].split('_')[1] # '1'
                    with open(osp.join(dir_cam_calib, yml_file_name), 'r') as yml_file:
                        dict_temp = yaml.safe_load(yml_file)
                    dict_cam_calib[key_name] = get_matrices_from_dict_calib(dict_temp) # img_size, intrinsics, distortion, T_ldr2cam

                # 📷 image
                image_path = sample['meta']['path']['front']
                img = cv2.imread(image_path)  # BGR 형식의 numpy array로 로드됨
                H_img, W_img, C = img.shape # (720, 2560, 3)
                img = img[:, :W_img//2, :]  # (720, 1280, 3)
                
                # 🔫 lidar
                ldr64 = sample['ldr64']
                calib = sample['meta']['calib'] # [-2.54, 0.3, 0.7]
                ldr64[:, :3] -= np.array(calib)

                # projection 수행
                proj_img, img_undistored = show_projected_point_cloud_yml_Qual1(img, ldr64, dict_cam_calib['1'], undistort=True) # (720, 1280, 3)
                proj_img = proj_img[80:, :, :] # (640, 1280, 3)
                # cv2.imwrite(f"./proj_img_test_{idx_datum}.png", proj_img)
                # breakpoint()
                
                
                # 2. l4dr 결과랑 gt를 이미지에 그리기
                intrinsics = dict_cam_calib['1'][1]
                T_ldr2cam = dict_cam_calib['1'][3]
                gt_bbox = sample['meta']['label']
                threshold = 0.3
                dict_datum_l4dr = self.dataset_test.collate_fn([sample])
                dict_out_l4dr = self._l4dr_model(dict_datum_l4dr)
                pred_bbox_l4dr = dict_out_l4dr['pred_dicts'][0]['pred_boxes'].detach().cpu().numpy()
                pred_scores_l4dr = dict_out_l4dr['pred_dicts'][0]['pred_scores'].detach().cpu().numpy()
                result_l4dr = drawBBox(
                    img_undistored,
                    intrinsics,
                    T_ldr2cam,
                    gt_bbox,
                    pred_bbox_l4dr,
                    pred_scores_l4dr,
                    calib,
                    threshold
                )
                result_l4dr = result_l4dr[80:, :, :]
                
                
                # 3. ours 결과랑 gt를 이미지에 그리기
                pred_bbox = dict_out['pred_dicts'][0]['pred_boxes'].detach().cpu().numpy() # (n, 7)
                pred_scores = dict_out['pred_dicts'][0]['pred_scores'].detach().cpu().numpy() # (n,)
                # pred_labels = dict_out['pred_dicts'][0]['pred_labels'].detach().cpu().numpy() # (n,)
                result_ours = drawBBox(img_undistored, intrinsics, T_ldr2cam, gt_bbox, pred_bbox, pred_scores, calib, threshold) # (720, 1280, 3)
                result_ours = result_ours[80:, :, :] # (640, 1280, 3)
                # cv2.imwrite(f"check_{idx_datum}.png", result_ours)
                
                
                # 4. 이미지 저장
                side_by_side = np.concatenate([proj_img, result_l4dr, result_ours], axis=1)
                save_path = f"./viz_supple_vanilla/temp/{weather_cond_tag}_{idx_datum}.png"
                cv2.imwrite(save_path, side_by_side)
                print(save_path)
                
                

    def visualizeSuppleOriginalVanilla(self, epoch=None, is_subset=False, viz_interval=100):
        import torchvision.transforms.functional as TF
        import torch.nn.functional as F
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        from PIL import Image
        import os.path as osp
        import yaml


        self.network.eval()
        self.network.training=False
        
        
        
        

        
        
        
        # 1. l4dr(vanilla) 모델 부르기
        l4dr_vanilla_config = './logs/L4DR_cam/scaled_size/L4DR_cam.yml'
        model_path = './logs/L4DR_cam/scaled_size/models/model_34.pt'
        if not hasattr(self, '_l4dr_model'):
            cfg_l4dr = EasyDict()
            cfg_l4dr.ROOT_DIR = getattr(self.cfg, 'ROOT_DIR', None)
            cfg_l4dr.LOCAL_RANK = getattr(self.cfg, 'LOCAL_RANK', 0)
            cfg_l4dr = cfg_from_yaml_file(l4dr_vanilla_config, cfg_l4dr)
            class _TmpPline:
                pass
            tmp_pline = _TmpPline()
            tmp_pline.cfg = cfg_l4dr
            self._l4dr_model = build_network(tmp_pline).cuda()
            state_dict = torch.load(model_path, map_location='cpu')
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            self._l4dr_model.load_state_dict(state_dict, strict=False)
            self._l4dr_model.eval()
            self._l4dr_model.training = False
        
        
        
        
        
        

        # 2. l4dr(오리지널) 모델 부르기
        l4dr_config_ori = './configs/L4DR.yml'
        model_path_ori = './logs/L4DR/l4dr_4th_reproduction_single_gpu/models/model_10.pt'
        cfg_l4dr_ori = EasyDict()
        cfg_l4dr_ori.ROOT_DIR = getattr(self.cfg, 'ROOT_DIR', None)
        cfg_l4dr_ori.LOCAL_RANK = getattr(self.cfg, 'LOCAL_RANK', 0)
        cfg_l4dr_ori = cfg_from_yaml_file(l4dr_config_ori, cfg_l4dr_ori)
        class _TmpPline_ori:
            pass
        tmp_pline_ori = _TmpPline_ori()
        tmp_pline_ori.cfg = cfg_l4dr_ori
        self._l4dr_model_ori = build_network(tmp_pline_ori).cuda()
        state_dict_ori = torch.load(model_path_ori, map_location='cpu')
        if isinstance(state_dict_ori, dict) and 'model_state_dict' in state_dict_ori:
            state_dict_ori = state_dict_ori['model_state_dict']
        self._l4dr_model_ori.load_state_dict(state_dict_ori, strict=False)
        self._l4dr_model_ori.eval()
        self._l4dr_model_ori.training = False
        
        
        
        
        
        
        
        
        save_file_folder = os.path.join(self.save_confmap_path, str(epoch))
        os.makedirs(save_file_folder)
        self.cam_calib_sequence = "./resources/cam_calib/calib_seq_v2"
        with torch.no_grad():
            for idx_datum in range(0, len(self.dataset_test), viz_interval):

                sample = self.dataset_test[idx_datum]
                dict_datum = self.dataset_test.collate_fn([sample])
                dict_out = self.network(dict_datum) # NOTE 일단 주석처리해서 빨리 image + lidar projected뽑기

                
                weather_cond_tag =  sample['meta']['desc']['climate']
                
                
                # 1. 이미지에 lidar projection
                # 🛠️ camera parameters
                seq = sample['meta']['seq']
                dir_cam_calib = self.cam_calib_sequence + '/seq_'+str(int(seq)).zfill(2)
                dict_cam_calib = dict()
                list_yml = ['cam_1.yml']
                for yml_file_name in list_yml:
                    key_name = yml_file_name.split('.')[0].split('_')[1] # '1'
                    with open(osp.join(dir_cam_calib, yml_file_name), 'r') as yml_file:
                        dict_temp = yaml.safe_load(yml_file)
                    dict_cam_calib[key_name] = get_matrices_from_dict_calib(dict_temp) # img_size, intrinsics, distortion, T_ldr2cam
                with open(osp.join('./resources/cam_calib/common/', 'cam_front0.yml'), 'r') as yml_file:
                    dict_yaml_temp = yaml.safe_load(yml_file)
                dict_cam_calib_temp = get_matrices_from_dict_calib(dict_yaml_temp)

                # 📷 image
                image_path = sample['meta']['path']['front']
                img = cv2.imread(image_path)  # BGR 형식의 numpy array로 로드됨
                H_img, W_img, C = img.shape # (720, 2560, 3)
                img = img[:, :W_img//2, :]  # (720, 1280, 3)
                
                # 🔫 lidar
                ldr64 = sample['ldr64']
                calib = sample['meta']['calib'] # [-2.54, 0.3, 0.7]
                ldr64[:, :3] -= np.array(calib)

                # projection 수행
                proj_img, img_undistored = show_projected_point_cloud_yml_Qual1(img, ldr64, dict_cam_calib['1'], undistort=True) # (720, 1280, 3)
                proj_img = proj_img[80:, :, :] # (640, 1280, 3)
                
                
                
                
                
                
                # 2. l4dr(original) 결과랑 gt를 이미지에 그리기
                intrinsics = dict_cam_calib['1'][1]
                T_ldr2cam = dict_cam_calib['1'][3]
                intrinsics_v1 = dict_cam_calib_temp[1]
                T_ldr2cam_v1 = dict_cam_calib_temp[3]
                gt_bbox = sample['meta']['label']
                threshold = 0.3
                dict_datum_l4dr = self.dataset_test.collate_fn([sample])
                dict_out_l4dr = self._l4dr_model_ori(dict_datum_l4dr)
                pred_bbox_l4dr = dict_out_l4dr['pred_dicts'][0]['pred_boxes'].detach().cpu().numpy()
                pred_scores_l4dr = dict_out_l4dr['pred_dicts'][0]['pred_scores'].detach().cpu().numpy()
                result_l4dr_ori = drawBBox(
                    img_undistored,
                    intrinsics,
                    T_ldr2cam,
                    gt_bbox,
                    pred_bbox_l4dr,
                    pred_scores_l4dr,
                    calib,
                    threshold,
                    intrinsics_v1,
                    T_ldr2cam_v1,
                )
                result_l4dr_ori = result_l4dr_ori[80:, :, :]
                
                
                
                
                
                
                
                
                # 2. l4dr(vanilla) 결과랑 gt를 이미지에 그리기
                intrinsics = dict_cam_calib['1'][1]
                T_ldr2cam = dict_cam_calib['1'][3]
                intrinsics_v1 = dict_cam_calib_temp[1]
                T_ldr2cam_v1 = dict_cam_calib_temp[3]
                gt_bbox = sample['meta']['label']
                threshold = 0.3
                dict_datum_l4dr = self.dataset_test.collate_fn([sample])
                dict_out_l4dr = self._l4dr_model(dict_datum_l4dr)
                pred_bbox_l4dr = dict_out_l4dr['pred_dicts'][0]['pred_boxes'].detach().cpu().numpy()
                pred_scores_l4dr = dict_out_l4dr['pred_dicts'][0]['pred_scores'].detach().cpu().numpy()
                result_l4dr = drawBBox(
                    img_undistored,
                    intrinsics,
                    T_ldr2cam,
                    gt_bbox,
                    pred_bbox_l4dr,
                    pred_scores_l4dr,
                    calib,
                    threshold,
                    intrinsics_v1,
                    T_ldr2cam_v1,
                )
                result_l4dr = result_l4dr[80:, :, :]
                
                
                # 3. ours 결과랑 gt를 이미지에 그리기
                intrinsics = dict_cam_calib['1'][1]
                T_ldr2cam = dict_cam_calib['1'][3]
                pred_bbox = dict_out['pred_dicts'][0]['pred_boxes'].detach().cpu().numpy() # (n, 7)
                pred_scores = dict_out['pred_dicts'][0]['pred_scores'].detach().cpu().numpy() # (n,)
                # pred_labels = dict_out['pred_dicts'][0]['pred_labels'].detach().cpu().numpy() # (n,)
                result_ours = drawBBox(img_undistored, intrinsics, T_ldr2cam, gt_bbox, pred_bbox, pred_scores, calib, threshold) # (720, 1280, 3)
                result_ours = result_ours[80:, :, :] # (640, 1280, 3)
                
                
                # 4. 이미지 저장
                side_by_side = np.concatenate([proj_img, result_l4dr_ori, result_l4dr, result_ours], axis=1)
                save_path = f"/home/user/heejun/L4DR/viz_for_professor/{weather_cond_tag}_{idx_datum}.png"
                cv2.imwrite(save_path, side_by_side)
                print(save_path)


    def visualizeQual2(self, epoch=None, list_conf_thr=None, is_subset=False, is_print_memory=False, savevis = False):
        import torchvision.transforms.functional as TF
        import torch.nn.functional as F
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        from PIL import Image
        import os.path as osp
        import yaml


        self.network.eval()
        self.network.training=False
        
        # 1. l4dr 모델 부르기
        l4dr_config = './configs/L4DR.yml'
        model_path = './logs/L4DR/l4dr_3rd_reproduction/models/model_10.pt'
        if not hasattr(self, '_l4dr_model'):
            cfg_l4dr = EasyDict()
            cfg_l4dr.ROOT_DIR = getattr(self.cfg, 'ROOT_DIR', None)
            cfg_l4dr.LOCAL_RANK = getattr(self.cfg, 'LOCAL_RANK', 0)
            cfg_l4dr = cfg_from_yaml_file(l4dr_config, cfg_l4dr)

            class _TmpPline:
                pass
            tmp_pline = _TmpPline()
            tmp_pline.cfg = cfg_l4dr

            self._l4dr_model = build_network(tmp_pline).cuda()
            state_dict = torch.load(model_path, map_location='cpu')
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            self._l4dr_model.load_state_dict(state_dict, strict=False)
            self._l4dr_model.eval()
            self._l4dr_model.training = False
        
        
        save_file_folder = os.path.join(self.save_confmap_path, str(epoch))
        os.makedirs(save_file_folder)
        self.cam_calib_sequence = "./resources/cam_calib/calib_seq_v2"
        with torch.no_grad():
            # for idx_datum, dict_datum in enumerate(data_loader):
            # for idx_datum in range(0, len(self.dataset_test), 100):
            for idx_datum in range(4543, 11000, 10):
                # if idx_datum % 500 != 0:
                #     tqdm_bar.update(1)
                #     continue
                sample = self.dataset_test[idx_datum]
                dict_datum = self.dataset_test.collate_fn([sample])
                dict_out = self.network(dict_datum) # NOTE 일단 주석처리해서 빨리 image + lidar projected뽑기

                
                road_cond_tag = sample['meta']['desc']['road_type']
                weather_cond_tag =  sample['meta']['desc']['climate']
                
                
                # 1. 이미지에 lidar projection
                # 🛠️ camera parameters
                seq = sample['meta']['seq']
                dir_cam_calib = self.cam_calib_sequence + '/seq_'+str(int(seq)).zfill(2)
                dict_cam_calib = dict()
                list_yml = ['cam_1.yml']
                for yml_file_name in list_yml:
                    key_name = yml_file_name.split('.')[0].split('_')[1] # '1'
                    with open(osp.join(dir_cam_calib, yml_file_name), 'r') as yml_file:
                        dict_temp = yaml.safe_load(yml_file)
                    dict_cam_calib[key_name] = get_matrices_from_dict_calib(dict_temp) # img_size, intrinsics, distortion, T_ldr2cam

                # 📷 image
                image_path = sample['meta']['path']['front']
                img = cv2.imread(image_path)  # BGR 형식의 numpy array로 로드됨
                H_img, W_img, C = img.shape # (720, 2560, 3)
                img = img[:, :W_img//2, :]  # (720, 1280, 3)
                
                # 🔫 lidar
                ldr64 = sample['ldr64']
                calib = sample['meta']['calib'] # [-2.54, 0.3, 0.7]
                ldr64[:, :3] -= np.array(calib)

                # projection 수행
                proj_img, img_undistored = show_projected_point_cloud_yml_Qual1(img, ldr64, dict_cam_calib['1'], undistort=True) # (720, 1280, 3)
                proj_img = proj_img[80:, :, :] # (640, 1280, 3)
                # cv2.imwrite(f"./proj_img_test_{idx_datum}.png", proj_img)
                # breakpoint()
                
                
                # 2. l4dr 결과랑 gt를 이미지에 그리기
                intrinsics = dict_cam_calib['1'][1]
                T_ldr2cam = dict_cam_calib['1'][3]
                gt_bbox = sample['meta']['label']
                threshold = 0.3
                dict_datum_l4dr = self.dataset_test.collate_fn([sample])
                dict_out_l4dr = self._l4dr_model(dict_datum_l4dr)
                pred_bbox_l4dr = dict_out_l4dr['pred_dicts'][0]['pred_boxes'].detach().cpu().numpy()
                pred_scores_l4dr = dict_out_l4dr['pred_dicts'][0]['pred_scores'].detach().cpu().numpy()
                result_l4dr = drawBBox(
                    img_undistored,
                    intrinsics,
                    T_ldr2cam,
                    gt_bbox,
                    pred_bbox_l4dr,
                    pred_scores_l4dr,
                    calib,
                    threshold
                )
                result_l4dr = result_l4dr[80:, :, :]
                
                
                # 3. ours 결과랑 gt를 이미지에 그리기
                pred_bbox = dict_out['pred_dicts'][0]['pred_boxes'].detach().cpu().numpy() # (n, 7)
                pred_scores = dict_out['pred_dicts'][0]['pred_scores'].detach().cpu().numpy() # (n,)
                # pred_labels = dict_out['pred_dicts'][0]['pred_labels'].detach().cpu().numpy() # (n,)
                result_ours = drawBBox(img_undistored, intrinsics, T_ldr2cam, gt_bbox, pred_bbox, pred_scores, calib, threshold) # (720, 1280, 3)
                result_ours = result_ours[80:, :, :] # (640, 1280, 3)
                # cv2.imwrite(f"check_{idx_datum}.png", result_ours)
                
                
                # 4. 이미지 저장
                side_by_side = np.concatenate([proj_img, result_l4dr, result_ours], axis=1)
                save_path = f"./viz_qual2/temp/{weather_cond_tag}_{idx_datum}.png"
                cv2.imwrite(save_path, side_by_side)
                print(save_path)


    def visualizeQual1(self, epoch=None, list_conf_thr=None, is_subset=False, is_print_memory=False, savevis = False):
        import torchvision.transforms.functional as TF
        import torch.nn.functional as F
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        from PIL import Image
        import os.path as osp
        import yaml
        # 0=red, 1=white
        red_white_cmap = LinearSegmentedColormap.from_list(
            "RedWhite", ["red", "white"]
        )
        # red_half_white_cmap = LinearSegmentedColormap.from_list(
        #     "RedHalfWhite",
        #     [(0.0, "red"),
        #     (0.5, "white"),
        #     (1.0, "white")]
        # )

        self.network.eval()
        self.network.training=False
        
        
        save_file_folder = os.path.join(self.save_confmap_path, str(epoch))
        os.makedirs(save_file_folder)
        # tqdm_bar = tqdm(total=len(self.dataset_test), desc='Test (Total): ')
        # data_loader = torch.utils.data.DataLoader(self.dataset_test, batch_size = 1, shuffle = False, collate_fn = self.dataset_test.collate_fn, num_workers = self.cfg.OPTIMIZER.NUM_WORKERS)
        self.cam_calib_sequence = "./resources/cam_calib/calib_seq_v2"
        with torch.no_grad():
            # for idx_datum, dict_datum in enumerate(data_loader):
            # for idx_datum in range(0, len(self.dataset_test), 100):
            for idx_datum in range(4500, 9800, 100):
                # if idx_datum % 500 != 0:
                #     tqdm_bar.update(1)
                #     continue
                sample = self.dataset_test[idx_datum]
                dict_datum = self.dataset_test.collate_fn([sample])
                dict_out = self.network(dict_datum) # NOTE 일단 주석처리해서 빨리 image + lidar projected뽑기

                
                # road_cond_tag, time_cond_tag, weather_cond_tag = dict_out['meta'][0]['desc']['road_type'], dict_out['meta'][0]['desc']['capture_time'], dict_out['meta'][0]['desc']['climate']
                road_cond_tag = sample['meta']['desc']['road_type']
                weather_cond_tag =  sample['meta']['desc']['climate']
                
                
                # 1. 이미지에 lidar projection
                # 🛠️ camera parameters
                seq = sample['meta']['seq']
                dir_cam_calib = self.cam_calib_sequence + '/seq_'+str(int(seq)).zfill(2)
                dict_cam_calib = dict()
                list_yml = ['cam_1.yml']
                for yml_file_name in list_yml:
                    key_name = yml_file_name.split('.')[0].split('_')[1] # '1'
                    with open(osp.join(dir_cam_calib, yml_file_name), 'r') as yml_file:
                        dict_temp = yaml.safe_load(yml_file)
                    dict_cam_calib[key_name] = get_matrices_from_dict_calib(dict_temp) # img_size, intrinsics, distortion, T_ldr2cam

                # 📷 image
                image_path = sample['meta']['path']['front']
                img = cv2.imread(image_path)  # BGR 형식의 numpy array로 로드됨
                H_img, W_img, C = img.shape # (720, 2560, 3)
                img = img[:, :W_img//2, :]  # (720, 1280, 3)
                
                # 🔫 lidar
                ldr64 = sample['ldr64']
                calib = sample['meta']['calib'] # [-2.54, 0.3, 0.7]
                ldr64[:, :3] -= np.array(calib)

                # projection 수행
                proj_img, img_undistored = show_projected_point_cloud_yml_Qual1(img, ldr64, dict_cam_calib['1'], undistort=True) # (720, 1280, 3)
                proj_img = proj_img[80:, :, :] # (640, 1280, 3)
                # cv2.imwrite(f"./proj_img_test_{idx_datum}.png", proj_img)
                # breakpoint()
                
                # 2. conf map 뽑기
                conf_map = dict_datum['conf_map']   # [1, 1, 40, 80]
                conf_map_resized = F.interpolate(conf_map, size=(640, 1280), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)  # → [640, 1280]
                conf_np = conf_map_resized.detach().cpu().numpy() # (640, 1280)
                B = (conf_np * 255).astype(np.uint8)
                G = (conf_np * 255).astype(np.uint8)
                R = np.full_like(conf_np, 255, dtype=np.uint8)
                conf_color = np.stack([B, G, R], axis=-1) # (640, 1280, 3)
                
                # 3. bbox 그리기
                intrinsics = dict_cam_calib['1'][1]
                T_ldr2cam = dict_cam_calib['1'][3]
                gt_bbox = sample['meta']['label']
                pred_bbox = dict_out['pred_dicts'][0]['pred_boxes'].detach().cpu().numpy() # (n, 7)
                pred_scores = dict_out['pred_dicts'][0]['pred_scores'].detach().cpu().numpy() # (n,)
                # pred_labels = dict_out['pred_dicts'][0]['pred_labels'].detach().cpu().numpy() # (n,)
                threshold = 0.3
                result_img = drawBBox(img_undistored, intrinsics, T_ldr2cam, gt_bbox, pred_bbox, pred_scores, calib, threshold) # (720, 1280, 3)
                result_img = result_img[80:, :, :] # (640, 1280, 3)
                # cv2.imwrite(f"check_{idx_datum}.png", result_img)
                # 4. 이미지 저장
                side_by_side = np.concatenate([proj_img, conf_color, result_img], axis=1)
                save_path = f"./viz_qual1/temp/{weather_cond_tag}_{idx_datum}.png"
                cv2.imwrite(save_path, side_by_side)
                print(save_path)
                # continue
            
                # # image_path = dict_datum['meta'][0]['path']['front']
                # image_path = sample['meta']['path']['front']
                # # 1) 원본 이미지 로드 (PIL -> tensor)
                # image = Image.open(image_path).convert("RGB")
                # image = TF.to_tensor(image)  # [C, 720, 2560]
                # C, H_img, W_img = image.shape # [3, 720, 2560]
                # image = image[:, 80:, :W_img//2]  # [3, 640, 1280]

                # # 2) 이미지 downscale → (320, 640)
                # image = TF.resize(image, [320, 640])  # [C, 320, 640]

                # # 3) conf_map도 (320, 640)으로 upsample
                # conf_map = conf_map.unsqueeze(0)  # [1, 40, 80]
                # conf_map_up = TF.resize(conf_map, [320, 640])  # [1, 320, 640]
                # conf_map_up = conf_map_up[0]  # [320, 640]

                # # 4) numpy 변환
                # resized_img = image.permute(1, 2, 0).cpu().numpy()  # [320, 640, 3]
                # conf_map_np = conf_map_up.detach().cpu().numpy()

                # # 5) 시각화
                # fig, axs = plt.subplots(1, 2, figsize=(10, 5))

                # # input image
                # axs[0].imshow(resized_img)
                # axs[0].set_title("Input Image")
                # # axs[0].axis("off")
                # axs[0].set_xticks([])  # 눈금 제거
                # axs[0].set_yticks([])  # 눈금 제거
                # for spine in axs[0].spines.values():
                #     spine.set_edgecolor("black")
                #     spine.set_linewidth(0.8)
                    
                    
                # # confidence map (0=red, 0.5=white, 1=blue)
                # # axs[1].imshow(conf_map_np, cmap=red_white_blue_cmap, vmin=0, vmax=1)
                # axs[1].imshow(conf_map_np, cmap=red_white_cmap, vmin=0, vmax=1)
                # axs[1].set_title("Confidence Map")
                # # axs[1].axis("off")
                # axs[1].set_xticks([])
                # axs[1].set_yticks([])
                # for spine in axs[1].spines.values():
                #     spine.set_edgecolor("black")
                #     spine.set_linewidth(0.8)
                
                # plt.tight_layout()

                # # 6) 저장
                # # filename = f"{idx_datum:05d}_{road_cond_tag}_{time_cond_tag}_{weather_cond_tag}.jpg"
                # filename = f"{idx_datum:05d}_{road_cond_tag}_{weather_cond_tag}.jpg"
                # save_file = os.path.join(save_file_folder, filename)
                # plt.savefig(save_file, dpi=150, bbox_inches="tight", pad_inches=0.05)
                # plt.close()
                # print(f'saved at: {save_file}')
                     

    def visualizeConfMap_v5(self, epoch=None, list_conf_thr=None, is_subset=False, is_print_memory=False, savevis = False):
        import torchvision.transforms.functional as TF
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        from PIL import Image
        # 0=red, 1=white
        red_white_cmap = LinearSegmentedColormap.from_list(
            "RedWhite", ["red", "white"]
        )
        # red_half_white_cmap = LinearSegmentedColormap.from_list(
        #     "RedHalfWhite",
        #     [(0.0, "red"),
        #     (0.5, "white"),
        #     (1.0, "white")]
        # )

        self.network.eval()
        self.network.training=False
        
        
        save_file_folder = os.path.join(self.save_confmap_path, str(epoch))
        os.makedirs(save_file_folder)
        # tqdm_bar = tqdm(total=len(self.dataset_test), desc='Test (Total): ')
        # data_loader = torch.utils.data.DataLoader(self.dataset_test, batch_size = 1, shuffle = False, collate_fn = self.dataset_test.collate_fn, num_workers = self.cfg.OPTIMIZER.NUM_WORKERS)

        with torch.no_grad():
            # for idx_datum, dict_datum in enumerate(data_loader):
            # for idx_datum in range(0, len(self.dataset_test), 100):
            for idx_datum in range(4400, 8000, 33):
                # if idx_datum % 500 != 0:
                #     tqdm_bar.update(1)
                #     continue
                sample = self.dataset_test[idx_datum]
                dict_datum = self.dataset_test.collate_fn([sample])
                dict_out = self.network(dict_datum) # inference

                road_cond_tag, time_cond_tag, weather_cond_tag = dict_out['meta'][0]['desc']['road_type'], dict_out['meta'][0]['desc']['capture_time'], dict_out['meta'][0]['desc']['climate']
                conf_map = dict_datum['conf_map'][0, 0]   # [H, W]
                
                image_path = dict_datum['meta'][0]['path']['front']

                # 1) 원본 이미지 로드 (PIL -> tensor)
                image = Image.open(image_path).convert("RGB")
                image = TF.to_tensor(image)  # [C, 720, 2560]
                C, H_img, W_img = image.shape # [3, 720, 2560]
                image = image[:, 80:, :W_img//2]  # [3, 640, 1280]

                # 2) 이미지 downscale → (320, 640)
                image = TF.resize(image, [320, 640])  # [C, 320, 640]

                # 3) conf_map도 (320, 640)으로 upsample
                conf_map = conf_map.unsqueeze(0)  # [1, 40, 80]
                conf_map_up = TF.resize(conf_map, [320, 640])  # [1, 320, 640]
                conf_map_up = conf_map_up[0]  # [320, 640]

                # 4) numpy 변환
                resized_img = image.permute(1, 2, 0).cpu().numpy()  # [320, 640, 3]
                conf_map_np = conf_map_up.detach().cpu().numpy()

                # 5) 시각화
                fig, axs = plt.subplots(1, 2, figsize=(10, 5))

                # input image
                axs[0].imshow(resized_img)
                axs[0].set_title("Input Image")
                # axs[0].axis("off")
                axs[0].set_xticks([])  # 눈금 제거
                axs[0].set_yticks([])  # 눈금 제거
                for spine in axs[0].spines.values():
                    spine.set_edgecolor("black")
                    spine.set_linewidth(0.8)
                    
                    
                # confidence map (0=red, 0.5=white, 1=blue)
                # axs[1].imshow(conf_map_np, cmap=red_white_blue_cmap, vmin=0, vmax=1)
                axs[1].imshow(conf_map_np, cmap=red_white_cmap, vmin=0, vmax=1)
                axs[1].set_title("Confidence Map")
                # axs[1].axis("off")
                axs[1].set_xticks([])
                axs[1].set_yticks([])
                for spine in axs[1].spines.values():
                    spine.set_edgecolor("black")
                    spine.set_linewidth(0.8)
                
                plt.tight_layout()

                # 6) 저장
                filename = f"{idx_datum:05d}_{road_cond_tag}_{time_cond_tag}_{weather_cond_tag}.jpg"
                save_file = os.path.join(save_file_folder, filename)
                plt.savefig(save_file, dpi=150, bbox_inches="tight", pad_inches=0.05)
                plt.close()
                print(f'saved at: {save_file}')
                
                
    def visualizeConfMap_V3(self, epoch=None, list_conf_thr=None, is_subset=False, is_print_memory=False, savevis = False):
        import torchvision.transforms.functional as TF
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        from PIL import Image
        # 0=red, 1=white
        # red_white_cmap = LinearSegmentedColormap.from_list(
        #     "RedWhite", ["red", "white"]
        # )
        red_half_white_cmap = LinearSegmentedColormap.from_list(
            "RedHalfWhite",
            [(0.0, "red"),
            (0.5, "white"),
            (1.0, "white")]
        )

        self.network.eval()
        self.network.training=False
        
        
        save_file_folder = os.path.join(self.save_confmap_path, str(epoch))
        os.makedirs(save_file_folder)
        # tqdm_bar = tqdm(total=len(self.dataset_test), desc='Test (Total): ')
        # data_loader = torch.utils.data.DataLoader(self.dataset_test, batch_size = 1, shuffle = False, collate_fn = self.dataset_test.collate_fn, num_workers = self.cfg.OPTIMIZER.NUM_WORKERS)

        with torch.no_grad():
            # for idx_datum, dict_datum in enumerate(data_loader):
            for idx_datum in range(0, len(self.dataset_test), 500):
                # if idx_datum % 500 != 0:
                #     tqdm_bar.update(1)
                #     continue
                sample = self.dataset_test[idx_datum]
                dict_datum = self.dataset_test.collate_fn([sample])
                dict_out = self.network(dict_datum) # inference
                
                road_cond_tag, time_cond_tag, weather_cond_tag = dict_out['meta'][0]['desc']['road_type'], dict_out['meta'][0]['desc']['capture_time'], dict_out['meta'][0]['desc']['climate']
                conf_map = dict_datum['conf_map'][0, 0]   # [H, W]
                image_path = dict_datum['meta'][0]['path']['front']

                # 1) 원본 이미지 로드 (PIL -> tensor)
                image = Image.open(image_path).convert("RGB")
                image = np.array(image)
                # image = image.astype(np.float32) / 255.0
                H_img, W_img, C = image.shape
                image = image[:, :W_img // 2, :] # (720, 1280, 3)
                
                
                # image = TF.to_tensor(image)  # [C, 720, 2560]
                # C, H_img, W_img = image.shape
                # image = image[:, :, :W_img//2]  # [C, 720, 1280]

                # 2) lidar point cloud 갖고오기 (uncalibrated된 거 갖고 와야 함)
                ldr64 = self.dataset_test.get_ldr64_from_path(sample, is_calib=False) # calibration X, [N, 9]
                self.show_projected_point_cloud_yml(image, ldr64, undistort=True)
                
                
                
            
                # 2) 이미지 downscale → (360, 640)
                image = TF.resize(image, [360, 640])  # [C, 360, 640]

                # 3) conf_map도 (360, 640)으로 upsample
                conf_map = conf_map.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
                conf_map_up = TF.resize(conf_map, [360, 640])  # [1, 360, 640]
                conf_map_up = conf_map_up[0, 0]  # [360, 640]

                # 4) numpy 변환
                resized_img = image.permute(1, 2, 0).cpu().numpy()  # [360, 640, 3]
                conf_map_np = conf_map_up.detach().cpu().numpy()

                # 5) 시각화
                fig, axs = plt.subplots(1, 2, figsize=(10, 5))

                # input image
                axs[0].imshow(resized_img)
                axs[0].set_title("Input Image")
                axs[0].axis("off")
                
                # confidence map (0=red, 0.5=white, 1=blue)
                # axs[1].imshow(conf_map_np, cmap=red_white_blue_cmap, vmin=0, vmax=1)
                axs[1].imshow(conf_map_np, cmap=red_half_white_cmap, vmin=0, vmax=1)
                axs[1].set_title("Confidence Map")
                axs[1].axis("off")
                
                
                plt.tight_layout()

                # 6) 저장
                filename = f"{idx_datum:05d}_{road_cond_tag}_{time_cond_tag}_{weather_cond_tag}.jpg"
                save_file = os.path.join(save_file_folder, filename)
                plt.savefig(save_file, dpi=150, bbox_inches="tight")
                plt.close()
                print(f'saved at: {save_file}')
                


    def show_projected_point_cloud_yml(self, img, pcd, undistort=True):
        """
        rl3dod 시각화 검증용
        """
        
        img_size = (1280, 720)
        intrinsics= np.load('/home/user/heejun/L4DR/resources/params_for_viz/intrinsics.npy')
        distortion= np.load('/home/user/heejun/L4DR/resources/params_for_viz/distortion.npy')
        T_ldr2cam = np.load('/home/user/heejun/L4DR/resources/params_for_viz/T_ldr2cam.npy')
        
        img_process = img # (720, 1280, 3)

        if undistort:
            ncm, _ = cv2.getOptimalNewCameraMatrix(intrinsics, distortion, img_size, alpha=0.0)
            
            for j in range(3):
                for i in range(3):
                    intrinsics[j,i] = ncm[j, i]
            # print(intrinsics)
            # print(ncm)
            map_x, map_y = cv2.initUndistortRectifyMap(intrinsics, distortion, None, ncm, img_size, cv2.CV_32FC1) # shape은 바뀌지 않는다.
            
            img_process = cv2.remap(img_process, map_x, map_y, cv2.INTER_LINEAR)
        

        
        
        T_cam2pix = np.insert(np.insert(intrinsics, 3, [0,0,0], axis=1), 3, [0,0,0,1], axis=0)
        
        T_ldr2cam = np.insert(T_ldr2cam, 3, [0,0,0,1], axis=0)
        T_ldr2pix = T_cam2pix@T_ldr2cam

        

        
        pcd = pcd[np.where(pcd[:,0]>0)]
        # print(f"shape of pcd: {pcd.shape}")
        pc_ldr = (np.insert(pcd[:,:3], 3, [1], axis=1)).T
        pc_cam = T_ldr2pix@pc_ldr
        pc_cam[:2,:] /= pc_cam[2,:]
        
        img_process = np.flip(img_process, axis=2) # bgr to rgb
        plt.figure(figsize=(12,5),dpi=96,tight_layout=True)
        img_h,img_w,_ = img_process.shape
        plt.axis([0,img_w,img_h,0])
        plt.imshow(img_process)
        pc_cam = (pc_cam.T)[:,:3]
        pc_cam = pc_cam[np.where(
            (pc_cam[:,0]>=0) & (pc_cam[:,0]<img_w) &
            (pc_cam[:,1]>=0) & (pc_cam[:,1]<img_h) &
            (pc_cam[:,2]>3))]
        
        plt.scatter(pc_cam[:,0],pc_cam[:,1],c=1/pc_cam[:,2],cmap='rainbow_r',alpha=0.2,s=1.5)
        plt.xticks([])
        plt.yticks([])
        # save_path = f"{str(idx)}_yml_{undistort}.png"
        
        # print(T_ldr2pix)
        # plt.show()
        # plt.savefig(save_path)
        save_path = f"/home/user/heejun/L4DR/test.png"
        print(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close()
        
    def visualizeConfMap_v2(self, epoch=None, list_conf_thr=None, is_subset=False, is_print_memory=False, savevis=False):
        import torchvision.transforms.functional as TF
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        from PIL import Image
        from io import BytesIO
        import numpy as np
        import cv2
        import os

        red_half_white_cmap = LinearSegmentedColormap.from_list(
            "RedHalfWhite",
            [(0.0, "red"), (0.5, "white"), (1.0, "white")]
        )

        self.network.eval()
        self.network.training = False

        save_file_folder = os.path.join(self.save_confmap_path, str(epoch))
        os.makedirs(save_file_folder, exist_ok=True)
        
        
        img_size = (1280, 720)
        intrinsics = np.load('/home/user/heejun/L4DR/resources/params_for_viz/intrinsics.npy')
        distortion = np.load('/home/user/heejun/L4DR/resources/params_for_viz/distortion.npy')
        T_ldr2cam = np.load('/home/user/heejun/L4DR/resources/params_for_viz/T_ldr2cam.npy')
        T_ldr2cam = np.insert(T_ldr2cam, 3, [0,0,0,1], axis=0)
        # undistort
        ncm, _ = cv2.getOptimalNewCameraMatrix(intrinsics, distortion, img_size, alpha=0.0)
        intrinsics[:3, :3] = ncm[:3, :3]
        map_x, map_y = cv2.initUndistortRectifyMap(intrinsics, distortion, None, ncm, img_size, cv2.CV_32FC1)
        
        
        with torch.no_grad():
            for idx_datum in range(0, len(self.dataset_test), 500):
                sample = self.dataset_test[idx_datum]
                dict_datum = self.dataset_test.collate_fn([sample])
                dict_out = self.network(dict_datum)

                road_cond_tag = dict_out['meta'][0]['desc']['road_type']
                time_cond_tag = dict_out['meta'][0]['desc']['capture_time']
                weather_cond_tag = dict_out['meta'][0]['desc']['climate']
                conf_map = dict_datum['conf_map'][0, 0]
                image_path = dict_datum['meta'][0]['path']['front']

                # 1️⃣ 원본 이미지 로드
                image = Image.open(image_path).convert("RGB")
                image = np.array(image)
                H_img, W_img, _ = image.shape
                image = image[:, :W_img // 2, :]  # (720, 1280, 3)

                # 2️⃣ Lidar Point Cloud 가져오기
                ldr64 = self.dataset_test.get_ldr64_from_path(sample, is_calib=False)

                # 3️⃣ Point cloud projected image 생성 (plt로 그리고 numpy로 변환)

                img_process = image.copy()


                img_process = cv2.remap(img_process, map_x, map_y, cv2.INTER_LINEAR)

                # project lidar to pixel
                T_cam2pix = np.insert(np.insert(intrinsics, 3, [0,0,0], axis=1), 3, [0,0,0,1], axis=0)
                
                T_ldr2pix = T_cam2pix @ T_ldr2cam

                pcd = ldr64[np.where(ldr64[:, 0] > 0)]
                pc_ldr = np.insert(pcd[:, :3], 3, [1], axis=1).T
                pc_cam = T_ldr2pix @ pc_ldr
                pc_cam[:2, :] /= pc_cam[2, :]

                img_process = np.flip(img_process, axis=2)  # BGR → RGB
                img_h, img_w, _ = img_process.shape # (720, 1280, 3)

                
                img_vis = draw_gt_bboxes_on_image(img_process, sample['meta']['label'], T_ldr2pix, save_path="/home/user/heejun/L4DR/gt_vis.png" )
                fig, ax = plt.subplots(figsize=(12,5), dpi=96, tight_layout=True)
                ax.axis([0, img_w, img_h, 0])
                ax.imshow(img_process)

                pc_cam = (pc_cam.T)[:, :3]
                pc_cam = pc_cam[np.where(
                    (pc_cam[:,0]>=0) & (pc_cam[:,0]<img_w) &
                    (pc_cam[:,1]>=0) & (pc_cam[:,1]<img_h) &
                    (pc_cam[:,2]>3))]
                
                ax.scatter(pc_cam[:,0], pc_cam[:,1], c=1/pc_cam[:,2], cmap='rainbow_r', alpha=0.2, s=1.5)
                ax.set_xticks([]); ax.set_yticks([])

                # plt → numpy
                buf = BytesIO()
                fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', pad_inches=0)
                plt.close(fig)
                buf.seek(0)
                proj_img = np.array(Image.open(buf).convert("RGB")) # (705, 1253, 3)
                new_h, _, _ = proj_img.shape 
                crop_size = int(round((new_h / H_img) * 80)) # 78
                proj_img = proj_img[crop_size:, :, :] # (627, 1253, 3)

                # 4️⃣ Downscale projected image (대신 사용)
                proj_img = cv2.resize(proj_img, (640, 320)) # (320, 640, 3)

                # 5️⃣ conf_map resize
                conf_map = conf_map.unsqueeze(0).unsqueeze(0)
                conf_map_up = TF.resize(conf_map, [320, 640])[0, 0]
                conf_map_np = conf_map_up.detach().cpu().numpy() # (320, 640)
                
                
                
                
                
                
                # bbox 그린 이미지 추가
                
                
                
                
                
                
                

                # 6️⃣ 시각화
                fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                axs[0].imshow(proj_img)
                axs[0].set_title("Projected PointCloud")
                axs[0].axis("off")

                axs[1].imshow(conf_map_np, cmap=red_half_white_cmap, vmin=0, vmax=1)
                axs[1].set_title("Confidence Map")
                axs[1].axis("off")

                plt.tight_layout()

                filename = f"{idx_datum:05d}_{road_cond_tag}_{time_cond_tag}_{weather_cond_tag}.jpg"
                save_file = os.path.join(save_file_folder, filename)
                plt.savefig(save_file, dpi=150, bbox_inches="tight")
                plt.close()
                print(f"saved at: {save_file}")
                
                
def draw_gt_bboxes_on_image(img_process, gt_labels, T_ldr2pix, save_path="gt_vis.png"):
    img_vis = img_process.copy()

    for gt in gt_labels:
        cls_name, (x, y, z, th, l, w, h), trk, avail = gt
        x -= -2.54
        y -= 0.3
        z -= 0.7
        x -= -2.54
        y -= 0.3
        z -= 0.7
        

        # 1️⃣ local corners (8 points)
        x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
        y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
        z_corners = [0, 0, 0, 0, h, h, h, h]
        corners = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)

        # 2️⃣ rotation and translation
        R = np.array([
            [np.cos(th), -np.sin(th), 0],
            [np.sin(th),  np.cos(th), 0],
            [0, 0, 1]
        ])
        corners_3d = R @ corners + np.array([[x], [y], [z]])

        # 3️⃣ transform to image pixels
        corners_3d_hom = np.vstack([corners_3d, np.ones((1, 8))])
        pts_2d_hom = T_ldr2pix @ corners_3d_hom
        pts_2d = (pts_2d_hom[:2] / pts_2d_hom[2]).T  # (8, 2)

        # 4️⃣ draw edges
        edges = [
            (0,1), (1,2), (2,3), (3,0),
            (4,5), (5,6), (6,7), (7,4),
            (0,4), (1,5), (2,6), (3,7)
        ]

        color = (0, 255, 0) if cls_name == "Sedan" else (255, 0, 0)
        for i, j in edges:
            pt1 = tuple(pts_2d[i].astype(int))
            pt2 = tuple(pts_2d[j].astype(int))
            cv2.line(img_vis, pt1, pt2, color, 2)

    # 5️⃣ save result
    cv2.imwrite(save_path, cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR))
    print(f"[✓] Saved GT visualization → {save_path}")

    return img_vis
