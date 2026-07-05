'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr

======== ------------------------------ ===========
* eddited: Heejun Park, VILab, KAIST
* e-mail: parkhee.ticket@kaist.ac.kr
* description: for camera + 4d radar + lidar fusion
======== ------------------------------ ===========
'''

import os
import os.path as osp
import torch
import numpy as np
import open3d as o3d

from tqdm import tqdm
import random
from torch.utils.data import Dataset

### ----- Camera Branch ----- ###
from utils.util_calib import *
import yaml
import cv2
### ----- Camera Branch ----- ###





class KRadarDetection_v2_2(Dataset):
    """
        Dataset for camera + 4d radar + lidar fusion
    """
    def __init__(self, cfg=None, split='all'):

        cfg_from_yaml = True
        self.cfg=cfg.DATASET

        self.label = self.cfg.label
        self.label_version = self.cfg.get('label_version', 'v2_0')
        self.load_label_in_advance = True if self.label.remove_0_obj else False
        
        self.item = self.cfg.item
        self.calib = self.cfg.calib
        self.ldr64 = self.cfg.ldr64
        self.rdr_sparse = self.cfg.rdr_sparse
        self.roi = self.cfg.roi
        ### --------------- Camera Branch --------------- ###
        self.cam = self.cfg.get('cam', None)
        self.cam_calib = self.cfg.get('cam_calib', None)
        self.cam_preprocess = self.cfg.get('cam_preprocess', None)
        self.crop_size = self.cam_preprocess.get('crop_size', None)
        self.ori_size = self.cam_preprocess.get('ori_size', None)
        self.scale_size = self.cam_preprocess.get('scale_size', None)
        self.mean = self.cam_preprocess.get('mean', None)
        self.std = self.cam_preprocess.get('std', None)
        self.dict_cam_calib = self.get_dict_cam_calib_from_yml() \
                            if self.cam_calib is not None else None
        ### --------------- Camera Branch --------------- ###

        self.list_dict_item = self.load_dict_item(self.cfg.path_data, split) #  11021
        if cfg_from_yaml:
            self.cfg.NUM = len(self)
        
        self.collate_ver = self.cfg.get('collate_fn', 'v1_0') # Post-processing
    
    def load_dict_item(self, path_data, split):
        def get_split(split_txt, list_dict_split, val):
            f = open(split_txt, 'r')
            lines = f.readlines()
            f.close()
            for line in lines:
                seq, label = line.split(',')
                list_dict_split[int(seq)][label.rstrip('\n')] = val
        list_dict_split = [dict() for _ in range(58+1)]
        get_split(path_data.split[0], list_dict_split, 'train')
        get_split(path_data.split[1], list_dict_split, 'test')
        
        list_seqs_w_header = []
        for path_header in path_data.list_dir_kradar:
            list_seqs = os.listdir(path_header)
            for seq in list_seqs:
                if seq.find("sparse")==-1:
                    list_seqs_w_header.extend([(seq, path_header)])
        list_seqs_w_header = sorted(list_seqs_w_header, key=lambda x: int(x[0]))

        list_dict_item = []
        for seq, path_header in list_seqs_w_header:
            list_labels = sorted(os.listdir(osp.join(path_header, seq, 'info_label')))
            for label in list_labels:
                path_label_v2_1 = osp.join(f'./tools/revise_label/kradar_revised_label_v2_1', 'KRadar_revised_visibility', seq, label)
                if label in list_dict_split[int(seq)].keys():
                    dict_item = dict(
                            meta = dict(
                            header = path_header, seq = seq,
                            label_v2_1 = path_label_v2_1,
                            split = list_dict_split[int(seq)][label]
                        ),
                    )
                    if self.load_label_in_advance:
                        dict_item = self.get_label(dict_item)
                    list_dict_item.append(dict_item)
        if split == 'all':
            pass
        else:
            list_dict_item = list(filter(lambda item: item['meta']['split']==split, list_dict_item))
        # Filter unavailable frames (frames wo objects) (only)
        if self.label.remove_0_obj: # 17458 -> 11021
            list_dict_item = list(filter(lambda item: item['meta']['num_obj']>0, list_dict_item))
        return list_dict_item
    
    def get_label(self, dict_item):
        meta = dict_item['meta']
        temp_key = 'label_' + self.label_version
        path_label = meta[temp_key]
        ver = self.label_version

        f = open(path_label)
        lines = f.readlines()
        f.close()
        list_tuple_objs = []
        deg2rad = np.pi/180.
        
        header = (lines[0]).rstrip('\n')
        try:
            temp_idx, tstamp = header.split(', ')
        except: # line breaking error for v2_0
            _, header_prime, line0 = header.split('*')
            header = '*' + header_prime
            temp_idx, tstamp = header.split(', ')
            # print('* b4: ', lines)
            lines.insert(1, '*'+line0)
            lines[0] = header
            # print('* after: ', lines)
        rdr, ldr64, camf, ldr128, camr = temp_idx.split('=')[1].split('_')
        tstamp = tstamp.split('=')[1]
        dict_idx = dict(rdr=rdr, ldr64=ldr64, camf=camf,\
                        ldr128=ldr128, camr=camr, tstamp=tstamp)
        
        if ver == 'v2_1':
            for line in lines[1:]:
                # print(line)
                list_vals = line.rstrip('\n').split(', ')
                avail = list_vals[1]
                idx_p = int(list_vals[2])
                cls_name = (list_vals[3])
                x = float(list_vals[4])
                y = float(list_vals[5])
                z = float(list_vals[6])
                th = float(list_vals[7])*deg2rad
                l = 2*float(list_vals[8])
                w = 2*float(list_vals[9])
                h = 2*float(list_vals[10])
                list_tuple_objs.append((cls_name, (x, y, z, th, l, w, h), (idx_p), avail))

        header = dict_item['meta']['header']
        seq = dict_item['meta']['seq']
        path_calib = osp.join(header, seq, 'info_calib', 'calib_radar_lidar.txt')
        dict_path = dict(
            calib = path_calib,
            front = osp.join(header, seq, 'cam-front', f'cam-front_{camf}.png'),
            ldr64 = osp.join(header, seq, 'os2-64', f'os2-64_{ldr64}.pcd'),
            desc = osp.join(header, seq, 'description.txt'),
        )

        onlyR = self.label.onlyR
        consider_cls = self.label.consider_cls
        if consider_cls | onlyR:
            list_temp = []
            for obj in list_tuple_objs:
                cls_name, _, _, avail = obj
                if consider_cls:
                    is_consider, _, _, _ = self.label[cls_name]
                    if not is_consider:
                        continue
                if onlyR:
                    if avail != 'R':
                        continue
                list_temp.append(obj)
            list_tuple_objs = list_temp

        dict_item['meta']['calib'] = self.get_calib_values(path_calib) if self.item.calib else None
        if self.label.calib:
            list_temp = []
            dx, dy, dz = dict_item['meta']['calib']
            for obj in list_tuple_objs:
                cls_name, (x, y, z, th, l, w, h), trk, avail = obj
                x = x + dx
                y = y + dy
                z = z + dz
                list_temp.append((cls_name, (x, y, z, th, l, w, h), trk, avail))
            list_tuple_objs = list_temp

        if self.label.consider_roi: # after calib
            x_min, y_min, z_min, x_max, y_max, z_max = self.roi.xyz
            check_azimuth_for_rdr = self.roi.check_azimuth_for_rdr
            azimuth_min, azimuth_max = self.roi.azimuth_deg
            rad2deg = 180./np.pi
            temp_list = []
            for obj in list_tuple_objs:
                cls_name, (x, y, z, th, l, w, h), trk, avail = obj
                azimuth = np.arctan2(y, x)*rad2deg
                if check_azimuth_for_rdr & ((azimuth < azimuth_min) | (azimuth > azimuth_max)):
                    continue
                if (x < x_min) | (x > x_max) | (y < y_min) | (y > y_max) | (z < z_min) | (z > z_max):
                    continue
                temp_list.append(obj)
            list_tuple_objs = temp_list

        num_obj = len(list_tuple_objs)

        dict_item['meta'].update(dict(
            path=dict_path, idx=dict_idx, label=list_tuple_objs, num_obj=num_obj))
        return dict_item
    
    def get_dict_cam_calib_from_yml(self):
        dict_cam_calib = dict()
        dir_cam_calib = self.cam_calib.dir
        
        key_name = 'front0'
        yml_file_name = 'cam_front0.yml'
        with open(osp.join(dir_cam_calib, yml_file_name), 'r') as yml_file:
            dict_temp = yaml.safe_load(yml_file)
        dict_cam_calib[key_name] = get_matrices_from_dict_calib(dict_temp) # img_size, intrinsics, distortion, T_ldr2cam
        
        if self.cam_calib is not None:
            img_size, intrinsics, distortion, T_ldr2cam = dict_cam_calib['front0']
            
            # intrinsic 보정
            ncm, _ = cv2.getOptimalNewCameraMatrix(intrinsics, distortion, img_size, alpha=0.0)
            intrinsics[:, :] = ncm
            self.map_x, self.map_y = cv2.initUndistortRectifyMap(intrinsics, distortion, None, ncm, img_size, cv2.CV_32FC1) # 나중에 get_camera에서 사용할 거 미리 생성하기
            
            # extrinsic 보정
            rotation = T_ldr2cam[:, :3]
            calib_vals = -1 * np.array([-2.54,0.3,0.7]).reshape(3,)
            T_ldr2cam[:, 3] = T_ldr2cam[:, 3] + rotation @ calib_vals
            
            # update
            dict_cam_calib['front0'] = (img_size, intrinsics, distortion, T_ldr2cam)
                
        if self.cam_preprocess is not None:
            # crop 길이 계산
            crop_h, crop_w = self.crop_size
            orig_h = self.ori_size[0]  # orig_h = 720
            self.y_crop = orig_h - crop_h  # 720 - 640 = 80

            # intrinsic 처리
            img_size, intrinsics, distortion, T_ldr2cam = dict_cam_calib['front0']
            intrinsics[1, 2] -= self.y_crop   # c_y 보정
            img_size = (crop_h, crop_w)  # height 갱신
            # dict_cam_calib['front0'] = (img_size, intrinsics, distortion, T_ldr2cam) # update
            
            # scale에 대한 intrinsic 처리
            scale_h, scale_w = self.scale_size
            img_size = (scale_h, scale_w)
            ratio_h = crop_h/scale_h
            ratio_w = crop_w/scale_w
            intrinsics[0, 0] /= ratio_w
            intrinsics[0, 2] /= ratio_w
            intrinsics[1, 1] /= ratio_h
            intrinsics[1, 2] /= ratio_h
            
            # update
            dict_cam_calib['front0'] = (img_size, intrinsics, distortion, T_ldr2cam) # update
            
        return dict_cam_calib
    
    def get_calib_values(self, path_calib):
        f = open(path_calib, 'r')
        lines = f.readlines()
        f.close()
        list_calib = list(map(lambda x: float(x), lines[1].split(',')))
        list_values = [list_calib[1], list_calib[2], self.calib['z_offset']] # X, Y, Z
        return list_values
    
    def sample_points(self, points):
        num_points = self.cfg.rdr_sparse['sample_points']
        if num_points == -1:
            return points
        if num_points < len(points):
            pts_depth = np.linalg.norm(points[:, 0:3], axis=1)
            pts_near_flag = pts_depth < 40.0
            far_idxs_choice = np.where(pts_near_flag == 0)[0]
            near_idxs = np.where(pts_near_flag == 1)[0]
            choice = []
            if num_points > len(far_idxs_choice):
                near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
                choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                    if len(far_idxs_choice) > 0 else near_idxs_choice
            else: 
                choice = np.arange(0, len(points), dtype=np.int32)
                choice = np.random.choice(choice, num_points, replace=False)
            np.random.shuffle(choice)
        else:
            choice = np.arange(0, len(points), dtype=np.int32)
            num_points -= len(points)
            while num_points > 0:
                extra_choice = np.random.choice(choice, min(num_points,len(points)), replace=False)
                num_points -= len(points)
                choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)
        return points[choice]
    
    def get_description(self, dict_item): # ./tools/tag_generator
        f = open(dict_item['meta']['path']['desc'])
        line = f.readline()
        road_type, capture_time, climate = line.split(',')
        dict_desc = {
            'capture_time': capture_time,
            'road_type': road_type,
            'climate': climate,
        }
        f.close()
        dict_item['meta']['desc'] = dict_desc
        
        return dict_item
    
    
    ### ----- Camera ----- ###
    def get_camera_img(self, dict_item):
        dict_path = dict_item['meta']['path']
        
        # 이미지 갖고오기
        img_front = cv2.imread(dict_path['front'])
        dict_item['front0'] = img_front[:,:1280,:]

        # 이미지 보정하기
        if self.cam_calib is not None:
            img_front = dict_item['front0']
            img_undistorted = cv2.remap(img_front, self.map_x, self.map_y, cv2.INTER_LINEAR)
            dict_item['front0'] = img_undistorted
        
        # 이미지 전처리
        if self.cam_preprocess is not None:
            # 이미지 Crop
            img = dict_item['front0']
            img_cropped = img[self.y_crop:, :, :]
            
            # 이미지 resize
            scale_h, scale_w = self.scale_size
            img_resized = cv2.resize(img_cropped, (scale_w, scale_h), interpolation=cv2.INTER_AREA)

            # NOTE 아래 보이는 코드는 디버그 용으로 사용 ㄱㄱ
            # if dict_path['front'].split('/')[3] == '10':
            #     cv2.imwrite('scaled_image.png', img_resized) # 이미지 저장
            #     np.save('scaled_intrinsic.npy', self.dict_cam_calib['front0'][1]) # intrinsic 저장
            #     np.save('calibrated_extrinsic.npy', self.dict_cam_calib['front0'][3]) # extrinsic 저장
            #     np.save('calibrated_pts.npy', dict_item['ldr64']) # pts 저장
            #     print('done')
            
            # 이미지 정규화
            img_resized = img_resized.astype(np.float32) # uint8 -> float32
            mean = np.array(self.mean, dtype=np.float32)
            std = np.array(self.std, dtype=np.float32)
            img_normalized = (img_resized - mean) / std
            dict_item['front0'] = img_normalized
        
        # 이미지 변환
        dict_item['front0'] = np.transpose(dict_item['front0'], (2, 0, 1)) # HWC -> CHW
        return dict_item
    
    def get_camera_param(self, dict_item):
        """
        LSS에서 사용되는 camera parameter를 가져오는 함수
        """
        def get_inverse(T_mat):
            # 1. Separate the 3x3 rotation matrix (R) and the 3x1 translation vector (t)
            R = T_mat[:3, :3]
            t = T_mat[:3, 3]
            
            # 2. Transpose the rotation matrix to get the inverse rotation
            R_T = R.T
            
            # 3. Calculate the new translation vector for the inverse transformation
            t_inv = -R_T @ t
            T_mat_inv = np.hstack((R_T, t_inv.reshape(-1, 1)))
            T_mat_inv = np.insert(T_mat_inv, 3, [0,0,0,1], axis=0) # [3, 4] -> [4, 4]
            
            return T_mat_inv
        
        intrinsics = self.dict_cam_calib['front0'][1] # [3, 3]
        intrinsics = np.insert(np.insert(intrinsics, 3, [0,0,0], axis=1), 3, [0,0,0,1], axis=0) # [3, 3] -> [4, 4]
        T_ldr2cam = self.dict_cam_calib['front0'][3] # [3, 4]
        T_ldr2cam = np.insert(T_ldr2cam, 3, [0,0,0,1], axis=0) # [3, 4] -> [4, 4]
        T_cam2ldr = get_inverse(T_ldr2cam) # [4, 4]
        T_ldr2img = intrinsics @ T_ldr2cam # [4, 4]
        T_ldr2ego = np.eye(4) # [4, 4] # NOTE kradar는 lidar랑 ego가 동일함
        
        ### ----- Augmentation 관련 행렬 ----- ###
        img_aug_mat = np.eye(4) # [4, 4]
        ldr_aug_mat = np.eye(4) # [4, 4]
        ### ----- Augmentation 관련 행렬 ----- ###
        
        dict_item['cam_param'] = {
            'camera2lidar' : np.expand_dims(T_cam2ldr, axis=0),
            'lidar2camera' : np.expand_dims(T_ldr2cam, axis=0),
            'lidar2image' : np.expand_dims(T_ldr2img, axis=0),
            'camera2image' : np.expand_dims(intrinsics, axis=0),
            'lidar2ego' : np.expand_dims(T_ldr2ego, axis=0),
            'image_aug_matrix' : np.expand_dims(img_aug_mat, axis=0),
            # 'lidar_aug_matrix' : np.expand_dims(ldr_aug_mat, axis=0),
            'lidar_aug_matrix' : ldr_aug_mat,
        }
        
        return dict_item
    ### ----- Camera ----- ###
    
    def get_ldr64(self, dict_item):
        if self.ldr64.processed: # with attr & calib & roi
            pass # TODO
        else:
            with open(dict_item['meta']['path']['ldr64'], 'r') as f:
                lines = [line.rstrip('\n') for line in f][self.ldr64.skip_line:]
                pc_lidar = [point.split() for point in lines]
                f.close()
            pc_lidar = np.array(pc_lidar, dtype = float).reshape(-1, self.ldr64.n_attr)

            if self.ldr64.inside_ldr64:
                pc_lidar = pc_lidar[np.where(
                    (pc_lidar[:, 0] > 0.01) | (pc_lidar[:, 0] < -0.01) |
                    (pc_lidar[:, 1] > 0.01) | (pc_lidar[:, 1] < -0.01))]
            
            if self.ldr64.calib:
                n_pts, _ = pc_lidar.shape
                calib_vals = np.array(dict_item['meta']['calib']).reshape(-1,3).repeat(n_pts, axis=0)
                pc_lidar[:,:3] = pc_lidar[:,:3] + calib_vals
            
            # if self.ldr64

        dict_item['ldr64'] = pc_lidar

        return dict_item
    
    def get_ldr64_from_path(self, dict_item, is_calib=True):

        with open(dict_item['meta']['path']['ldr64'], 'r') as f:
            lines = [line.rstrip('\n') for line in f][self.ldr64.skip_line:]
            pc_lidar = [point.split() for point in lines]
            f.close()
        pc_lidar = np.array(pc_lidar, dtype = float).reshape(-1, self.ldr64.n_attr)

        if self.ldr64.inside_ldr64:
            pc_lidar = pc_lidar[np.where(
                (pc_lidar[:, 0] > 0.01) | (pc_lidar[:, 0] < -0.01) |
                (pc_lidar[:, 1] > 0.01) | (pc_lidar[:, 1] < -0.01))]
        
        if is_calib:
            n_pts, _ = pc_lidar.shape
            calib_vals = np.array(dict_item['meta']['calib']).reshape(-1,3).repeat(n_pts, axis=0)
            pc_lidar[:,:3] = pc_lidar[:,:3] + calib_vals
        

        return pc_lidar

    
    def get_rdr_sparse(self, dict_item):
        if self.rdr_sparse.processed:
            dir_rdr_sparse = self.rdr_sparse.dir
            seq = dict_item['meta']['seq']
            rdr_idx = dict_item['meta']['idx']['rdr']
            path_rdr_sparse = osp.join(dir_rdr_sparse, seq, f'sprdr_{rdr_idx}.npy')
            rdr_sparse = np.load(path_rdr_sparse)
        else: # from cube or tesseract (TODO)
            pass
        dict_item['rdr_sparse'] = rdr_sparse
        
        return dict_item
    
    def filter_roi(self, dict_item):
        x_min, y_min, z_min, x_max, y_max, z_max = self.roi.xyz
        list_keys = self.roi.keys
        for temp_key in list_keys:
            if temp_key in ['rdr_sparse', 'ldr64']:
                temp_data = dict_item[temp_key]
                temp_data = temp_data[np.where(
                    (temp_data[:, 0] > x_min) & (temp_data[:, 0] < x_max) &
                    (temp_data[:, 1] > y_min) & (temp_data[:, 1] < y_max) &
                    (temp_data[:, 2] > z_min) & (temp_data[:, 2] < z_max))]
                dict_item[temp_key] = temp_data
            elif temp_key == 'ldr64_uem':
                temp_data = dict_item['ldr64'].copy()
                temp_data = temp_data[np.where(
                    (temp_data[:, 0] > x_min) & (temp_data[:, 0] < x_max) &
                    (temp_data[:, 1] > y_min) & (temp_data[:, 1] < y_max) &
                    (temp_data[:, 2] > z_min) & (temp_data[:, 2] < z_max))]
                dict_item[temp_key] = temp_data
        
        return dict_item

    def __len__(self):
        return len(self.list_dict_item)
    
    def __getitem__(self, idx):
        # FIX: Copy dict_item to prevent accumulating data in self.list_dict_item
        # Without this copy, images/point clouds/depth maps stored in dict_item
        # persist in memory because dict_item is a reference to the stored dict.
        dict_item = self.list_dict_item[idx].copy()  # Shallow copy top-level dict
        dict_item['meta'] = dict_item['meta'].copy()  # Copy meta dict since get_label modifies it
        dict_item = self.get_label(dict_item) if not self.load_label_in_advance else dict_item
        dict_item = self.get_ldr64(dict_item) if self.item['ldr64'] else dict_item
        dict_item = self.get_rdr_sparse(dict_item) if self.item['rdr_sparse'] else dict_item
        dict_item = self.filter_roi(dict_item) if self.roi.filter else dict_item
        if self.item['rdr_sparse']:
            dict_item['rdr_sparse'] = self.sample_points(dict_item['rdr_sparse']) if ('sample_points' in self.cfg.rdr_sparse.keys()) else dict_item['rdr_sparse']

        dict_item = self.get_description(dict_item)
        
        ### Camera ###
        dict_item = self.get_camera_img(dict_item) if self.item['cam'] else dict_item
        dict_item = self.get_camera_param(dict_item) if self.item['cam'] else dict_item
        ### Camera ###
        


        return dict_item
    
    ### Vis ###
    def create_cylinder_mesh(self, radius, p0, p1, color=[1, 0, 0]):
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=np.linalg.norm(np.array(p1)-np.array(p0)))
        cylinder.paint_uniform_color(color)
        frame = np.array(p1) - np.array(p0)
        frame /= np.linalg.norm(frame)
        R = o3d.geometry.get_rotation_matrix_from_xyz((np.arccos(frame[2]), np.arctan2(-frame[0], frame[1]), 0))
        cylinder.rotate(R, center=[0, 0, 0])
        cylinder.translate((np.array(p0) + np.array(p1)) / 2)
        return cylinder
    
    def draw_3d_box_in_cylinder(self, vis, center, theta, l, w, h, color=[1, 0, 0], radius=0.1, in_cylinder=True):
        R = np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta),  np.cos(theta), 0],
                    [0,              0,             1]])
        corners = np.array([[l/2, w/2, h/2], [l/2, w/2, -h/2], [l/2, -w/2, h/2], [l/2, -w/2, -h/2],
                            [-l/2, w/2, h/2], [-l/2, w/2, -h/2], [-l/2, -w/2, h/2], [-l/2, -w/2, -h/2]])
        corners_rotated = np.dot(corners, R.T) + center
        lines = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7],
                [0, 4], [1, 5], [2, 6], [3, 7]]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners_rotated)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([color for i in range(len(lines))])
        if in_cylinder:
            for line in lines:
                cylinder = self.create_cylinder_mesh(radius, corners_rotated[line[0]], corners_rotated[line[1]], color)
                vis.add_geometry(cylinder)
        else:
            vis.add_geometry(line_set)

    def create_sphere(self, radius=0.2, resolution=30, rgb=[0., 0., 0.], center=[0., 0., 0.]):
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius, resolution)
        color = np.array(rgb)
        mesh_sphere.vertex_colors = o3d.utility.Vector3dVector([color for _ in range(len(mesh_sphere.vertices))])
        x, y, z = center
        transform = np.identity(4)
        transform[0, 3] = x
        transform[1, 3] = y
        transform[2, 3] = z
        mesh_sphere.transform(transform)
        return mesh_sphere
    
    def vis_in_open3d(self, dict_item, vis_list=['rdr_sparse', 'ldr64', 'label']):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
        if 'ldr64' in vis_list:
            pc_lidar = dict_item['ldr64']
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc_lidar[:,:3])
            vis.add_geometry(pcd)

        if 'rdr_sparse' in vis_list:
            rdr_sparse = dict_item['rdr_sparse']
            pcd_rdr = o3d.geometry.PointCloud()
            pcd_rdr.points = o3d.utility.Vector3dVector(rdr_sparse[:,:3])
            pcd_rdr.paint_uniform_color([0.,0.,0.])
            vis.add_geometry(pcd_rdr)
        
        if 'label' in vis_list:
            label = dict_item['meta']['label']
            for obj in label:
                cls_name, (x, y, z, th, l, w, h), trk, avail = obj
                consider, logit_idx, rgb, bgr = self.label[cls_name]
                if consider:
                    self.draw_3d_box_in_cylinder(vis, (x, y, z), th, l, w, h, color=rgb, radius=0.05)
            vis.run()
            vis.destroy_window()
    ### Vis ###

    ### Distribution ###
    def get_distribution_of_label(self, consider_avail=True):
        dict_label = self.label.copy()
        dict_label.pop('calib')
        dict_label.pop('onlyR')
        dict_label.pop('Label')
        dict_label.pop('consider_cls')
        dict_label.pop('consider_roi')
        dict_label.pop('remove_0_obj')
        
        dict_for_dist = dict()
        dict_for_value = dict()
        for obj_name in dict_label.keys():
            dict_for_dist[obj_name] = 0
            dict_for_value[obj_name] = [0., 0., 0.]
        
        if consider_avail:
            dict_avail = dict()
            list_avails = ['R', 'L', 'L1']
            for avail in list_avails:
                dict_temp = dict()
                for obj_name in dict_label.keys():
                    dict_temp[obj_name] = 0
                dict_avail[avail] = dict_temp

        for dict_item in tqdm(self.list_dict_item):
            dict_item = self.get_label(dict_item)
            for obj in dict_item['meta']['label']:
                cls_name, (x, y, z, th, l, w, h), trk, avail = obj
                dict_for_dist[cls_name] += 1
                dict_for_value[cls_name][0] += l
                dict_for_value[cls_name][1] += w
                dict_for_value[cls_name][2] += h
                try:
                    if consider_avail:
                        dict_avail[avail][cls_name] += 1
                except:
                    print(dict_item['meta']['label_v2_1'])

        for obj_name in dict_for_dist.keys():
            n_obj = dict_for_dist[obj_name]
            l, w, h = dict_for_value[obj_name]
            print('* # of ', obj_name, ': ', n_obj)
            divider = np.maximum(n_obj, 1)
            print('* lwh of ', obj_name, ': ', l/divider, ', ', w/divider, ', ', h/divider)
        
        if consider_avail:
            for avail in list_avails:
                print('-'*30, avail, '-'*30)
                for obj_name in dict_avail[avail].keys():
                    print('* # of ', obj_name, ': ', dict_avail[avail][obj_name])
    ### Distribution ###

    def collate_fn(self, list_batch):
        if None in list_batch:
            print('* Exception error (Dataset): collate fn 0')
            return None
        
        # gt_boxes (B, M, 8)
        dict_batch = dict()
        
        list_keys = list_batch[0].keys()
        for k in list_keys:
            if k != 'cam_param':
                dict_batch[k] = []
        
        dict_batch['label'] = []
        dict_batch['num_objs'] = []
        dict_batch['gt_boxes'] = [] # for UEM
        dict_batch['climate_list'] = []
        
        ### ----- Camera Parameters ----- ###
        dict_batch['camera2lidar'] = []
        dict_batch['lidar2camera'] = []
        dict_batch['lidar2image'] = []
        dict_batch['camera2image'] = []
        dict_batch['lidar2ego'] = []
        dict_batch['image_aug_matrix'] = []
        dict_batch['lidar_aug_matrix'] = []
        ### ----- Camera Parameters ----- ###
        
        max_objs = 0 # for gt_boxes (M)
        for batch_idx, dict_item in enumerate(list_batch):
            for k, v in dict_item.items():
                if k == 'meta':
                    dict_batch['meta'].append(v)
                    list_objs = []
                    list_gt_boxes = []
                    for tuple_obj in dict_item['meta']['label']:
                        cls_name, vals, trk_id, _ = tuple_obj
                        _, logit_idx, _, _ = self.label[cls_name]
                        list_objs.append((cls_name, logit_idx, vals, trk_id))
                        x, y, z, th, l, w, h = vals
                        list_gt_boxes.append([x, y, z, l, w, h, th, logit_idx])
                    dict_batch['label'].append(list_objs)
                    dict_batch['num_objs'].append(dict_item['meta']['num_obj'])
                    dict_batch['gt_boxes'].append(list_gt_boxes)
                    dict_batch['climate_list'].append(dict_item['meta']['desc']['climate']) # for UEM
                    max_objs = max(max_objs, dict_item['meta']['num_obj'])
                elif k in ['rdr_sparse', 'ldr64', 'front0', 'ldr64_uem']:
                    dict_batch[k].append(torch.from_numpy(dict_item[k]).float())
                elif k == 'cam_param':
                    for matrix_name, matrix_value in v.items():
                        dict_batch[matrix_name].append(torch.from_numpy(matrix_value).float())

        dict_batch['batch_size'] = batch_idx+1
        
        ### ----- LSS ----- ###
        dict_batch['ldr_points_list'] = dict_batch['ldr64']
        ### ----- LSS ----- ###
        
        batch_size = dict_batch['batch_size']
        gt_boxes = np.zeros((batch_size, max_objs, 8))
        for batch_idx in range(batch_size):
            gt_box = np.array(dict_batch['gt_boxes'][batch_idx])
            gt_boxes[batch_idx,:dict_batch['num_objs'][batch_idx],:] = gt_box
        dict_batch['gt_boxes'] = torch.tensor(gt_boxes, dtype=torch.float32)

        for k in list_keys:
            if k in ['rdr_sparse', 'ldr64', 'ldr64_uem']:
                batch_indices = []
                for batch_idx, pc in enumerate(dict_batch[k]):
                    batch_indices.append(torch.full((len(pc),), batch_idx))
                
                dict_batch[k] = torch.cat(dict_batch[k], dim=0)
                dict_batch['batch_indices_'+k] = torch.cat(batch_indices)
            elif k == 'front0':
                dict_batch[k] = torch.stack(dict_batch[k], dim=0)

        ### ----- Camera Parameters ----- ###
        if self.cfg.item['cam']:
            dict_batch['camera2lidar'] = torch.stack(dict_batch['camera2lidar'], dim=0)
            dict_batch['lidar2camera'] = torch.stack(dict_batch['lidar2camera'], dim=0)
            dict_batch['lidar2image'] = torch.stack(dict_batch['lidar2image'], dim=0)
            dict_batch['camera2image'] = torch.stack(dict_batch['camera2image'], dim=0)
            dict_batch['lidar2ego'] = torch.stack(dict_batch['lidar2ego'], dim=0)
            dict_batch['image_aug_matrix'] = torch.stack(dict_batch['image_aug_matrix'], dim=0)
            dict_batch['lidar_aug_matrix'] = torch.stack(dict_batch['lidar_aug_matrix'], dim=0)
        ### ----- Camera Parameters ----- ###
        
        
        
        
        return dict_batch

if __name__ == '__main__':
    kradar_detection = KRadarDetection_v2_2(split='all')
    print(len(kradar_detection)) # 34994 for all

    # dict_item = kradar_detection[5000]
    # kradar_detection.vis_in_open3d(dict_item, ['ldr64', 'label', 'rdr_sparse'])
    kradar_detection.get_distribution_of_label()
