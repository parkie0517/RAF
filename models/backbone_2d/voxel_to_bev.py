"""
* 3D Sparse Voxel Features -> 2D BEV Feature Map
* Created by HJ
* 나중에 필요없으면 삭제 ㄱㄱ
"""
import torch
import torch.nn as nn

import spconv.pytorch as spconv
import numpy as np


class VOXEL_TO_BEV(nn.Module):
    def __init__(self, 
                 model_cfg,
                 point_cloud_range, # [ 0. , -6.4, -2. , 72. ,  6.4,  6. ]
                 grid_size, # 0.4
                 encoding_stride, # [1, 2, 2]
                 ):
        super(VOXEL_TO_BEV, self).__init__()
        self.model_cfg = model_cfg

        x_min, x_max = point_cloud_range[0], point_cloud_range[3] # (0.0, 72.0)
        y_min, y_max = point_cloud_range[1], point_cloud_range[4] # (-6.4, 6.4)
        z_min, z_max = point_cloud_range[2], point_cloud_range[5] # (-2.0, 6.0)
        
        self.grid_size = grid_size
        for stride in encoding_stride:
            self.grid_size *= stride
                
        
        self.z_shape = int((z_max-z_min) / self.grid_size) # 5
        self.y_shape = int((y_max-y_min) / self.grid_size) # 8
        self.x_shape = int((x_max-x_min) / self.grid_size) # 45
        # self.spatial_shape = [self.z_shape, self.y_shape, self.x_shape]

        # ========================
        # BEV 변환 모듈 정의
        # ========================
        in_channels = model_cfg.get("IN_CHANNELS", 256)  # 입력 feature 채널 수
        out_channels = model_cfg.get("OUT_CHANNELS", 128) # 출력 BEV 채널 수
        kernel_size = model_cfg.get("KERNEL_SIZE", 3)
        stride = model_cfg.get("STRIDE", 1)
        padding = model_cfg.get("PADDING", 1)

        # z축 collapse conv
        self.toBEV = spconv.SparseConv3d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(self.z_shape, 1, 1)  # z축 collapse
        )
        self.bn_bev = nn.BatchNorm1d(in_channels)

        # 2D upsampling conv
        self.convtrans2d = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.bn2d = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()



    def forward(self, batch_dict):
        sp_tensor = batch_dict['uem_sp_tenosr']
        # L_features = batch_dict['uem_lidar_feature']# [N_new, 256]
        # L_indices = batch_dict['uem_lidar_indices'] # [N_new, 4] # [B, Z, Y, X]
        
        # # 화욜 출근해서 요기부터 코딩 ㄱㄱ
        # batch_size = int(L_indices[:, 0].max().item()) + 1

        # # SparseConvTensor로 변환
        # sp_tensor = spconv.SparseConvTensor(
        #     features=L_features,
        #     indices=L_indices.int(),
        #     spatial_shape=self.spatial_shape,
        #     batch_size=batch_size
        # )

        # Z축 collapse
        bev_sp = self.toBEV(sp_tensor) # z축 collapse [5, 8, 45] ----> [1, 8, 45]
        bev_sp = bev_sp.replace_feature(self.bn_bev(bev_sp.features))
        bev_sp = bev_sp.replace_feature(self.relu(bev_sp.features))

        # Dense 변환 -> 2D BEV
        bev_dense = bev_sp.dense().squeeze(2)  # dense한 일반 텐서로 변경 [B, 256, 8, 45]

        bev_dense = self.convtrans2d(bev_dense) # [B, 256, 8, 45] ----> [B, 128, 8, 45]
        bev_dense = self.bn2d(bev_dense)
        bev_dense = self.relu(bev_dense)

        # batch_dict에 저장
        batch_dict['uem_bev_feat'] = bev_dense
        
        
        return batch_dict