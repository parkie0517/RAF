import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class VoxelScatter(nn.Module):
    """
    pillar 형식의 데이터를 BEV(Bird's Eye View) 형식의 데이터로 변환하는 클래스
    """
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size # (480, 80, 50)
        self.nz = 1 # 1로 update

    def forward(self, batch_dict, **kwargs):
        lidar_pillar_features, lidar_coords = batch_dict['lidar_voxel_features'], batch_dict['lidar_voxel_coords'] # [4575, 64], [4575, 4]
        radar_pillar_features, radar_coords = batch_dict['radar_voxel_features'], batch_dict['radar_voxel_coords'] # [3688, 64], [3688, 4]
        lidar_batch_spatial_features = []
        radar_batch_spatial_features = []
        radar_pillar_features = radar_pillar_features.reshape(-1,radar_pillar_features.shape[-1]) # [3688, 64]
        radar_coords = radar_coords.reshape(-1,radar_coords.shape[-1]) # [3688, 4]
        batch_size = lidar_coords[:, 0].max().int().item() + 1
        
        for lidar_batch_idx in range(batch_size): # 2
            lidar_spatial_feature = torch.zeros( 
                self.num_bev_features[0], # 64
                self.nz * self.nx * self.ny, # self.nx= 480, self.ny=80
                dtype=lidar_pillar_features.dtype,
                device=lidar_pillar_features.device)

            lidar_batch_mask = lidar_coords[:, 0] == lidar_batch_idx
            lidar_this_coords = lidar_coords[lidar_batch_mask, :]
            lidar_indices = lidar_this_coords[:, 1] + lidar_this_coords[:, 2] * self.nx + lidar_this_coords[:, 3]
            lidar_indices = lidar_indices.type(torch.long)
            lidar_pillars = lidar_pillar_features[lidar_batch_mask, :]
            lidar_pillars = lidar_pillars.t()
            lidar_spatial_feature[:, lidar_indices] = lidar_pillars
            lidar_batch_spatial_features.append(lidar_spatial_feature)
        for radar_batch_idx in range(batch_size):
            radar_spatial_feature = torch.zeros(
                self.num_bev_features[1],  # 64
                self.nz * self.nx * self.ny,
                dtype=radar_pillar_features.dtype,
                device=radar_pillar_features.device)

            radar_batch_mask = radar_coords[:, 0] == radar_batch_idx
            radar_this_coords = radar_coords[radar_batch_mask, :]
            radar_indices = radar_this_coords[:, 1] + radar_this_coords[:, 2] * self.nx + radar_this_coords[:, 3]
            radar_indices = radar_indices.type(torch.long)
            radar_pillars = radar_pillar_features[radar_batch_mask, :]
            radar_pillars = radar_pillars.t()
            radar_spatial_feature[:, radar_indices] = radar_pillars
            radar_batch_spatial_features.append(radar_spatial_feature)

        lidar_batch_spatial_features = torch.stack(lidar_batch_spatial_features, 0) # [2, 64, 38400]
        radar_batch_spatial_features = torch.stack(radar_batch_spatial_features, 0)  # [2, 64, 38400]
        
        lidar_batch_spatial_features = lidar_batch_spatial_features.view(batch_size, self.num_bev_features[0] * self.nz, self.ny, self.nx) # [2, 64, 80, 480]
        radar_batch_spatial_features = radar_batch_spatial_features.view(batch_size, self.num_bev_features[1] * self.nz, self.ny, self.nx) # [2, 64, 80, 480]
        
        batch_dict['lidar_spatial_features'] = lidar_batch_spatial_features
        batch_dict['radar_spatial_features'] = radar_batch_spatial_features
        
        batch_dict['spatial_features'] = torch.cat((lidar_batch_spatial_features, radar_batch_spatial_features), 1) # [B, C, W_bev, H_bev] # [2, 128, 80, 480]
        return batch_dict