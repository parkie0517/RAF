import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .vfe_template import VFETemplate
# import matplotlib.pyplot as plt
# import math


class BiDF_VFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.use_norm = self.model_cfg.USE_NORM # True
        self.with_distance = self.model_cfg.WITH_DISTANCE # False
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ # True
        self.use_preground_score = self.model_cfg.USE_RadarSCORE # True
        # num_point_features[0] for lidar, num_point_features[1] for radar
        num_point_features_l = num_point_features[0]
        num_point_features_r = num_point_features[1]
        
        num_point_features_l += 6 if self.use_absolute_xyz else 3
        num_point_features_r += 6 if self.use_absolute_xyz else 3
        # center_x, center_y, center_z, mean_x, mean_y, mean_z we need 6 new
        if self.with_distance:
            num_point_features_l += 1
            num_point_features_r += 1
        if self.use_preground_score:
            num_point_features_r += 1
        # LiDAR : x y z cx cy cz dx dy dz I1
        # Radar : x y z cx cy cz dx dy dz I2 Score
        # Fusion : x y z Lcx Lcy Lcz mx my mz Rcx Rcy Rcz I1 I2 Score
        ex_point_features = num_point_features_l + num_point_features_r - 6
        num_point_features_r = ex_point_features
        num_point_features_l = ex_point_features
        self.num_point_features_r = num_point_features_r
        self.num_point_features_l = num_point_features_l
        # print("common feature dim (use preground_score) = ", num_point_features_r)








        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features_l] + list(self.num_filters)
        
        
        l_pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            l_pfn_layers.append(
                VMPLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.l_pfn_layers = nn.ModuleList(l_pfn_layers)

        self.num_filters = self.model_cfg.NUM_FILTERS_Radar
        assert len(self.num_filters) > 0
        num_filters = [num_point_features_r] + list(self.num_filters)
        
        
        r_pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            r_pfn_layers.append(
                VMPLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.r_pfn_layers = nn.ModuleList(r_pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]
        
    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        lidar_voxel_features, lidar_voxel_num_points, lidar_coords = batch_dict['lidar_voxels'], batch_dict['lidar_voxel_num_points'], batch_dict['lidar_voxel_coords']
        radar_voxel_features, radar_voxel_num_points, radar_coords = batch_dict['radar_voxels'], batch_dict['radar_voxel_num_points'], batch_dict['radar_voxel_coords']
        L_coords = lidar_coords[:,:] # lidar_coords랑 동일함
        R_coords = radar_coords[:,:]

        lidar_points_mean = lidar_voxel_features[:, :, :3].sum(dim=1, keepdim=True) / lidar_voxel_num_points.type_as(lidar_voxel_features).view(-1, 1, 1) # [n, 1, 3] # 각 복셀의 모든 point들의 평균 좌표
        radar_points_mean = radar_voxel_features[:, :, :3].sum(dim=1, keepdim=True) / radar_voxel_num_points.type_as(radar_voxel_features).view(-1, 1, 1)
        lidar_f_cluster = lidar_voxel_features[:, :, :3] - lidar_points_mean # 각 point가 속한 voxel 내의 평균 좌표와의 거리 # [N, 32, 3]
        radar_f_cluster = radar_voxel_features[:, :, :3] - radar_points_mean

        lidar_f_center = torch.zeros_like(lidar_voxel_features[:, :, :3]) 
        radar_f_center = torch.zeros_like(radar_voxel_features[:, :, :3])
        lidar_f_center[:, :, 0] = lidar_voxel_features[:, :, 0] - (lidar_coords[:, 3].to(lidar_voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        lidar_f_center[:, :, 1] = lidar_voxel_features[:, :, 1] - (lidar_coords[:, 2].to(lidar_voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        lidar_f_center[:, :, 2] = lidar_voxel_features[:, :, 2] - (lidar_coords[:, 1].to(lidar_voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)
        radar_f_center[:, :, 0] = radar_voxel_features[:, :, 0] - (radar_coords[:, 3].to(radar_voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        radar_f_center[:, :, 1] = radar_voxel_features[:, :, 1] - (radar_coords[:, 2].to(radar_voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        radar_f_center[:, :, 2] = radar_voxel_features[:, :, 2] - (radar_coords[:, 1].to(radar_voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)
        # lidar_f_center --> [4575, 32, 3] # 이거는, 각 점들과 점이 속한 voxel의 기하학적 중심과의 거리차이.
        dist_matrix = torch.sum((L_coords.unsqueeze(1) - R_coords)**2, dim=2)  # [4575, 3771] # i번째 lidar voxel과 j번째 radar voxel 간의 거리 (voxel 중심 간의 거리) 

        # 서로 일치하는 복셀들의 경우, 인덱스 반환
        common_L, common_R = torch.where(dist_matrix==0) # 372, 372

        # mask = torch.ones(len(L_coords)).bool() 
        # mask[common_L] = False
        # only_L = torch.where(mask)[0].long()
        # # 找到R中独有的点
        # mask = torch.ones(len(R_coords)).bool()
        # mask[common_R] = False
        # only_R = torch.where(mask)[0].long()

        
        
        #print(len(L_coords), len(R_coords), len(only_L), len(only_R), len(common_L), len(common_R))
        #接下来把Lidar合并到radar voxel（包括特征合并）
        len_radar = 1
        if len(radar_voxel_num_points) > 0:
            len_radar = int(radar_voxel_num_points.max())

        com_features = torch.zeros((len(radar_voxel_num_points), len_radar, self.num_point_features_r)).cuda() # [n, 20, 15]
        
        # len_radar는 현재 배치에서 가장 점이 많은 Radar Voxel의 점 개수
        for i in range(len_radar): # i는 0부터 len_radar - 1까지의 값으로, Voxel 내의 각 포인트 인덱스를 나타냅니다.
            
            # 1. 유효한 공통 Voxel 찾기
            now_feature_idx = 0
            valid_mask = radar_voxel_num_points[common_R] >= i+1 #只覆盖非空的点
            valid_common_R = common_R[valid_mask]
            valid_common_L = common_L[valid_mask]
            #print(radar_voxel_features[valid_common_R[0], i, :3])

            # 2. 융합 특성 생성
            com_features[:, i, now_feature_idx : now_feature_idx + 3] = radar_voxel_features[:, i, :3] #3
            now_feature_idx += 3

            #Intensity 覆盖为均值（radar部分的lidar特征设置为0）
            extraF_L = lidar_voxel_features[valid_common_L, :, 3:].sum(dim=1) / lidar_voxel_num_points[valid_common_L].type_as(lidar_voxel_features).view(-1, 1)
            com_features[valid_common_R, i, now_feature_idx : now_feature_idx + 1] = extraF_L #1
            now_feature_idx += 1
            
            # com_features[valid_common_L, replaced_idx, now_feature_idx : now_feature_idx + 1] = 0 #1
            # now_feature_idx += 1


            #radar to lidar偏移
            common_lidar_points_mean = lidar_voxel_features[valid_common_L, :, :3].sum(dim=1) / lidar_voxel_num_points[valid_common_L].type_as(lidar_voxel_features).view(-1, 1)
            radartolidar_f_cluster = radar_voxel_features[valid_common_R, i, :3] - common_lidar_points_mean
            com_features[valid_common_R, i, now_feature_idx : now_feature_idx + radartolidar_f_cluster.shape[-1]] = radartolidar_f_cluster #3
            now_feature_idx += radartolidar_f_cluster.shape[-1]
            
            com_features[:, i, now_feature_idx : now_feature_idx + radar_f_center.shape[-1]] = radar_f_center[:, i] #3
            now_feature_idx += radar_f_center.shape[-1]

            com_features[:, i, now_feature_idx : now_feature_idx + radar_f_cluster.shape[-1]] = radar_f_cluster[:, i] #3
            now_feature_idx += radar_f_cluster.shape[-1]

            
            #radar特征部分修改
            com_features[:, i, now_feature_idx : now_feature_idx + radar_voxel_features.shape[-1] - 3] = radar_voxel_features[:, i, 3:]
            now_feature_idx += radartolidar_f_cluster.shape[-1]
        
        len_lidar = int(lidar_voxel_num_points.max())
        l_ex_features = torch.zeros((len(lidar_voxel_num_points), 32, self.num_point_features_l)).cuda()
        now_feature_idx = 0
        l_ex_features[:, :, now_feature_idx : now_feature_idx + lidar_voxel_features.shape[-1]] = lidar_voxel_features #4
        now_feature_idx += lidar_voxel_features.shape[-1]
        #print(now_feature_idx)

        l_ex_features[:, :, now_feature_idx : now_feature_idx + lidar_f_cluster.shape[-1]] = lidar_f_cluster #3
        now_feature_idx += lidar_f_cluster.shape[-1]
        #print(now_feature_idx)

        l_ex_features[:, :, now_feature_idx : now_feature_idx + lidar_f_center.shape[-1]] = lidar_f_center #3
        now_feature_idx += lidar_f_center.shape[-1]
        #print(now_feature_idx)

        #计算lidar to radar共同部分中的cluster和feature均值用于特征传播
        #(N,3); (N,feature_dim-1)
        mask = self.get_paddings_indicator(radar_voxel_num_points[common_R], 32, axis=0)
        mask = mask & (radar_voxel_features[common_R, :, -2] == -1) 
        # 求valid且t=0的mask并求和算有多少个，t在-2维度
        num_valid = mask.sum(dim=1)
        l2r_mask = (num_valid > 0)

        l2r_com_L = common_L[l2r_mask]
        l2r_com_R = common_R[l2r_mask]
        num_valid = num_valid[l2r_mask]
        mask = mask[l2r_mask].unsqueeze(-1)
        #计算lidar to radar共同部分的cluster(注意只有common(L，R)的部分才有)

        common_radar_points_mean = (radar_voxel_features[l2r_com_R, :, :3] * mask).sum(dim=1, keepdim=True) / num_valid.type_as(radar_voxel_features).view(-1, 1, 1)
        lidartoradar_f_cluster = lidar_voxel_features[l2r_com_L, :, :3] - common_radar_points_mean

        l_ex_features[l2r_com_L, :, now_feature_idx : now_feature_idx + lidartoradar_f_cluster.shape[-1]] = lidartoradar_f_cluster #3
        now_feature_idx += lidartoradar_f_cluster.shape[-1]
        #radar特征部分先填充均值后面是radar的话会覆盖
        extraFea_R = (radar_voxel_features[l2r_com_R, :, 3:] * mask).sum(dim=1, keepdim=True) / num_valid.type_as(radar_voxel_features).view(-1, 1, 1)
        l_ex_features[l2r_com_L, :, now_feature_idx :] = extraFea_R #4
        now_feature_idx += extraFea_R.shape[-1]

        lidar_features = l_ex_features
        final_voxel_count = lidar_features.shape[1]
        mask = self.get_paddings_indicator(lidar_voxel_num_points, final_voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(lidar_features)
        lidar_features *= mask
        for pfn in self.l_pfn_layers:
            lidar_features = pfn(lidar_features) # (n, 32, 15) ---> (n, 1, 64)
        lidar_features = lidar_features.squeeze()

        radar_features = com_features
        final_voxel_count = radar_features.shape[1]
        mask = self.get_paddings_indicator(radar_voxel_num_points, final_voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(radar_features)
        radar_features *= mask
        for pfn in self.r_pfn_layers:
            radar_features = pfn(radar_features)
        radar_features = radar_features.squeeze()

        batch_dict['lidar_pillar_features'] = lidar_features # [n_l, 64]
        
        # 가끔가다가, radar denoising module에 의해 모든 레이더 포인트가 필터링될 때 있음. 그럴 땐 radar_features의 shape이 [64]임. 그래서 아래처럼 unsqueeze해야 한다.
        if radar_features.ndim == 1:
            radar_features = radar_features.unsqueeze(0) # [64] -> [1, 64]
        batch_dict['radar_pillar_features'] = radar_features # [n_r, 64]
       
        return batch_dict



class VMPLayer(nn.Module):
    """
    Voxel Mean Pooling Layer
        : performs voxel mean pooling only on points that are not zero.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part*self.part:(num_part+1)*self.part])
                               for num_part in range(num_parts+1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x)

        # --- MODIFICATION START ---
        # Create a mask to identify non-zero rows (valid points)
        # Assuming the first feature column is always non-zero for valid points.
        # This is a common practice. If not, use (x != 0).any(dim=-1, keepdim=True).
        valid_mask = (inputs[:, :, 0] != 0).unsqueeze(-1).float()
        
        # Apply the mask to zero out padded features
        x = x * valid_mask

        # Calculate the sum and count of valid points for each voxel
        # Use sum pooling to sum up the features of valid points
        x_sum = torch.sum(x, dim=1, keepdim=True)
        
        # Get the number of non-zero points per voxel
        valid_point_count = torch.sum(valid_mask, dim=1, keepdim=True)
        
        # Avoid division by zero
        valid_point_count[valid_point_count == 0] = 1.0 
        
        # Calculate the mean
        x_mean = x_sum / valid_point_count

        if self.last_vfe:
            return x_mean
        else:
            x_repeat = x_mean.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated
        # --- MODIFICATION END ---