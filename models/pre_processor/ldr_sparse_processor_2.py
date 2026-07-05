'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
* first edit: Yujeong Chae
* second edit: HJ Park
'''

import torch
import torch.nn as nn
import time


class LiDARSparseProcessor_2(nn.Module):
    """
    use this b4 encoding 3d lidar data for uem
    """
    def __init__(self,
                 model_cfg,
                 point_cloud_range # [ 0. , -6.4, -2. , 72. ,  6.4,  6. ]
                 ):
        super(LiDARSparseProcessor_2, self).__init__()
        self.model_cfg = model_cfg

        x_min, x_max = point_cloud_range[0], point_cloud_range[3]
        y_min, y_max = point_cloud_range[1], point_cloud_range[4]
        z_min, z_max = point_cloud_range[2], point_cloud_range[5]
        self.min_roi = [x_min, y_min, z_min] # [0.0, -6.4, -2.0]

        self.grid_size = model_cfg.GRID_SIZE # 0.4
        self.input_dim = model_cfg.INPUT_DIM # 4


    def forward(self, batch_dict):
        
        
        sp_cube = batch_dict['ldr64_uem'].cuda() # [N, 9]
        sp_indices = batch_dict['batch_indices_ldr64_uem'].cuda() # [N]

        sp_cube = sp_cube[:,:self.input_dim] # [N, 9] -> [N, 4] # x, y, z, intensity

        # Get z, y, x coord
        x_min, y_min, z_min = self.min_roi # [0.0, -6.4, -2.0]
        grid_size = self.grid_size
        
        x_coord, y_coord, z_coord = sp_cube[:, 0:1], sp_cube[:, 1:2], sp_cube[:, 2:3]

        
        # z_ind = torch.ceil((z_coord-z_min) / grid_size).long()
        # y_ind = torch.ceil((y_coord-y_min) / grid_size).long()
        # x_ind = torch.ceil((x_coord-x_min) / grid_size).long()
        z_ind = torch.floor((z_coord-z_min) / grid_size).long() # NOTE 위에 보이는 게 기존 코드. 기존 코드로하면 idx가 1로 시작함. floor()로 수정해서 0으로 시작하게 만듦
        y_ind = torch.floor((y_coord-y_min) / grid_size).long()
        x_ind = torch.floor((x_coord-x_min) / grid_size).long()
            
        sp_indices = torch.cat((sp_indices.unsqueeze(-1), z_ind, y_ind, x_ind), dim = -1) # [N, 4]

        batch_dict['sp_features_l'] = sp_cube # [N, 4]
        batch_dict['sp_indices_l'] = sp_indices # [N, 4]
        
        return batch_dict
