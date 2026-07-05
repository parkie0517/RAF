'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
* modified from: lrdr_sp_pw.py
* eddited: HJ
'''

import torch
import torch.nn as nn

import spconv.pytorch as spconv
from einops.layers.torch import Rearrange

class SparseBackbone(nn.Module):
    """
    SparseBackbone
        for encoding 3D lidar and radar voxel features
        and returning lidar and radar BEV maps
    """
    def __init__(self,
                model_cfg,
                input_dim,
                roi,
                voxel_size,
                 ):
        super(SparseBackbone, self).__init__()
        self.model_cfg = model_cfg
        
        x_min, x_max, y_min, y_max, z_min, z_max = roi # [0.,-6.4,-2.,72.,6.4,6.0]

        z_shape = int(round((z_max-z_min) / voxel_size[2]))
        y_shape = int(round((y_max-y_min) / voxel_size[1]))
        x_shape = int(round((x_max-x_min) / voxel_size[0]))

        self.spatial_shape = [z_shape, y_shape, x_shape]
        self.input_dim = input_dim # 64

        list_enc_channel = self.model_cfg.ENCODING.CHANNEL # [64, 128, 64] 
        list_enc_padding = self.model_cfg.ENCODING.PADDING # [1, 1, 1] 
        list_enc_stride  = self.model_cfg.ENCODING.STRIDE # [1, 1, 1] 
        
        # 1x1 conv / 4->ENCODING.CHANNEL[0]
        self.l_input_conv = spconv.SparseConv3d(
            in_channels=self.input_dim, out_channels=list_enc_channel[0],
            kernel_size=1, stride=1, padding=0, dilation=1, indice_key = 'sp0') 
        self.r_input_conv = spconv.SparseConv3d(
            in_channels=self.input_dim, out_channels=list_enc_channel[0],
            kernel_size=1, stride=1, padding=0, dilation=1, indice_key = 'sp0') 
        # encoder
        self.num_layer = len(list_enc_channel)
        for idx_enc in range(self.num_layer):
            if idx_enc == 0:
                temp_in_ch = list_enc_channel[0] # [64, 128, 64]
            else:
                temp_in_ch = list_enc_channel[idx_enc-1] # in [64, 128, 64]
            temp_ch = list_enc_channel[idx_enc]
            temp_pd = list_enc_padding[idx_enc]
            setattr(self, f'l_spconv{idx_enc}', \
                spconv.SparseConv3d(in_channels=temp_in_ch, out_channels=temp_ch, kernel_size=3, \
                    stride=list_enc_stride[idx_enc], padding=temp_pd, dilation=1, indice_key=f'sp{idx_enc}'))
            setattr(self, f'l_bn{idx_enc}', nn.BatchNorm1d(temp_ch))
            setattr(self, f'l_subm{idx_enc}a', \
                spconv.SubMConv3d(in_channels=temp_ch, out_channels=temp_ch, kernel_size=3, stride=1, padding=0, dilation=1, indice_key=f'subm{idx_enc}'))
            setattr(self, f'l_bn{idx_enc}a', nn.BatchNorm1d(temp_ch))
            setattr(self, f'l_subm{idx_enc}b', \
                spconv.SubMConv3d(in_channels=temp_ch, out_channels=temp_ch, kernel_size=3, stride=1, padding=0, dilation=1, indice_key=f'subm{idx_enc}'))
            setattr(self, f'l_bn{idx_enc}b', nn.BatchNorm1d(temp_ch))
            setattr(self, f'r_spconv{idx_enc}', \
                spconv.SparseConv3d(in_channels=temp_in_ch, out_channels=temp_ch, kernel_size=3, \
                    stride=list_enc_stride[idx_enc], padding=temp_pd, dilation=1, indice_key=f'sp{idx_enc}'))
            setattr(self, f'r_bn{idx_enc}', nn.BatchNorm1d(temp_ch))
            setattr(self, f'r_subm{idx_enc}a', \
                spconv.SubMConv3d(in_channels=temp_ch, out_channels=temp_ch, kernel_size=3, stride=1, padding=0, dilation=1, indice_key=f'subm{idx_enc}'))
            setattr(self, f'r_bn{idx_enc}a', nn.BatchNorm1d(temp_ch))
            setattr(self, f'r_subm{idx_enc}b', \
                spconv.SubMConv3d(in_channels=temp_ch, out_channels=temp_ch, kernel_size=3, stride=1, padding=0, dilation=1, indice_key=f'subm{idx_enc}'))
            setattr(self, f'r_bn{idx_enc}b', nn.BatchNorm1d(temp_ch))
        
        # activation
        self.relu = nn.ReLU()
        

    def forward(self, dict_item):
        lidar_features, lidar_indices = dict_item['lidar_voxel_features'], dict_item['lidar_voxel_coords']
        radar_features, radar_indices = dict_item['radar_voxel_features'], dict_item['radar_voxel_coords']
        l_input_sp_tensor = spconv.SparseConvTensor(
            features=lidar_features,
            indices=lidar_indices.int(),
            spatial_shape=self.spatial_shape,
            batch_size=dict_item['batch_size']
        )
        l_x = self.l_input_conv(l_input_sp_tensor)
        r_input_sp_tensor = spconv.SparseConvTensor(
            features=radar_features,
            indices=radar_indices.int(),
            spatial_shape=self.spatial_shape,
            batch_size=dict_item['batch_size']
        )
        r_x = self.r_input_conv(r_input_sp_tensor)

        # print(x.dense().shape)

        list_bev_features = []
        x = l_x
        for idx_layer in range(self.num_layer):
            # print(idx_layer)
            
            x = getattr(self, f'l_spconv{idx_layer}')(x)
            x = x.replace_feature(getattr(self, f'l_bn{idx_layer}')(x.features))
            x = x.replace_feature(self.relu(x.features))
            x = getattr(self, f'l_subm{idx_layer}a')(x)
            x = x.replace_feature(getattr(self, f'l_bn{idx_layer}a')(x.features))
            x = x.replace_feature(self.relu(x.features))
            x = getattr(self, f'l_subm{idx_layer}b')(x)
            x = x.replace_feature(getattr(self, f'l_bn{idx_layer}b')(x.features))
            x = x.replace_feature(self.relu(x.features))
            # print(x.dense().shape)

            if self.is_z_embed:
                bev_dense = getattr(self, f'chzcat{idx_layer}')(x.dense())
                bev_dense = getattr(self, f'convtrans2d{idx_layer}')(bev_dense)
            else:
                bev_sp = getattr(self, f'l_toBEV{idx_layer}')(x)
                bev_sp = bev_sp.replace_feature(getattr(self, f'l_bnBEV{idx_layer}')(bev_sp.features))
                bev_sp = bev_sp.replace_feature(self.relu(bev_sp.features))
                # print(bev_sp.dense().shape)

                # B, C, 1, Y/st, X/st -> B, C, Y, X
                bev_dense = getattr(self, f'l_convtrans2d{idx_layer}')(bev_sp.dense().squeeze(2))
            
            bev_dense = getattr(self, f'l_bnt{idx_layer}')(bev_dense)
            bev_dense = self.relu(bev_dense)

            list_bev_features.append(bev_dense)

        x = r_x
        for idx_layer in range(self.num_layer):
            # print(idx_layer)
            
            x = getattr(self, f'r_spconv{idx_layer}')(x)
            x = x.replace_feature(getattr(self, f'r_bn{idx_layer}')(x.features))
            x = x.replace_feature(self.relu(x.features))
            x = getattr(self, f'r_subm{idx_layer}a')(x)
            x = x.replace_feature(getattr(self, f'r_bn{idx_layer}a')(x.features))
            x = x.replace_feature(self.relu(x.features))
            x = getattr(self, f'r_subm{idx_layer}b')(x)
            x = x.replace_feature(getattr(self, f'r_bn{idx_layer}b')(x.features))
            x = x.replace_feature(self.relu(x.features))
            # print(x.dense().shape)

            if self.is_z_embed:
                bev_dense = getattr(self, f'chzcat{idx_layer}')(x.dense())
                bev_dense = getattr(self, f'convtrans2d{idx_layer}')(bev_dense)
            else:
                bev_sp = getattr(self, f'r_toBEV{idx_layer}')(x)
                bev_sp = bev_sp.replace_feature(getattr(self, f'r_bnBEV{idx_layer}')(bev_sp.features))
                bev_sp = bev_sp.replace_feature(self.relu(bev_sp.features))
                # print(bev_sp.dense().shape)

                # B, C, 1, Y/st, X/st -> B, C, Y, X
                bev_dense = getattr(self, f'r_convtrans2d{idx_layer}')(bev_sp.dense().squeeze(2))
            
            bev_dense = getattr(self, f'r_bnt{idx_layer}')(bev_dense)
            bev_dense = self.relu(bev_dense)

            list_bev_features.append(bev_dense)

        bev_features = torch.cat(list_bev_features, dim = 1)
        # print(bev_features.shape)
        dict_item['bev_feat'] = bev_features # B, C, Y, X

        return dict_item
