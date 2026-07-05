import numpy as np
import torch
import torch.nn as nn


class BaseBEVBackbone_MGF_RT(nn.Module):
    """
    RT는 radar teacher를 의미함.
    RP는 radar_plus feature를 의미함.
    Radar -> Radar+를 만들기 위한 Radar Teacher 클래스
    Radar-only encoding으로 spatial_features_2d_rp 생성
    Output shape: spatial_features_2d와 동일 (768ch)
    """
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)

        # Radar-only: input_channels는 radar의 channel (64)
        c_in_list = [input_channels, *num_filters[:-1]]

        self.R_blocks = nn.ModuleList()
        self.R_deblocks = nn.ModuleList()

        for idx in range(num_levels):
            # Radar encoding blocks
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.R_blocks.append(nn.Sequential(*cur_layers))

            # Radar upsample blocks (output: num_upsample_filters[idx] for 768ch total)
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride > 1 or (stride == 1 and not self.model_cfg.get('USE_CONV_FOR_NO_STRIDE', False)):
                    self.R_deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int32)
                    self.R_deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        # Output channels: sum of num_upsample_filters (should be 768)
        self.num_bev_features = sum(num_upsample_filters)

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                radar_spatial_features: [B, 64, H, W]
        Returns:
            data_dict:
                spatial_features_2d_rp: [B, 768, H', W'] (radar plus feature)
        """
        radar_x = data_dict['radar_spatial_features']
        ups = []

        for i in range(len(self.R_blocks)):
            radar_x = self.R_blocks[i](radar_x)
            if len(self.R_deblocks) > 0:
                ups.append(self.R_deblocks[i](radar_x))
            else:
                ups.append(radar_x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]
        
        # Output: radar plus feature
        data_dict['spatial_features_2d_rp'] = x # [B, 768, 100, 240]

        return data_dict
