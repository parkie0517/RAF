import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F


class BaseBEVBackbone_Robu_CR(nn.Module):
    """
    Patch-based cross-attention fusion across radar and camera BEV features,
    followed by convolutional encoding (mirrors the backbone stages used in MF_cam).
    """
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        # Use equal channel split across two modalities (radar, camera)
        input_channels = 64 + 64 # hard coded
        assert input_channels % 2 == 0, "Input channels must be divisible by 2 for ASF fusion."
        self.single_in_channels = input_channels // 2

        # Patch settings (keeps spatial grid 40x240 from 80x480 with 2x2 patches)
        self.patch_size = (2, 2)
        self.patch_area = self.patch_size[0] * self.patch_size[1]

        # Feature dimensions
        # self.c_u = 256
        # self.num_queries = 18  # matches desired fused channel width before conv
        # self.num_heads = 16
        self.c_u = 128
        self.num_queries = 16  # matches desired fused channel width before conv
        self.num_heads = 8

        assert self.c_u % self.patch_area == 0, "C_u must be divisible by patch area."
        self.c_q = self.c_u // self.patch_area  # 64 when C_u=256 and patch=2x2, 32 when C_u=128 and patch=2x2,

        # Per-sensor patch encoders
        in_dim = self.single_in_channels * self.patch_area
        self.ln_s = nn.LayerNorm(in_dim)
        self.mlp_s = nn.Sequential(
            nn.Linear(in_dim, self.c_u),
            nn.GELU(),
            nn.Linear(self.c_u, self.c_u),
            nn.GELU(),
        )
        self.ln_s_out = nn.LayerNorm(self.c_u)

        # Cross-attention
        self.q_ref = nn.Parameter(torch.randn(self.num_queries, self.c_u))
        self.attn = nn.MultiheadAttention(embed_dim=self.c_u, num_heads=self.num_heads, batch_first=True)

        # Post-attention normalization
        self.ln_post = nn.LayerNorm(self.c_u)
        self.mlp_post = nn.Sequential(
            nn.Linear(self.c_u, self.c_u),
            nn.GELU(),
            nn.Linear(self.c_u, self.c_u),
            nn.GELU(),
        )
        self.ln_post_out = nn.LayerNorm(self.c_u)

        # Encoded patch channel dim and final output channel dim for the head
        # self.encoder_out_channels = self.num_queries * self.c_q  # 18 * 64 = 1152
        # self.num_bev_features = 1152  # keep downstream head interface consistent
        self.encoder_out_channels = self.num_queries * self.c_q  # 16 * 32 = 512
        self.num_bev_features = 512  # keep downstream head interface consistent

        # Convolutional encoder (single-branch) mirroring BaseBEVBackbone
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
        c_in_list = [self.encoder_out_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
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
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride > 1 or (stride == 1 and not self.model_cfg.get('USE_CONV_FOR_NO_STRIDE', False)):
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters) if len(upsample_strides) > 0 else (num_filters[-1] if num_filters else self.num_bev_features)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        # Project back to expected channel size for the head (1152 by construction)
        self.final_proj = nn.Sequential(
            nn.Conv2d(c_in, self.num_bev_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.num_bev_features, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )

    def _patchify(self, x):
        """
        x: [B, C, H, W]
        returns: [B, N_p, C*P_H*P_W] where N_p = (H/P_H) * (W/P_W)
        """
        B, C, H, W = x.shape
        ph, pw = self.patch_size
        assert H % ph == 0 and W % pw == 0, "Feature map must be divisible by patch size."
        x = x.view(B, C, H // ph, ph, W // pw, pw)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()  # [B, H', W', C, ph, pw]
        x = x.view(B, (H // ph) * (W // pw), C * ph * pw)
        return x, H // ph, W // pw

    def _encode_sensor(self, x):
        # x: [B, N_p, C*P_H*P_W]
        x = self.ln_s(x)
        x = self.mlp_s(x)
        x = self.ln_s_out(x)
        return x  # [B, N_p, C_u]

    def forward(self, data_dict):
        radar_x = data_dict['radar_spatial_features'] # [B, 64, 80, 480]
        camera_x = data_dict['camera_spatial_features'] # [B, 64, 80, 480]

        # Patchify each modality
        radar_tokens, h_patch, w_patch = self._patchify(radar_x)
        camera_tokens, _, _ = self._patchify(camera_x)

        # Encode patches to shared dimension
        radar_tokens = self._encode_sensor(radar_tokens)
        camera_tokens = self._encode_sensor(camera_tokens)

        # Stack per patch: [B, N_p, 2, C_u]
        fused_tokens = torch.stack([camera_tokens, radar_tokens], dim=2)
        B, N_p, _, _ = fused_tokens.shape

        # Prepare attention inputs
        kv = fused_tokens.view(B * N_p, 2, self.c_u)  # [B*N_p, 2, C_u]
        q = self.q_ref.unsqueeze(0).expand(B * N_p, -1, -1)  # [B*N_p, N_q, C_u]

        attn_out, _ = self.attn(q, kv, kv)  # [B*N_p, N_q, C_u]

        # Post-feature normalization
        attn_out = self.ln_post(attn_out)
        attn_out = self.mlp_post(attn_out)
        attn_out = self.ln_post_out(attn_out)

        # Reshape back to BEV grid of patches
        attn_out = attn_out.view(B, h_patch, w_patch, self.num_queries, self.c_u)  # [B, H', W', N_q, C_u]

        # Split C_u into (C_q, patch_area) and average over patch elements
        attn_out = attn_out.view(B, h_patch, w_patch, self.num_queries, self.c_q, self.patch_area)
        attn_out = attn_out.mean(dim=-1)  # [B, H', W', N_q, C_q]

        # Merge queries and channels, permute to NCHW
        fused_bev = attn_out.view(B, h_patch, w_patch, self.num_queries * self.c_q)
        fused_bev = fused_bev.permute(0, 3, 1, 2).contiguous()  # [B, 1152, H', W']

        # Convolutional encoding (single branch)
        x = fused_bev
        ups = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        x = self.final_proj(x)
        # print(x.shape)
        # breakpoint()
        data_dict['spatial_features_2d'] = x # [B, 512, 40, 240]
        return data_dict
