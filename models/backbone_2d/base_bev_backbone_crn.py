import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F


class BaseBEVBackbone_MF_RC(nn.Module):
    """
    Radar + Camera only fusion using 2-stage cross-attention.
    No LiDAR is used in this module.

    Architecture:
    - Step A: Build shared query Q from concatenated radar+camera features
    - Step B: Two separate cross-attentions with same Q (radar as K/V, camera as K/V)
             Uses spatial pooling to reduce memory for attention computation.
    - Step C: Fuse outputs by summation
    - Step D: BEV neck to produce final [B, 768, H, W] output
    """
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        # input_channels = 128 (64 radar + 64 camera)
        self.radar_channels = input_channels // 2  # 64
        self.camera_channels = input_channels // 2  # 64
        self.hidden_channels = 256  # intermediate dim for attention
        self.output_channels = 768  # final output channels to match downstream head
        self.num_heads = 8

        # Spatial pooling factor for memory-efficient attention
        # Input: [80, 480] -> Pooled: [20, 60] = 1200 tokens (manageable for attention)
        self.pool_size = (20, 60)

        # Step A: Query projection (from concatenated radar+camera)
        self.proj_q = nn.Conv2d(input_channels, self.hidden_channels, kernel_size=1, bias=False)
        self.proj_q_norm = nn.BatchNorm2d(self.hidden_channels)

        # Step B: Key/Value projections for radar cross-attention
        self.proj_k_radar = nn.Conv2d(self.radar_channels, self.hidden_channels, kernel_size=1, bias=False)
        self.proj_v_radar = nn.Conv2d(self.radar_channels, self.hidden_channels, kernel_size=1, bias=False)
        self.proj_k_radar_norm = nn.BatchNorm2d(self.hidden_channels)
        self.proj_v_radar_norm = nn.BatchNorm2d(self.hidden_channels)

        # Step B: Key/Value projections for camera cross-attention
        self.proj_k_cam = nn.Conv2d(self.camera_channels, self.hidden_channels, kernel_size=1, bias=False)
        self.proj_v_cam = nn.Conv2d(self.camera_channels, self.hidden_channels, kernel_size=1, bias=False)
        self.proj_k_cam_norm = nn.BatchNorm2d(self.hidden_channels)
        self.proj_v_cam_norm = nn.BatchNorm2d(self.hidden_channels)

        # Multi-head attention layers
        self.radar_cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_channels,
            num_heads=self.num_heads,
            batch_first=True,
            dropout=0.1
        )
        self.camera_cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_channels,
            num_heads=self.num_heads,
            batch_first=True,
            dropout=0.1
        )

        # Layer norms after attention
        self.radar_attn_norm = nn.LayerNorm(self.hidden_channels)
        self.camera_attn_norm = nn.LayerNorm(self.hidden_channels)

        # Post-attention conv to restore resolution (to intermediate H/2, W/2)
        self.upsample_conv = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.hidden_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        # Step D: BEV neck - simple CNN to further encode fused features
        # Note: We output at half resolution to match feature_map_stride=2
        # Input: [B, 256, H, W] -> Output: [B, 768, H/2, W/2]
        # First conv with stride 2 to downsample from [80, 480] to [40, 240]
        self.bev_neck = nn.Sequential(
            # Stride-2 conv to downsample spatial dimensions
            nn.Conv2d(self.hidden_channels, 384, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(384, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, self.output_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.output_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        # Output feature dimension for downstream modules
        self.num_bev_features = self.output_channels  # 768

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                radar_spatial_features: [B, 64, H, W]
                camera_spatial_features: [B, 64, H, W]
        Returns:
            data_dict with 'spatial_features_2d': [B, 768, H, W]
        """
        # Get radar and camera BEV features (no LiDAR)
        radar_feat = data_dict['radar_spatial_features']   # [B, 64, H, W]
        cam_feat = data_dict['camera_spatial_features']    # [B, 64, H, W]


        B, _, H, W = radar_feat.shape

        # Step A: Build shared query Q from concatenated features
        q_in = torch.cat([radar_feat, cam_feat], dim=1)  # [B, 128, H, W]
        Q = self.proj_q(q_in)  # [B, 256, H, W]
        Q = self.proj_q_norm(Q)

        # Project K and V for radar
        K_radar = self.proj_k_radar_norm(self.proj_k_radar(radar_feat))  # [B, 256, H, W]
        V_radar = self.proj_v_radar_norm(self.proj_v_radar(radar_feat))  # [B, 256, H, W]

        # Project K and V for camera
        K_cam = self.proj_k_cam_norm(self.proj_k_cam(cam_feat))  # [B, 256, H, W]
        V_cam = self.proj_v_cam_norm(self.proj_v_cam(cam_feat))  # [B, 256, H, W]

        # Spatially pool for memory-efficient attention
        # Pool from [40, 240] to [10, 60] = 600 tokens
        Q_pooled = F.adaptive_avg_pool2d(Q, self.pool_size)  # [B, 256, 10, 60]
        K_radar_pooled = F.adaptive_avg_pool2d(K_radar, self.pool_size)
        V_radar_pooled = F.adaptive_avg_pool2d(V_radar, self.pool_size)
        K_cam_pooled = F.adaptive_avg_pool2d(K_cam, self.pool_size)
        V_cam_pooled = F.adaptive_avg_pool2d(V_cam, self.pool_size)

        H_pool, W_pool = self.pool_size

        # Reshape for attention: [B, C, H, W] -> [B, H*W, C]
        Q_flat = Q_pooled.flatten(2).permute(0, 2, 1)  # [B, 600, 256]

        # Step B1: Cross-attention with radar as Key/Value
        K_radar_flat = K_radar_pooled.flatten(2).permute(0, 2, 1)  # [B, 600, 256]
        V_radar_flat = V_radar_pooled.flatten(2).permute(0, 2, 1)  # [B, 600, 256]

        radar_attn_out, _ = self.radar_cross_attn(Q_flat, K_radar_flat, V_radar_flat)  # [B, 600, 256]
        radar_attn_out = self.radar_attn_norm(radar_attn_out)

        # Step B2: Cross-attention with camera as Key/Value
        K_cam_flat = K_cam_pooled.flatten(2).permute(0, 2, 1)  # [B, 600, 256]
        V_cam_flat = V_cam_pooled.flatten(2).permute(0, 2, 1)  # [B, 600, 256]

        cam_attn_out, _ = self.camera_cross_attn(Q_flat, K_cam_flat, V_cam_flat)  # [B, 600, 256]
        cam_attn_out = self.camera_attn_norm(cam_attn_out)

        # Step C: Fuse the two attention outputs by summation
        fused_attn = radar_attn_out + cam_attn_out  # [B, 600, 256]

        # Reshape back to BEV format: [B, H*W, C] -> [B, C, H_pool, W_pool]
        fused_bev_pooled = fused_attn.permute(0, 2, 1).view(B, self.hidden_channels, H_pool, W_pool)  # [B, 256, 10, 60]

        # Upsample back to original resolution
        fused_bev = F.interpolate(fused_bev_pooled, size=(H, W), mode='bilinear', align_corners=False)  # [B, 256, 40, 240]

        # Post-upsample conv for refinement
        fused_bev = self.upsample_conv(fused_bev)  # [B, 256, 40, 240]

        # Step D: Apply BEV neck
        x = self.bev_neck(fused_bev)  # [B, 768, 40, 240]

        # Step E: Store final output
        data_dict['spatial_features_2d'] = x  # [B, 768, 40, 240]

        return data_dict