from typing import Tuple

import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32
from torch import nn

# from mmdet3d.models.builder import VTRANSFORMS # krdar는 mmdet3d 폴더가 없음
from models.builder import VTRANSFORMS

from .base import BaseDepthTransform

__all__ = ["DepthLSSTransform"]


@VTRANSFORMS.register_module()
class DepthLSSTransform(BaseDepthTransform):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        downsample: int = 1,
        depth_supervision: bool = False,
        depth_weight: float = 3.0,
        depth_sigma: float = 1.0,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
        )
        self.depth_supervision = depth_supervision
        self.depth_weight = depth_weight
        self.depth_sigma = depth_sigma
        if self.depth_supervision:
            ds_h = image_size[0] // feature_size[0]
            ds_w = image_size[1] // feature_size[1]
            self._downsample_factor = (ds_h, ds_w)
        self.dtransform = nn.Sequential(
            nn.Conv2d(1, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 32, 5, stride=4, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.depthnet = nn.Sequential(
            nn.Conv2d(in_channels + 64, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, self.D + self.C, 1),
        )
        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    stride=downsample,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

    @force_fp32()
    def get_cam_feats(self, x, d): # image feature, depth
        B, N, C, fH, fW = x.shape # [4, 1, 256,  80,  160]

        d = d.view(B * N, *d.shape[2:])
        x = x.view(B * N, C, fH, fW)

        d = self.dtransform(d)
        x = torch.cat([d, x], dim=1) # d는 positional embedding 같은 역할 수행 있으나 마나 상관 없음. 물론 없앨 거면, init에서 모델 정의할 때 수정 필요
        x = self.depthnet(x)

        depth = x[:, : self.D].softmax(dim=1)
        x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2) # 확률 분포 x feature

        x = x.view(B, N, self.C, self.D, fH, fW) # self.C는 output channel임
        x = x.permute(0, 1, 3, 4, 5, 2)

        if self.depth_supervision:
            depth = depth.view(B, N, self.D, fH, fW)
            return x, depth
        return x

    def _depth_to_soft_onehot(self, depth):
        """Convert sparse depth map to soft one-hot targets using Gaussian smoothing."""
        B, N, fH, fW = depth.shape
        bins = torch.arange(self.D, device=depth.device).view(1, 1, self.D, 1, 1)
        bin_coord = (depth - self.dbound[0]) / self.dbound[2]
        valid = (depth > 0) & (bin_coord >= 0) & (bin_coord <= self.D - 1)
        bin_coord = bin_coord.unsqueeze(2)
        dist = bins - bin_coord
        sigma = max(self.depth_sigma, 1e-6)
        gauss = torch.exp(-0.5 * (dist / sigma) ** 2)
        gauss = gauss * valid.unsqueeze(2).float()
        denom = gauss.sum(dim=2, keepdim=True).clamp(min=1e-6)
        return gauss / denom

    def _build_depth_gt_from_map(self):
        """Build depth GT by downsampling the sparse depth map already computed in base forward.

        Reuses self._sparse_depth_map (set by BaseDepthTransform.forward) instead of
        re-projecting LiDAR points. Uses vectorized min-pooling — no Python loops.
        """
        # _sparse_depth_map: [B, N, C, H, W] where C=1 for scalar depth
        depth_img = self._sparse_depth_map[:, :, 0, :, :]  # [B, N, H, W]
        # Replace 0 (empty) with inf so min-pool ignores them
        depth_img = depth_img.clone()
        depth_img[depth_img == 0] = float('inf')
        # Min-pool to feature resolution via: min(x) = -max(-x)
        ds_h, ds_w = self._downsample_factor
        BN_shape = depth_img.shape[:2]
        depth_flat = depth_img.view(-1, 1, depth_img.shape[2], depth_img.shape[3])  # [B*N, 1, H, W]
        depth_feat = -F.max_pool2d(-depth_flat, kernel_size=(ds_h, ds_w), stride=(ds_h, ds_w))
        depth_feat = depth_feat.view(*BN_shape, depth_feat.shape[2], depth_feat.shape[3])  # [B, N, fH, fW]
        # Replace inf (no valid depth) back to 0
        depth_feat[depth_feat == float('inf')] = 0.0
        return self._depth_to_soft_onehot(depth_feat)

    def forward(self, *args, **kwargs):
        result = super().forward(*args, **kwargs)
        if isinstance(result, tuple):
            bev, depth_prob = result
            bev = self.downsample(bev)
            if self.training:
                depth_gt = self._build_depth_gt_from_map()
                return bev, depth_prob, depth_gt
            return bev
        else:
            return self.downsample(result)