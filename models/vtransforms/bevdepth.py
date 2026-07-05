from typing import Dict, Tuple

import torch
import time
from mmcv.runner import force_fp32
from torch import nn

from models.builder import VTRANSFORMS
from .base import BaseTransform

__all__ = ["BEVDepthTransform"]


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act = nn.ReLU()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class DepthNetLite(nn.Module):
    def __init__(self, in_channels, mid_channels, context_channels,
                 depth_channels, use_cam_aware=True):
        super().__init__()
        self.use_cam_aware = use_cam_aware
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.context_conv = nn.Conv2d(
            mid_channels, context_channels, kernel_size=1, padding=0)
        self.depth_conv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, depth_channels, kernel_size=1, padding=0),
        )

        if self.use_cam_aware:
            self.mlp_dim = 31
            self.bn = nn.BatchNorm1d(self.mlp_dim)
            self.depth_mlp = Mlp(self.mlp_dim, mid_channels, mid_channels)
            self.depth_se = SELayer(mid_channels)
            self.context_mlp = Mlp(self.mlp_dim, mid_channels, mid_channels)
            self.context_se = SELayer(mid_channels)

    def _build_mlp_input(self, mats_dict: Dict[str, torch.Tensor]):
        intrin = mats_dict["intrin_mats"]  # B, N, 4, 4
        ida = mats_dict["ida_mats"]  # B, N, 4, 4
        sensor2ego = mats_dict["sensor2ego_mats"]  # B, N, 4, 4
        bda = mats_dict.get("bda_mat", None)  # B, 4, 4
        if bda is None:
            bda = intrin.new_zeros((intrin.shape[0], 4, 4))
            bda[:, 0, 0] = 1
            bda[:, 1, 1] = 1
            bda[:, 2, 2] = 1
            bda[:, 3, 3] = 1
        bda = bda.view(bda.shape[0], 1, 4, 4).repeat(1, intrin.shape[1], 1, 1)

        mlp_input = torch.cat(
            [
                torch.stack(
                    [
                        intrin[..., 0, 0],
                        intrin[..., 1, 1],
                        intrin[..., 0, 2],
                        intrin[..., 1, 2],
                        ida[..., 0, 0],
                        ida[..., 0, 1],
                        ida[..., 0, 3],
                        ida[..., 1, 0],
                        ida[..., 1, 1],
                        ida[..., 1, 3],
                        bda[..., 0, 0],
                        bda[..., 0, 1],
                        bda[..., 1, 0],
                        bda[..., 1, 1],
                        bda[..., 2, 2],
                    ],
                    dim=-1,
                ),
                sensor2ego.view(sensor2ego.shape[0], sensor2ego.shape[1], -1),
            ],
            dim=-1,
        )
        return mlp_input

    def forward(self, x, mats_dict=None):
        x = self.reduce_conv(x)
        if self.use_cam_aware and mats_dict is not None:
            mlp_input = self._build_mlp_input(mats_dict)
            mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
            context_se = self.context_mlp(mlp_input)[..., None, None]
            context = self.context_se(x, context_se)
            context = self.context_conv(context)
            depth_se = self.depth_mlp(mlp_input)[..., None, None]
            depth = self.depth_se(x, depth_se)
            depth = self.depth_conv(depth)
        else:
            context = self.context_conv(x)
            depth = self.depth_conv(x)
        return depth, context


@VTRANSFORMS.register_module()
class BEVDepthTransform(BaseTransform):
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
        depthnet_cfg: Dict = None,
        loss_cfg: Dict = None,
        downsample: int = 1,
        **kwargs,
    ) -> None:
        if loss_cfg is None and "LOSS" in kwargs:
            loss_cfg = kwargs["LOSS"]
        if depthnet_cfg is None and "DEPTHNET" in kwargs:
            depthnet_cfg = kwargs["DEPTHNET"]

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
        depthnet_cfg = depthnet_cfg or {}
        loss_cfg = loss_cfg or {}
        mid_channels = depthnet_cfg.get("mid_channels", in_channels)
        use_cam_aware = depthnet_cfg.get("use_cam_aware", True)
        self.depthnet = DepthNetLite(
            in_channels, mid_channels, self.C, self.D, use_cam_aware=use_cam_aware
        )

        self.loss_cfg = loss_cfg
        self.depth_supervision = loss_cfg.get("DEPTH_SUPERVISION", False)
        self.depth_weight = loss_cfg.get("DEPTH_SUPERVISION_WEIGHT", 1.0)
        self.depth_min_dist = loss_cfg.get("DEPTH_MIN_DIST", 0.0)
        self.depth_gt_type = loss_cfg.get("DEPTH_GT_TYPE", "soft_onehot")
        self.depth_sigma = loss_cfg.get("DEPTH_GT_GAUSSIAN_SIGMA", 1.0)
        self.depth_gt_single_frame = loss_cfg.get("DEPTH_GT_SINGLE_FRAME", True)
        self.depth_gt_inference = loss_cfg.get("DEPTH_GT_INFERENCE", False)
        self.debug_timing = loss_cfg.get("DEBUG_TIMING", False)
        self.debug_timing_interval = loss_cfg.get("DEBUG_TIMING_INTERVAL", 50)
        self._timing_count = 0
        self._timing_acc = {"cam": 0.0, "bev": 0.0, "gt": 0.0, "total": 0.0}

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

        ds_h = image_size[0] // feature_size[0]
        ds_w = image_size[1] // feature_size[1]
        assert ds_h * feature_size[0] == image_size[0]
        assert ds_w * feature_size[1] == image_size[1]
        self.downsample_factor = (ds_h, ds_w)

    @force_fp32()
    def get_cam_feats(self, x, mats_dict):
        B, N, C, fH, fW = x.shape
        x = x.view(B * N, C, fH, fW)
        depth_logits, context = self.depthnet(x, mats_dict)
        depth_prob = depth_logits.softmax(dim=1)
        x = depth_prob.unsqueeze(1) * context.unsqueeze(2)
        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        depth_prob = depth_prob.view(B, N, self.D, fH, fW)
        return x, depth_prob

    def _depth_to_soft_onehot(self, depth):
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

    @force_fp32()
    def _build_depth_gt(self, points, lidar2image, img_aug_matrix, lidar_aug_matrix):
        B = len(points)
        N = lidar2image.shape[1]
        fH, fW = self.feature_size
        device = points[0].device
        depth_map = torch.full((B, N, fH, fW), float("inf"), device=device)
        ds_h, ds_w = self.downsample_factor

        for b in range(B):
            if points[b].numel() == 0:
                continue
            cur_coords = points[b][:, :3]
            cur_coords = cur_coords - lidar_aug_matrix[b][:3, 3]
            cur_coords = torch.inverse(lidar_aug_matrix[b][:3, :3]).matmul(
                cur_coords.transpose(1, 0)
            )
            for c in range(N):
                coords = lidar2image[b, c, :3, :3].matmul(cur_coords)
                coords += lidar2image[b, c, :3, 3].reshape(3, 1)
                dist = coords[2, :]
                mask = dist > self.depth_min_dist
                coords[2, :] = torch.clamp(coords[2, :], 1e-5, 1e5)
                coords[:2, :] /= coords[2:3, :]

                coords = img_aug_matrix[b, c, :3, :3].matmul(coords)
                coords += img_aug_matrix[b, c, :3, 3].reshape(3, 1)

                xs = coords[0, :]
                ys = coords[1, :]
                mask = (
                    mask
                    & (xs >= 0)
                    & (xs < self.image_size[1])
                    & (ys >= 0)
                    & (ys < self.image_size[0])
                )
                if mask.sum() == 0:
                    continue
                xs = xs[mask]
                ys = ys[mask]
                d = dist[mask]

                fx = torch.floor(xs / ds_w).long()
                fy = torch.floor(ys / ds_h).long()
                valid = (fx >= 0) & (fx < fW) & (fy >= 0) & (fy < fH)
                if valid.sum() == 0:
                    continue
                fx = fx[valid]
                fy = fy[valid]
                d = d[valid]

                idx = fy * fW + fx
                flat = depth_map[b, c].view(-1)
                if hasattr(torch.Tensor, "scatter_reduce_"):
                    flat.scatter_reduce_(0, idx, d, reduce="amin", include_self=True)
                else:
                    for i in range(idx.numel()):
                        j = idx[i].item()
                        if d[i] < flat[j]:
                            flat[j] = d[i]

        depth_map[depth_map == float("inf")] = 0.0
        if self.depth_gt_type == "soft_onehot":
            return self._depth_to_soft_onehot(depth_map)
        return depth_map

    @force_fp32()
    def forward(
        self,
        img,
        points,
        radar,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        **kwargs,
    ):
        depth_gt_in = kwargs.get("depth_gt", None)
        intrins = camera_intrinsics[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]
        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]

        geom = self.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans,
        )
        mats_dict = {
            "intrin_mats": camera_intrinsics,
            "ida_mats": img_aug_matrix,
            "bda_mat": lidar_aug_matrix,
            "sensor2ego_mats": camera2lidar,
        }

        if self.debug_timing and img.is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        x, depth_prob = self.get_cam_feats(img, mats_dict)
        if self.debug_timing and img.is_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        bev = self.bev_pool(geom, x)
        if self.debug_timing and img.is_cuda:
            torch.cuda.synchronize()
        t2 = time.perf_counter()

        bev = self.downsample(bev)

        if self.depth_supervision and (self.training or self.depth_gt_inference):
            if depth_gt_in is None:
                if self.debug_timing and img.is_cuda:
                    torch.cuda.synchronize()
                t3 = time.perf_counter()
                depth_gt = self._build_depth_gt(
                    points, lidar2image, img_aug_matrix, lidar_aug_matrix
                )
                if self.debug_timing and img.is_cuda:
                    torch.cuda.synchronize()
                t4 = time.perf_counter()
            else:
                depth_gt = depth_gt_in
                if depth_gt.dim() == 4:
                    depth_gt = self._depth_to_soft_onehot(depth_gt)
                t3 = time.perf_counter()
                t4 = t3
            if self.debug_timing:
                self._timing_count += 1
                self._timing_acc["cam"] += (t1 - t0)
                self._timing_acc["bev"] += (t2 - t1)
                self._timing_acc["gt"] += (t4 - t3)
                self._timing_acc["total"] += (t4 - t0)
                if self._timing_count % self.debug_timing_interval == 0:
                    n = float(self._timing_count)
                    print(
                        f"[BEVDepthTiming] cam={self._timing_acc['cam']/n:.4f}s "
                        f"bev={self._timing_acc['bev']/n:.4f}s "
                        f"gt={self._timing_acc['gt']/n:.4f}s "
                        f"total={self._timing_acc['total']/n:.4f}s",
                        flush=True,
                    )
            return bev, depth_prob, depth_gt
        if self.debug_timing:
            self._timing_count += 1
            self._timing_acc["cam"] += (t1 - t0)
            self._timing_acc["bev"] += (t2 - t1)
            self._timing_acc["total"] += (t2 - t0)
            if self._timing_count % self.debug_timing_interval == 0:
                n = float(self._timing_count)
                print(
                    f"[BEVDepthTiming] cam={self._timing_acc['cam']/n:.4f}s "
                    f"bev={self._timing_acc['bev']/n:.4f}s "
                    f"gt=0.0000s total={self._timing_acc['total']/n:.4f}s",
                    flush=True,
                )
        return bev
