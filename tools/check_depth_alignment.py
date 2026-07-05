#!/usr/bin/env python
"""
Sanity-check script to verify depth GT alignment with training projection.

This script:
1. Loads a sample from the dataset
2. Projects LiDAR points using the SAME matrices used in training
3. Loads the saved depth_gt map
4. Checks overlap between projected pixels and depth_gt nonzero pixels
"""
import os
import os.path as osp
import sys
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

import numpy as np
import cv2
import yaml
import torch
from utils.util_calib import get_matrices_from_dict_calib


def load_lidar_points(pcd_path, skip_lines=13, n_attr=9):
    """Load LiDAR points from PCD file"""
    with open(pcd_path, 'r') as f:
        lines = [line.rstrip('\n') for line in f][skip_lines:]
        pc_lidar = [point.split() for point in lines]
    pc_lidar = np.array(pc_lidar, dtype=float).reshape(-1, n_attr)
    # Remove points at origin
    mask = (np.abs(pc_lidar[:, 0]) > 0.01) | (np.abs(pc_lidar[:, 1]) > 0.01)
    return pc_lidar[mask]


def project_with_training_matrices(points, intrinsics, T_ldr2cam, calib_offset, img_size=(640, 1280)):
    """
    Project LiDAR points using the SAME approach as training.

    Args:
        points: [N, 3] LiDAR points (x, y, z)
        intrinsics: [3, 3] camera intrinsics (already adjusted for crop)
        T_ldr2cam: [3, 4] LiDAR to camera transform (already corrected)
        calib_offset: [3] LiDAR calibration offset
        img_size: (H, W) cropped image size

    Returns:
        u, v, depth: projected pixel coordinates and depth values
    """
    H, W = img_size

    # Apply LiDAR calibration offset
    pts = points[:, :3].copy()
    pts = pts + calib_offset

    # Transform to camera coordinates
    T_4x4 = np.eye(4)
    T_4x4[:3, :] = T_ldr2cam
    pts_hom = np.hstack([pts, np.ones((len(pts), 1))])
    pts_cam = (T_4x4 @ pts_hom.T).T

    # Filter points behind camera
    valid = pts_cam[:, 2] > 1.0
    pts_cam = pts_cam[valid]

    if len(pts_cam) == 0:
        return np.array([]), np.array([]), np.array([])

    # Project to image plane
    K_4x4 = np.eye(4)
    K_4x4[:3, :3] = intrinsics
    pts_img = (K_4x4 @ pts_cam.T).T
    u = pts_img[:, 0] / pts_img[:, 2]
    v = pts_img[:, 1] / pts_img[:, 2]
    depth = pts_cam[:, 2]

    # Filter points outside image
    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return u[valid], v[valid], depth[valid]


def project_with_depthgt_method(points, intrinsics_orig, T_ldr2cam_orig, calib_offset, z_offset,
                                 y_crop, img_size_orig=(720, 1280)):
    """
    Project LiDAR points using the SAME approach as depth GT generation.

    This uses original intrinsics and subtracts y_crop from v coordinates.
    """
    H_orig, W = img_size_orig
    H_crop = H_orig - y_crop

    # Apply LiDAR calibration offset (same as depth GT generation)
    pts = points[:, :3].copy()
    pts[:, 0] += calib_offset[0]
    pts[:, 1] += calib_offset[1]
    pts[:, 2] += z_offset

    # Apply extrinsic calibration correction (same as depth GT generation)
    rotation = T_ldr2cam_orig[:, :3]
    calib_vals = -1 * np.array([-2.54, 0.3, 0.7]).reshape(3,)
    T_corrected = T_ldr2cam_orig.copy()
    T_corrected[:, 3] = T_ldr2cam_orig[:, 3] + rotation @ calib_vals

    # Transform to camera coordinates
    T_4x4 = np.eye(4)
    T_4x4[:3, :] = T_corrected
    pts_hom = np.hstack([pts, np.ones((len(pts), 1))])
    pts_cam = (T_4x4 @ pts_hom.T).T

    # Filter points behind camera
    valid = pts_cam[:, 2] > 1.0
    pts_cam = pts_cam[valid]

    if len(pts_cam) == 0:
        return np.array([]), np.array([]), np.array([])

    # Project to image plane with original intrinsics
    K_4x4 = np.eye(4)
    K_4x4[:3, :3] = intrinsics_orig
    pts_img = (K_4x4 @ pts_cam.T).T
    u = pts_img[:, 0] / pts_img[:, 2]
    v = pts_img[:, 1] / pts_img[:, 2]
    depth = pts_cam[:, 2]

    # Filter points outside original image first
    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H_orig)
    u = u[valid]
    v = v[valid]
    depth = depth[valid]

    # Apply crop (subtract y_crop from v)
    v_cropped = v - y_crop
    valid = v_cropped >= 0
    return u[valid], v_cropped[valid], depth[valid]


def main():
    print("=" * 60)
    print("Depth GT Alignment Check")
    print("=" * 60)

    # Load dataset to get calibration matrices
    from pipelines.pipeline_detection_v1_0 import PipelineDetection_v1_0
    pline = PipelineDetection_v1_0(
        path_cfg='configs/CLEAR/BEVDepth.yml',
        mode='train',
        rank=-1,
        tag='check_align'
    )

    dataset = pline.dataset_train
    sample = dataset[0]

    meta = sample['meta']
    seq = str(int(meta['seq']))
    print(f"\nSample: seq={seq}")

    # Get depth GT
    depth_gt = sample['depth_gt']
    if depth_gt.ndim == 3:
        depth_gt = depth_gt[0]  # [H, W]
    print(f"Depth GT shape: {depth_gt.shape}")
    print(f"Depth GT non-zero pixels: {(depth_gt > 0).sum()}")

    # Get training calibration matrices
    seq_calib = dataset.dict_cam_calib.get(seq)
    img_size_train, intrinsics_train, distortion_train, T_ldr2cam_train = seq_calib

    print(f"\n=== Training Calibration ===")
    print(f"Intrinsics (cy adjusted for crop):\n{intrinsics_train}")
    print(f"T_ldr2cam (already corrected):\n{T_ldr2cam_train}")

    # Load original calibration (before crop adjustment)
    calib_dir = './resources/cam_calib/calib_seq_v2'
    seq_dir = f'seq_{int(seq):02d}'
    yml_path = osp.join(calib_dir, seq_dir, 'cam_1.yml')
    with open(yml_path, 'r') as f:
        dict_calib = yaml.safe_load(f)
    img_size_orig, intrinsics_orig, distortion_orig, T_ldr2cam_orig = get_matrices_from_dict_calib(dict_calib)

    # Apply undistortion to original intrinsics (as in depth GT generation)
    ncm, _ = cv2.getOptimalNewCameraMatrix(intrinsics_orig, distortion_orig, img_size_orig, alpha=0.0)
    intrinsics_orig_undist = ncm.copy()

    print(f"\n=== Depth GT Generation Calibration ===")
    print(f"Intrinsics (original, undistorted):\n{intrinsics_orig_undist}")
    print(f"T_ldr2cam (original, before correction):\n{T_ldr2cam_orig}")

    # Get LiDAR calibration offset
    calib_offset = np.array(meta['calib'])
    print(f"\nLiDAR calibration offset: {calib_offset}")

    # Load LiDAR points
    seq_path = osp.join('./data/kradar', seq)
    # Find the first frame's LiDAR file
    pcd_files = sorted([f for f in os.listdir(osp.join(seq_path, 'os2-64')) if f.endswith('.pcd')])
    pcd_path = osp.join(seq_path, 'os2-64', pcd_files[0])
    print(f"Loading LiDAR from: {pcd_path}")

    lidar_points = load_lidar_points(pcd_path)
    print(f"LiDAR points: {len(lidar_points)}")

    # Project using training method
    y_crop = 80  # 720 - 640
    u_train, v_train, depth_train = project_with_training_matrices(
        lidar_points, intrinsics_train, T_ldr2cam_train, calib_offset,
        img_size=(640, 1280)
    )
    print(f"\n=== Training Projection ===")
    print(f"Points in image: {len(u_train)}")

    # Project using depth GT method
    u_gt, v_gt, depth_gt_proj = project_with_depthgt_method(
        lidar_points, intrinsics_orig_undist, T_ldr2cam_orig,
        (calib_offset[0], calib_offset[1]), calib_offset[2],
        y_crop, img_size_orig=(720, 1280)
    )
    print(f"\n=== Depth GT Generation Projection ===")
    print(f"Points in image: {len(u_gt)}")

    # Compare projections
    print(f"\n=== Projection Comparison ===")
    if len(u_train) > 0 and len(u_gt) > 0:
        # They might have different number of points due to filtering order
        # Compare the depth values at similar locations
        n_compare = min(len(u_train), len(u_gt), 1000)

        print(f"Training u range: [{u_train.min():.1f}, {u_train.max():.1f}]")
        print(f"Depth GT u range: [{u_gt.min():.1f}, {u_gt.max():.1f}]")
        print(f"Training v range: [{v_train.min():.1f}, {v_train.max():.1f}]")
        print(f"Depth GT v range: [{v_gt.min():.1f}, {v_gt.max():.1f}]")

    # Check alignment with saved depth GT
    print(f"\n=== Alignment with Saved Depth GT ===")
    H, W = depth_gt.shape

    # Using training projection
    u_int = np.clip(u_train.astype(np.int32), 0, W-1)
    v_int = np.clip(v_train.astype(np.int32), 0, H-1)
    gt_at_proj = depth_gt[v_int, u_int]
    has_gt = gt_at_proj > 0

    print(f"Training projection points with depth GT match: {has_gt.sum()} / {len(u_train)} ({100*has_gt.mean():.1f}%)")

    if has_gt.sum() > 0:
        depth_diff = np.abs(depth_train[has_gt] - gt_at_proj[has_gt])
        print(f"Depth difference at matches - mean: {depth_diff.mean():.3f}m, max: {depth_diff.max():.3f}m")

        # Show some samples
        print(f"\nSample depth comparisons:")
        for i in range(min(5, has_gt.sum())):
            idx = np.where(has_gt)[0][i]
            print(f"  Projected: {depth_train[idx]:.2f}m, GT: {gt_at_proj[idx]:.2f}m, diff: {abs(depth_train[idx] - gt_at_proj[idx]):.3f}m")

    # Also check using depth GT projection method
    print(f"\n=== Using Depth GT Projection Method ===")
    u_int_gt = np.clip(u_gt.astype(np.int32), 0, W-1)
    v_int_gt = np.clip(v_gt.astype(np.int32), 0, H-1)
    gt_at_proj_gt = depth_gt[v_int_gt, u_int_gt]
    has_gt_2 = gt_at_proj_gt > 0

    print(f"Depth GT projection points with depth GT match: {has_gt_2.sum()} / {len(u_gt)} ({100*has_gt_2.mean():.1f}%)")

    if has_gt_2.sum() > 0:
        depth_diff_2 = np.abs(depth_gt_proj[has_gt_2] - gt_at_proj_gt[has_gt_2])
        print(f"Depth difference at matches - mean: {depth_diff_2.mean():.3f}m, max: {depth_diff_2.max():.3f}m")

    print("\n" + "=" * 60)
    if has_gt.sum() > 0 and depth_diff.mean() < 1.0:
        print("ALIGNMENT CHECK PASSED: Training projection aligns with depth GT")
    else:
        print("ALIGNMENT CHECK FAILED: Significant mismatch between projections")
    print("=" * 60)


if __name__ == '__main__':
    main()
