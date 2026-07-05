"""
Depth GT Generation Script for K-Radar Dataset

This script generates depth ground truth from LiDAR point clouds for BEVDepth training.
Based on BEVDepth's depth generation method, adapted for K-Radar dataset.

Usage:
    python tools/generate_depth_gt.py --data_root ./data/kradar --output_dir ./data/depth_gt --split train
"""

import os
import os.path as osp
import argparse
import numpy as np
import cv2
import yaml
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add project root to path
import sys
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from utils.util_calib import get_matrices_from_dict_calib


def parse_args():
    parser = argparse.ArgumentParser(description='Generate Depth GT for K-Radar')
    parser.add_argument('--data_root', type=str, default='./data/kradar',
                        help='Root directory of K-Radar dataset')
    parser.add_argument('--output_dir', type=str, default='./data/depth_gt',
                        help='Output directory for depth GT')
    parser.add_argument('--split_file', type=str, default='./resources/split/train.txt',
                        help='Split file for train/test')
    parser.add_argument('--calib_dir', type=str, default='./resources/cam_calib/calib_seq_v2',
                        help='Camera calibration directory')
    parser.add_argument('--ori_size', type=int, nargs=2, default=[720, 1280],
                        help='Original image size [H, W]')
    parser.add_argument('--crop_size', type=int, nargs=2, default=[640, 1280],
                        help='Cropped image size [H, W]')
    parser.add_argument('--min_depth', type=float, default=1.0,
                        help='Minimum depth value')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of parallel workers')
    parser.add_argument('--z_offset', type=float, default=0.7,
                        help='Z offset for LiDAR calibration')
    return parser.parse_args()


def load_split_file(split_file):
    """Load split file and return dict of {seq: [frame_ids]}"""
    split_dict = {}
    with open(split_file, 'r') as f:
        for line in f:
            seq, frame_id = line.strip().split(',')
            seq = str(int(seq))  # Remove leading zeros
            if seq not in split_dict:
                split_dict[seq] = []
            split_dict[seq].append(frame_id)
    return split_dict


def load_camera_calibration(calib_dir, seq):
    """Load camera calibration for a sequence"""
    seq_dir = f'seq_{int(seq):02d}'
    yml_path = osp.join(calib_dir, seq_dir, 'cam_1.yml')

    if not osp.exists(yml_path):
        # Try alternative naming
        for d in os.listdir(calib_dir):
            if d.startswith('seq_') and str(int(d.split('_')[-1])) == seq:
                yml_path = osp.join(calib_dir, d, 'cam_1.yml')
                break

    if not osp.exists(yml_path):
        return None

    with open(yml_path, 'r') as f:
        dict_calib = yaml.safe_load(f)

    return get_matrices_from_dict_calib(dict_calib)


def load_lidar_calibration(calib_path):
    """Load LiDAR calibration values (dx, dy, dz)"""
    with open(calib_path, 'r') as f:
        lines = f.readlines()
    calib_values = list(map(float, lines[1].split(',')))
    return calib_values[1], calib_values[2]  # X, Y offsets


def load_lidar_points(pcd_path, skip_lines=13, n_attr=9):
    """Load LiDAR points from PCD file"""
    with open(pcd_path, 'r') as f:
        lines = [line.rstrip('\n') for line in f][skip_lines:]
        pc_lidar = [point.split() for point in lines]
    pc_lidar = np.array(pc_lidar, dtype=float).reshape(-1, n_attr)

    # Remove points at origin
    mask = (np.abs(pc_lidar[:, 0]) > 0.01) | (np.abs(pc_lidar[:, 1]) > 0.01)
    pc_lidar = pc_lidar[mask]

    return pc_lidar


def project_lidar_to_image(lidar_points, intrinsics, T_ldr2cam, img_size,
                           calib_offset, z_offset, y_crop, min_depth=1.0):
    """
    Project LiDAR points to image plane and generate depth map.

    Args:
        lidar_points: [N, 9] LiDAR points
        intrinsics: [3, 3] camera intrinsic matrix
        T_ldr2cam: [3, 4] LiDAR to camera transform
        img_size: (W, H) original image size
        calib_offset: (dx, dy) LiDAR calibration offset
        z_offset: z offset for LiDAR calibration
        y_crop: number of pixels to crop from top
        min_depth: minimum depth threshold

    Returns:
        depth_map: [H_crop, W] depth map
    """
    # Apply LiDAR calibration
    points = lidar_points[:, :3].copy()
    points[:, 0] += calib_offset[0]
    points[:, 1] += calib_offset[1]
    points[:, 2] += z_offset

    # Apply extrinsic calibration correction (same as in dataset)
    rotation = T_ldr2cam[:, :3]
    calib_vals = -1 * np.array([-2.54, 0.3, 0.7]).reshape(3,)
    T_ldr2cam_corrected = T_ldr2cam.copy()
    T_ldr2cam_corrected[:, 3] = T_ldr2cam[:, 3] + rotation @ calib_vals

    # Transform to camera coordinates
    T_ldr2cam_4x4 = np.eye(4)
    T_ldr2cam_4x4[:3, :] = T_ldr2cam_corrected

    points_hom = np.hstack([points, np.ones((points.shape[0], 1))])  # [N, 4]
    points_cam = (T_ldr2cam_4x4 @ points_hom.T).T  # [N, 4]

    # Filter points behind camera
    valid_mask = points_cam[:, 2] > min_depth
    points_cam = points_cam[valid_mask]

    if len(points_cam) == 0:
        return np.zeros((img_size[1] - y_crop, img_size[0]), dtype=np.float32)

    # Project to image plane
    intrinsics_4x4 = np.eye(4)
    intrinsics_4x4[:3, :3] = intrinsics

    points_img = (intrinsics_4x4 @ points_cam.T).T  # [N, 4]
    points_img[:, 0] /= points_img[:, 2]
    points_img[:, 1] /= points_img[:, 2]

    # Get pixel coordinates and depth
    u = points_img[:, 0]
    v = points_img[:, 1]
    depth = points_cam[:, 2]

    # Filter points outside original image
    W, H = img_size
    valid_mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u = u[valid_mask]
    v = v[valid_mask]
    depth = depth[valid_mask]

    # Apply crop (remove top y_crop pixels)
    v_cropped = v - y_crop
    valid_mask = v_cropped >= 0
    u = u[valid_mask]
    v_cropped = v_cropped[valid_mask]
    depth = depth[valid_mask]

    # Create depth map
    H_crop = H - y_crop
    depth_map = np.zeros((H_crop, W), dtype=np.float32)

    if len(u) > 0:
        # Convert to integer coordinates
        u_int = u.astype(np.int32)
        v_int = v_cropped.astype(np.int32)

        # Clip to valid range
        u_int = np.clip(u_int, 0, W - 1)
        v_int = np.clip(v_int, 0, H_crop - 1)

        # For duplicate pixels, keep the closest depth
        # Sort by depth (ascending) so closer points overwrite farther ones
        sort_idx = np.argsort(-depth)  # descending, so farther points first
        u_int = u_int[sort_idx]
        v_int = v_int[sort_idx]
        depth = depth[sort_idx]

        depth_map[v_int, u_int] = depth

    return depth_map


def process_frame(args_tuple):
    """Process a single frame to generate depth GT"""
    (seq, frame_id, data_root, output_dir, calib_dir,
     ori_size, crop_size, z_offset, min_depth) = args_tuple

    # Paths
    seq_path = osp.join(data_root, seq)
    label_path = osp.join(seq_path, 'info_label', frame_id)

    # Parse frame info from label file
    with open(label_path, 'r') as f:
        header = f.readline().strip()

    try:
        temp_idx, _ = header.split(', ')
    except:
        _, header_prime, _ = header.split('*')
        header = '*' + header_prime
        temp_idx, _ = header.split(', ')

    rdr, ldr64, camf, ldr128, camr = temp_idx.split('=')[1].split('_')

    # Load camera calibration
    cam_calib = load_camera_calibration(calib_dir, seq)
    if cam_calib is None:
        return None, f"No calibration for seq {seq}"

    img_size_tuple, intrinsics, distortion, T_ldr2cam = cam_calib
    img_size = (img_size_tuple[0], img_size_tuple[1]) if isinstance(img_size_tuple, tuple) else img_size_tuple

    # Apply undistortion to intrinsics
    ncm, _ = cv2.getOptimalNewCameraMatrix(intrinsics, distortion, img_size, alpha=0.0)
    intrinsics = ncm.copy()

    # Load LiDAR calibration
    lidar_calib_path = osp.join(seq_path, 'info_calib', 'calib_radar_lidar.txt')
    if not osp.exists(lidar_calib_path):
        return None, f"No LiDAR calibration for seq {seq}"
    dx, dy = load_lidar_calibration(lidar_calib_path)

    # Load LiDAR points
    pcd_path = osp.join(seq_path, 'os2-64', f'os2-64_{ldr64}.pcd')
    if not osp.exists(pcd_path):
        return None, f"PCD file not found: {pcd_path}"

    lidar_points = load_lidar_points(pcd_path)

    # Calculate crop offset
    y_crop = ori_size[0] - crop_size[0]  # 720 - 640 = 80

    # Generate depth map
    depth_map = project_lidar_to_image(
        lidar_points, intrinsics, T_ldr2cam,
        (ori_size[1], ori_size[0]),  # (W, H)
        (dx, dy), z_offset, y_crop, min_depth
    )

    # Save depth map
    output_seq_dir = osp.join(output_dir, seq)
    os.makedirs(output_seq_dir, exist_ok=True)
    output_path = osp.join(output_seq_dir, f'cam-front_{camf}.npy')
    np.save(output_path, depth_map)

    return output_path, None


def main():
    args = parse_args()

    print("=" * 60)
    print("Depth GT Generation for K-Radar")
    print("=" * 60)
    print(f"Data root: {args.data_root}")
    print(f"Output dir: {args.output_dir}")
    print(f"Split file: {args.split_file}")
    print(f"Original size: {args.ori_size}")
    print(f"Crop size: {args.crop_size}")
    print(f"Min depth: {args.min_depth}")
    print("=" * 60)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load split file
    split_dict = load_split_file(args.split_file)
    print(f"Loaded {sum(len(v) for v in split_dict.values())} frames from {len(split_dict)} sequences")

    # Prepare task list
    tasks = []
    for seq, frame_ids in split_dict.items():
        for frame_id in frame_ids:
            tasks.append((
                seq, frame_id, args.data_root, args.output_dir, args.calib_dir,
                args.ori_size, args.crop_size, args.z_offset, args.min_depth
            ))

    print(f"Processing {len(tasks)} frames...")

    # Process frames
    success_count = 0
    error_count = 0

    if args.num_workers > 1:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(process_frame, task): task for task in tasks}
            for future in tqdm(as_completed(futures), total=len(futures)):
                result, error = future.result()
                if error:
                    error_count += 1
                else:
                    success_count += 1
    else:
        for task in tqdm(tasks):
            result, error = process_frame(task)
            if error:
                error_count += 1
            else:
                success_count += 1

    print("=" * 60)
    print(f"Done! Success: {success_count}, Errors: {error_count}")
    print(f"Depth GT saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
