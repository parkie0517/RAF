import argparse
import os
import os.path as osp
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.util_config import cfg, cfg_from_yaml_file
from datasets.kradar_detection_v2_3 import KRadarDetection_v2_3


def build_depth_map(points, lidar2image, image_size, feature_size, min_dist):
    h, w = image_size
    fH, fW = feature_size
    ds_h = h // fH
    ds_w = w // fW

    depth_map = np.full((fH, fW), np.inf, dtype=np.float32)
    if points.shape[0] == 0:
        return np.zeros((fH, fW), dtype=np.float32)

    pts = points[:, :3].T
    pts_h = np.vstack((pts, np.ones((1, pts.shape[1]), dtype=np.float32)))
    proj = lidar2image @ pts_h
    z = proj[2, :]
    x = proj[0, :] / np.clip(z, 1e-5, 1e5)
    y = proj[1, :] / np.clip(z, 1e-5, 1e5)

    mask = (
        (z > min_dist)
        & (x >= 0)
        & (x < w)
        & (y >= 0)
        & (y < h)
    )
    if not np.any(mask):
        return np.zeros((fH, fW), dtype=np.float32)

    x = x[mask]
    y = y[mask]
    z = z[mask]
    fx = np.floor(x / ds_w).astype(np.int64)
    fy = np.floor(y / ds_h).astype(np.int64)
    valid = (fx >= 0) & (fx < fW) & (fy >= 0) & (fy < fH)
    if not np.any(valid):
        return np.zeros((fH, fW), dtype=np.float32)

    fx = fx[valid]
    fy = fy[valid]
    z = z[valid].astype(np.float32)

    idx = fy * fW + fx
    flat = depth_map.reshape(-1)
    np.minimum.at(flat, idx, z)
    depth_map = depth_map.reshape(fH, fW)
    depth_map[~np.isfinite(depth_map)] = 0.0
    return depth_map


_DATASET = None
_IMAGE_SIZE = None
_FEATURE_SIZE = None
_MIN_DIST = None
_OUT_DIR = None


def _init_worker(cfg_file, split, out_dir):
    global _DATASET, _IMAGE_SIZE, _FEATURE_SIZE, _MIN_DIST, _OUT_DIR
    cfg_from_yaml_file(cfg_file, cfg)
    _DATASET = KRadarDetection_v2_3(cfg, split=split)
    _IMAGE_SIZE = cfg.MODEL.VTRANSFORM.image_size
    _FEATURE_SIZE = cfg.MODEL.VTRANSFORM.feature_size
    _MIN_DIST = cfg.MODEL.VTRANSFORM.LOSS.DEPTH_MIN_DIST
    _OUT_DIR = Path(out_dir)
    _OUT_DIR.mkdir(parents=True, exist_ok=True)


def _process_index(idx, overwrite=False):
    global _DATASET, _IMAGE_SIZE, _FEATURE_SIZE, _MIN_DIST, _OUT_DIR
    dict_item = _DATASET.list_dict_item[idx]
    if not _DATASET.load_label_in_advance:
        dict_item = _DATASET.get_label(dict_item)

    points = _DATASET.get_ldr64_from_path(dict_item, is_calib=True)
    dict_item["ldr64"] = points
    if _DATASET.roi.filter:
        x_min, y_min, z_min, x_max, y_max, z_max = _DATASET.roi.xyz
        mask = (
            (points[:, 0] > x_min) & (points[:, 0] < x_max) &
            (points[:, 1] > y_min) & (points[:, 1] < y_max) &
            (points[:, 2] > z_min) & (points[:, 2] < z_max)
        )
        points = points[mask]

    dict_item = _DATASET.get_camera_param(dict_item)
    lidar2image = dict_item["cam_param"]["lidar2image"][0]

    seq = dict_item["meta"]["seq"]
    camf = dict_item["meta"]["idx"]["camf"]
    seq_dir = _OUT_DIR / f"{seq}"
    seq_dir.mkdir(parents=True, exist_ok=True)
    out_path = seq_dir / f"cam-front_{camf}.npy"
    if out_path.exists() and not overwrite:
        return

    depth_map = build_depth_map(
        points, lidar2image, _IMAGE_SIZE, _FEATURE_SIZE, _MIN_DIST
    )
    np.save(out_path, depth_map.astype(np.float32))


def _process_index_wrapper(args):
    idx, overwrite = args
    return _process_index(idx, overwrite=overwrite)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", required=True)
    parser.add_argument("--split", default="all")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    dataset = KRadarDetection_v2_3(cfg, split=args.split)

    out_dir = cfg.DATASET.get("depth_gt", {}).get("dir", "./data/kradar/depth_gt")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.num_workers <= 1:
        _init_worker(args.cfg_file, args.split, out_dir)
        for idx in tqdm(range(len(_DATASET.list_dict_item)), desc="gen_depth_gt"):
            _process_index(idx, overwrite=args.overwrite)
    else:
        with Pool(
            processes=args.num_workers,
            initializer=_init_worker,
            initargs=(args.cfg_file, args.split, out_dir),
        ) as pool:
            for _ in tqdm(
                pool.imap_unordered(
                    _process_index_wrapper,
                    [(i, args.overwrite) for i in range(len(dataset.list_dict_item))],
                ),
                total=len(dataset.list_dict_item),
                desc="gen_depth_gt",
            ):
                pass


if __name__ == "__main__":
    main()
