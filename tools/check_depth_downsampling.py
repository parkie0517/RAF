#!/usr/bin/env python
"""
Verify depth GT downsampling and bin conversion matches depth predictions.

This script:
1. Loads a sample from the dataset
2. Shows depth GT value distribution
3. Manually applies the downsample and bin conversion
4. Does a forward pass to get depth predictions
5. Compares the GT bins with prediction bins
"""
import os
import os.path as osp
import sys
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F


def main():
    print("=" * 60)
    print("Depth GT Downsampling and Bin Conversion Check")
    print("=" * 60)

    # Load dataset
    from pipelines.pipeline_detection_v1_0 import PipelineDetection_v1_0
    pline = PipelineDetection_v1_0(
        path_cfg='configs/CLEAR/BEVDepth.yml',
        mode='train',
        rank=-1,
        tag='check_ds'
    )

    dataset = pline.dataset_train
    sample = dataset[0]

    # Get depth GT
    depth_gt = sample['depth_gt']  # This should be [1, H, W] or [H, W]
    print(f"\n=== Depth GT from Dataset ===")
    print(f"Shape: {depth_gt.shape}")
    print(f"dtype: {depth_gt.dtype}")

    if depth_gt.ndim == 3:
        depth_gt = depth_gt[0]  # [H, W]

    H, W = depth_gt.shape
    print(f"Full resolution: {H} x {W}")

    # Depth value statistics
    nonzero_mask = depth_gt > 0
    depth_values = depth_gt[nonzero_mask]
    print(f"Non-zero pixels: {nonzero_mask.sum()}")
    print(f"Depth range: [{depth_values.min():.2f}m, {depth_values.max():.2f}m]")
    print(f"Depth mean: {depth_values.mean():.2f}m, std: {depth_values.std():.2f}m")

    # Get depth config
    dbound = [1.0, 72.0, 0.5]  # From config
    downsample_factor = 8
    depth_channels = int((dbound[1] - dbound[0]) / dbound[2])  # 142 bins

    print(f"\n=== Depth Config ===")
    print(f"dbound: {dbound}")
    print(f"downsample_factor: {downsample_factor}")
    print(f"depth_channels: {depth_channels}")
    print(f"Downsampled resolution: {H // downsample_factor} x {W // downsample_factor}")

    # Manual downsample (matching get_downsampled_gt_depth)
    depth_gt_tensor = torch.from_numpy(depth_gt).float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    B, N = 1, 1

    gt_depths = depth_gt_tensor.view(
        B * N,
        H // downsample_factor,
        downsample_factor,
        W // downsample_factor,
        downsample_factor,
        1,
    )
    gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
    gt_depths = gt_depths.view(-1, downsample_factor * downsample_factor)

    # Replace zeros with 1e5 and take min
    gt_depths_tmp = torch.where(gt_depths == 0.0,
                                1e5 * torch.ones_like(gt_depths),
                                gt_depths)
    gt_depths_min = torch.min(gt_depths_tmp, dim=-1).values
    gt_depths_downsampled = gt_depths_min.view(B * N, H // downsample_factor, W // downsample_factor)

    print(f"\n=== After Downsampling ===")
    print(f"Downsampled shape: {gt_depths_downsampled.shape}")
    ds_nonzero = gt_depths_downsampled[gt_depths_downsampled < 1e4]  # Exclude the 1e5 placeholders
    print(f"Non-zero pixels after downsample: {len(ds_nonzero)}")
    print(f"Depth range after downsample: [{ds_nonzero.min():.2f}m, {ds_nonzero.max():.2f}m]")

    # Convert to bin indices
    gt_depths_bins = (gt_depths_downsampled - (dbound[0] - dbound[2])) / dbound[2]

    # Valid bins are in [0, depth_channels)
    valid_mask = (gt_depths_bins >= 0.0) & (gt_depths_bins < depth_channels + 1)
    gt_depths_bins = torch.where(valid_mask, gt_depths_bins, torch.zeros_like(gt_depths_bins))

    print(f"\n=== Bin Indices ===")
    valid_bins = gt_depths_bins[gt_depths_bins > 0]
    print(f"Valid bin count: {len(valid_bins)}")
    print(f"Bin range: [{valid_bins.min():.1f}, {valid_bins.max():.1f}]")
    print(f"Mean bin: {valid_bins.mean():.1f}")

    # One-hot encoding
    gt_depths_onehot = F.one_hot(gt_depths_bins.long(), num_classes=depth_channels + 1)
    gt_depths_onehot = gt_depths_onehot.view(-1, depth_channels + 1)[:, 1:]  # Remove bin 0 (invalid)

    print(f"\n=== One-Hot Encoding ===")
    print(f"One-hot shape: {gt_depths_onehot.shape}")
    print(f"Expected: [num_pixels={80*160}, depth_channels={depth_channels}]")

    # Count pixels with valid depth (any 1 in one-hot)
    has_valid = gt_depths_onehot.sum(dim=1) > 0
    print(f"Pixels with valid depth GT: {has_valid.sum().item()} / {len(has_valid)}")

    # Now let's run the actual model forward pass
    print(f"\n=== Model Forward Pass ===")

    # Create a batch
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                           collate_fn=dataset.collate_fn)
    batch = next(iter(dataloader))

    # Model is already on GPU from pipeline initialization
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = pline.network  # Already on GPU
    model.eval()

    # Move batch data to device
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)

    # Force training mode temporarily to get depth predictions
    model.training = True
    with torch.no_grad():
        # Forward pass (depth_pred is stored in batch_dict when training)
        batch = model(batch)

    depth_pred = batch.get('depth_pred')
    if depth_pred is not None:
        print(f"depth_pred shape: {depth_pred.shape}")  # Should be [B*N, D, H/8, W/8]
        print(f"depth_pred range: [{depth_pred.min():.4f}, {depth_pred.max():.4f}]")

        # Apply softmax to get probabilities
        depth_probs = F.softmax(depth_pred, dim=1)
        print(f"depth_probs range after softmax: [{depth_probs.min():.4f}, {depth_probs.max():.4f}]")

        # Check probability distribution
        print(f"\n=== Depth Prediction Analysis ===")

        # Get max probability for each pixel
        max_probs, pred_bins = depth_probs.max(dim=1)
        print(f"Max probability range: [{max_probs.min():.4f}, {max_probs.max():.4f}]")
        print(f"Predicted bin range: [{pred_bins.min()}, {pred_bins.max()}]")

        # Compare with GT bins
        gt_bins_flat = gt_depths_bins.long().view(-1).to(device) - 1  # Subtract 1 since one-hot excludes bin 0
        gt_bins_flat = torch.clamp(gt_bins_flat, min=0, max=depth_channels-1)
        pred_bins_flat = pred_bins.view(-1)

        # Only compare where we have valid GT
        valid_gt_mask = has_valid.to(device)
        gt_valid = gt_bins_flat[valid_gt_mask]
        pred_valid = pred_bins_flat[valid_gt_mask]

        # Check accuracy
        correct = (gt_valid == pred_valid).float()
        print(f"\n=== GT vs Prediction Comparison (valid pixels only) ===")
        print(f"Number of valid GT pixels: {len(gt_valid)}")
        print(f"Bin match accuracy: {correct.mean():.4f} ({correct.sum().long()}/{len(gt_valid)})")

        # Check mean depth probabilities at GT bin
        depth_probs_flat = depth_probs.view(depth_probs.shape[0], depth_probs.shape[1], -1)  # [B*N, D, h*w]
        depth_probs_flat = depth_probs_flat.squeeze(0).permute(1, 0)  # [h*w, D]
        gt_bin_probs = depth_probs_flat[valid_gt_mask, gt_valid]
        print(f"Mean probability at GT bin: {gt_bin_probs.mean():.6f}")
        print(f"Expected random baseline (1/{depth_channels}): {1.0/depth_channels:.6f}")

        # Show some examples
        print(f"\n=== Sample Comparisons ===")
        sample_indices = torch.where(valid_gt_mask)[0][:10]
        for i in sample_indices:
            gt_bin = gt_bins_flat[i].item()
            pred_bin = pred_bins_flat[i].item()
            gt_prob = depth_probs_flat[i, gt_bin].item()
            pred_prob = depth_probs_flat[i, pred_bin].item()
            print(f"  Pixel {i.item()}: GT bin={gt_bin}, pred bin={pred_bin}, "
                  f"P(GT)={gt_prob:.4f}, P(pred)={pred_prob:.4f}")
    else:
        print("ERROR: depth_preds not found in batch!")
        print(f"Available keys: {list(batch.keys())}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
