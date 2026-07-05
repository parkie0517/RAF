#!/usr/bin/env python
"""
Quick test to verify depth loss is being computed after the fix.
"""
import os
import os.path as osp
import sys
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

import torch


def main():
    print("=" * 60)
    print("Verify Depth Loss Computation")
    print("=" * 60)

    # Load dataset
    from pipelines.pipeline_detection_v1_0 import PipelineDetection_v1_0
    pline = PipelineDetection_v1_0(
        path_cfg='configs/CLEAR/BEVDepth.yml',
        mode='train',
        rank=-1,
        tag='verify_loss'
    )

    dataset = pline.dataset_train
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False,
                           collate_fn=dataset.collate_fn)
    batch = next(iter(dataloader))

    print(f"\n=== Batch Contents ===")
    print(f"depth_gt in batch: {'depth_gt' in batch}")
    if 'depth_gt' in batch:
        print(f"depth_gt shape: {batch['depth_gt'].shape}")

    # Model is already on GPU from pipeline initialization
    model = pline.network
    model.train()

    # Forward pass
    print(f"\n=== Forward Pass ===")
    batch = model(batch)

    print(f"depth_pred in batch after forward: {'depth_pred' in batch}")
    print(f"lidar_depth in batch after forward: {'lidar_depth' in batch}")

    if 'depth_pred' in batch:
        print(f"depth_pred shape: {batch['depth_pred'].shape}")
    if 'lidar_depth' in batch:
        print(f"lidar_depth shape: {batch['lidar_depth'].shape}")

    # Compute loss
    print(f"\n=== Loss Computation ===")
    loss = model.loss(batch)
    print(f"Total loss: {loss.item():.4f}")

    if 'logging' in batch:
        print(f"Logged values: {batch['logging']}")
        if 'depth_loss' in batch['logging']:
            print(f"depth_loss: {batch['logging']['depth_loss']:.4f}")
        else:
            print("WARNING: depth_loss not in logging dict!")
    else:
        print("WARNING: No logging dict in batch!")

    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
