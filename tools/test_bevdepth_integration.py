#!/usr/bin/env python
"""
Test script for BEVDepth integration
Tests: Dataset v2.4, StudentBEVDepth model, forward pass
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from easydict import EasyDict
from torch.utils.data import DataLoader

def load_config():
    """Load BEVDepth config"""
    import yaml
    config_path = 'configs/CLEAR/BEVDepth.yml'
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return EasyDict(cfg)

def create_synthetic_batch(cfg):
    """Create synthetic batch data for testing model without dataset"""
    print("[INFO] Creating synthetic batch data for model testing...")

    batch_size = 2
    # Image size from config
    H, W = cfg.DATASET.cam_preprocess.scale_size  # [640, 1280]

    batch = {
        'front0': torch.randn(batch_size, 3, H, W),
        'gt_boxes': torch.zeros(batch_size, 50, 8),  # [x, y, z, dx, dy, dz, heading, class]
        'batch_size': batch_size,
        'mats_dict': {
            'sensor2ego_mats': torch.eye(4).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1, 1),  # [B, 1, 1, 4, 4]
            'intrin_mats': torch.eye(4).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1, 1),      # [B, 1, 1, 4, 4]
            'ida_mats': torch.eye(4).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1, 1),         # [B, 1, 1, 4, 4]
            'bda_mat': torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1),  # [B, 4, 4]
        },
        'depth_gt': torch.rand(batch_size, 1, H, W) * 70 + 1,  # Random depth 1-71m
    }

    # Add some GT boxes
    batch['gt_boxes'][0, 0] = torch.tensor([30.0, 0.0, 0.5, 4.2, 2.1, 2.0, 0.0, 1.0])  # Sedan
    batch['gt_boxes'][1, 0] = torch.tensor([40.0, 5.0, 1.0, 9.5, 3.2, 3.7, 0.5, 2.0])  # Bus

    # Set proper intrinsic matrix (approximate K-Radar camera)
    fx, fy = 1000.0, 1000.0
    cx, cy = W / 2, H / 2
    for i in range(batch_size):
        batch['mats_dict']['intrin_mats'][i, 0, 0, 0, 0] = fx
        batch['mats_dict']['intrin_mats'][i, 0, 0, 1, 1] = fy
        batch['mats_dict']['intrin_mats'][i, 0, 0, 0, 2] = cx
        batch['mats_dict']['intrin_mats'][i, 0, 0, 1, 2] = cy

    print(f"[OK] Synthetic batch created")
    print(f"    - front0: {batch['front0'].shape}")
    print(f"    - gt_boxes: {batch['gt_boxes'].shape}")
    print(f"    - depth_gt: {batch['depth_gt'].shape}")

    return batch

def test_dataset():
    """Test Dataset v2.4 loading"""
    print("\n" + "="*60)
    print("TEST 1: Dataset v2.4 Loading")
    print("="*60)

    cfg = load_config()

    try:
        from datasets import __all__ as datasets_all

        # Check dataset exists
        assert 'KRadarDetection_v2_4' in datasets_all, "KRadarDetection_v2_4 not found in datasets"
        print("[OK] KRadarDetection_v2_4 found in datasets")

        # Create dataset
        dataset = datasets_all['KRadarDetection_v2_4'](cfg, split='train')
        print(f"[OK] Dataset created with {len(dataset)} samples")

        # Create dataloader to test collate_fn
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            collate_fn=dataset.collate_fn
        )

        # Load one batch
        batch = next(iter(dataloader))
        print(f"[OK] Batch loaded, keys: {list(batch.keys())}")

        # Check required keys for BEVDepth
        required_keys = ['front0', 'mats_dict', 'gt_boxes']
        for key in required_keys:
            assert key in batch, f"Missing key: {key}"
        print(f"[OK] All required keys present: {required_keys}")

        # Check mats_dict structure
        mats_dict = batch['mats_dict']
        mats_keys = ['sensor2ego_mats', 'intrin_mats', 'ida_mats', 'bda_mat']
        for key in mats_keys:
            assert key in mats_dict, f"Missing mats_dict key: {key}"
        print(f"[OK] mats_dict has all required keys: {mats_keys}")

        # Check shapes
        print(f"    - front0: {batch['front0'].shape}")
        print(f"    - gt_boxes: {batch['gt_boxes'].shape}")
        print(f"    - sensor2ego_mats: {mats_dict['sensor2ego_mats'].shape}")
        print(f"    - intrin_mats: {mats_dict['intrin_mats'].shape}")
        print(f"    - ida_mats: {mats_dict['ida_mats'].shape}")
        print(f"    - bda_mat: {mats_dict['bda_mat'].shape}")

        # Check depth_gt if available
        if 'depth_gt' in batch and batch['depth_gt'] is not None:
            print(f"    - depth_gt: {batch['depth_gt'].shape}")
        else:
            print("[WARN] depth_gt not available (generation may still be running)")

        return dataset, batch

    except ImportError as e:
        print(f"[WARN] Dataset import failed: {e}")
        print("[INFO] Falling back to synthetic data for model testing")
        batch = create_synthetic_batch(cfg)
        return None, batch

def test_model_instantiation():
    """Test StudentBEVDepth model creation"""
    print("\n" + "="*60)
    print("TEST 2: StudentBEVDepth Model Instantiation")
    print("="*60)

    from models.skeletons import build_skeleton
    cfg = load_config()

    # Build model
    model = build_skeleton(cfg, mode='train')
    print(f"[OK] Model created: {type(model).__name__}")

    # Check model components
    assert hasattr(model, 'bevdepth'), "Missing bevdepth component"
    assert hasattr(model, 'dense_head'), "Missing dense_head component"
    print("[OK] Model has bevdepth and dense_head components")

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[OK] Total parameters: {total_params:,}")
    print(f"[OK] Trainable parameters: {trainable_params:,}")

    return model

def test_forward_pass(model, batch):
    """Test model forward pass"""
    print("\n" + "="*60)
    print("TEST 3: Forward Pass")
    print("="*60)

    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")
    model = model.to(device)

    # Print batch info
    print(f"[OK] front0 batch shape: {batch['front0'].shape}")
    print(f"[OK] gt_boxes batch shape: {batch['gt_boxes'].shape}")

    # Move batch to device
    batch_cuda = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch_cuda[k] = v.to(device)
        elif isinstance(v, dict):
            batch_cuda[k] = {kk: vv.to(device) if isinstance(vv, torch.Tensor) else vv for kk, vv in v.items()}
        else:
            batch_cuda[k] = v

    # Forward pass
    try:
        with torch.no_grad():
            output = model(batch_cuda)
        print(f"[OK] Forward pass successful!")
        print(f"[OK] Output keys: {list(output.keys()) if isinstance(output, dict) else type(output)}")

        if isinstance(output, dict):
            if 'cls_preds' in output:
                print(f"[OK] cls_preds shape: {output['cls_preds'].shape}")
            if 'box_preds' in output:
                print(f"[OK] box_preds shape: {output['box_preds'].shape}")

    except Exception as e:
        print(f"[ERROR] Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

def test_loss_computation(model, batch):
    """Test loss computation"""
    print("\n" + "="*60)
    print("TEST 4: Loss Computation")
    print("="*60)

    model.train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Move batch to device
    batch_cuda = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch_cuda[k] = v.to(device)
        elif isinstance(v, dict):
            batch_cuda[k] = {kk: vv.to(device) if isinstance(vv, torch.Tensor) else vv for kk, vv in v.items()}
        else:
            batch_cuda[k] = v

    try:
        # Forward pass first
        output = model(batch_cuda)

        # Then compute loss
        loss_result = model.loss(output)
        print(f"[OK] Loss computation successful!")

        # Handle both dict and tensor returns
        if isinstance(loss_result, dict):
            print(f"[OK] Loss dict keys: {list(loss_result.keys())}")
            for key, val in loss_result.items():
                if isinstance(val, torch.Tensor):
                    print(f"    - {key}: {val.item():.4f}")
                else:
                    print(f"    - {key}: {val}")
        elif isinstance(loss_result, (tuple, list)):
            print(f"[OK] Loss returned {len(loss_result)} values")
            for i, val in enumerate(loss_result):
                if isinstance(val, torch.Tensor):
                    print(f"    - loss[{i}]: {val.item():.4f}")
                elif isinstance(val, dict):
                    print(f"    - loss[{i}] keys: {list(val.keys())}")
                else:
                    print(f"    - loss[{i}]: {val}")
        else:
            print(f"[OK] Loss value: {loss_result.item() if hasattr(loss_result, 'item') else loss_result}")

    except Exception as e:
        print(f"[ERROR] Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def main():
    print("="*60)
    print("BEVDepth Integration Test")
    print("="*60)

    all_passed = True

    # Test 1: Dataset
    try:
        dataset, batch = test_dataset()
    except Exception as e:
        print(f"[ERROR] Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
        return

    # Test 2: Model
    try:
        model = test_model_instantiation()
    except Exception as e:
        print(f"[ERROR] Model test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
        return

    # Test 3: Forward pass
    try:
        if not test_forward_pass(model, batch):
            all_passed = False
    except Exception as e:
        print(f"[ERROR] Forward pass test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # Test 4: Loss computation
    try:
        if not test_loss_computation(model, batch):
            all_passed = False
    except Exception as e:
        print(f"[ERROR] Loss test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED")
    print("="*60)


if __name__ == '__main__':
    main()
