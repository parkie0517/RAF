# Student model for BEVDepth cam-only training
# Uses BaseLSSFPN from BEVDepth
# Created: 2026-01-20

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ops.iou3d_nms import iou3d_nms_utils
from models import head
from models.model_utils import model_nms_utils
from .utils import common_utils

# BEVDepth modules
from models.bevdepth import BaseLSSFPN

# mmdet modules for BEV backbone/neck
from mmdet.models import build_backbone, build_neck


class StudentBEVDepth(nn.Module):
    """
    BEVDepth cam-only model using BaseLSSFPN
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model_cfg = cfg.MODEL
        self.dataset_cfg = cfg.DATASET

        # Class configuration
        self.num_class = 0
        self.class_names = []
        dict_label = self.cfg.DATASET.label.copy()
        list_for_pop = ['calib', 'onlyR', 'Label', 'consider_cls', 'consider_roi', 'remove_0_obj']
        for temp_key in list_for_pop:
            dict_label.pop(temp_key, None)
        self.dict_cls_name_to_id = dict()
        for k, v in dict_label.items():
            _, logit_idx, _, _ = v
            self.dict_cls_name_to_id[k] = logit_idx
            self.dict_cls_name_to_id['Background'] = 0
            if logit_idx > 0:
                self.num_class += 1
                self.class_names.append(k)

        self.model_cfg.DENSE_HEAD.CLASS_NAMES_EACH_HEAD.append(self.class_names)

        # BEV grid configuration
        point_cloud_range = np.array(self.dataset_cfg.roi.xyz)
        voxel_size = self.dataset_cfg.roi.voxel_size
        grid_size = (point_cloud_range[3:6] - point_cloud_range[0:3]) / np.array(voxel_size)
        grid_size = np.round(grid_size).astype(np.int64)

        model_info_dict = dict(
            grid_size=grid_size,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
        )

        # Build BaseLSSFPN (includes backbone, neck, depthnet, voxel pooling)
        vtransform_cfg = self.model_cfg.VTRANSFORM
        self.bevdepth = BaseLSSFPN(
            x_bound=vtransform_cfg.xbound,
            y_bound=vtransform_cfg.ybound,
            z_bound=vtransform_cfg.zbound,
            d_bound=vtransform_cfg.dbound,
            final_dim=vtransform_cfg.image_size,
            downsample_factor=vtransform_cfg.downsample,
            output_channels=vtransform_cfg.out_channels,
            img_backbone_conf=self.model_cfg.IMG_BACKBONE,
            img_neck_conf=self.model_cfg.IMG_NECK,
            depth_net_conf=vtransform_cfg.depthnet_cfg,
            use_da=False,
        )

        # Store loss config
        self.loss_cfg = vtransform_cfg.get('LOSS', {})

        # Depth loss parameters (from BEVDepth)
        self.downsample_factor = vtransform_cfg.downsample
        self.dbound = vtransform_cfg.dbound
        self.depth_channels = int((self.dbound[1] - self.dbound[0]) / self.dbound[2])
        self.depth_supervision_weight = self.loss_cfg.get('DEPTH_SUPERVISION_WEIGHT', 1.0)
        # BEV Backbone and Neck (from BEVDepth bev_depth_head.py)
        if hasattr(self.model_cfg, 'BEV_BACKBONE') and self.model_cfg.BEV_BACKBONE is not None:
            self.bev_backbone = build_backbone(self.model_cfg.BEV_BACKBONE)
            self.bev_backbone.init_weights()
            del self.bev_backbone.maxpool  # Remove maxpool like BEVDepth does

            self.bev_neck = build_neck(self.model_cfg.BEV_NECK)
            self.bev_neck.init_weights()

            # Update num_bev_features for detection head
            bev_neck_out_channels = sum(self.model_cfg.BEV_NECK.out_channels)  # 256
            model_info_dict['num_bev_features_cam'] = bev_neck_out_channels
        else:
            self.bev_backbone = None
            self.bev_neck = None
            model_info_dict['num_bev_features_cam'] = vtransform_cfg.out_channels

        # Detection head

        self.dense_head = head.__all__[self.model_cfg.DENSE_HEAD.NAME](
            model_cfg=self.model_cfg.DENSE_HEAD,
            input_channels=model_info_dict['num_bev_features_cam'],
            num_class=self.num_class if not self.model_cfg.DENSE_HEAD.CLASS_AGNOSTIC else 1,
            class_names=self.class_names,
            grid_size=model_info_dict['grid_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False),
            voxel_size=model_info_dict.get('voxel_size', False)
        )

        self.model_info_dict = model_info_dict
        self.is_logging = cfg.GENERAL.LOGGING.IS_LOGGING

    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def prepare_bevdepth_inputs(self, batch_dict):
        """
        Convert batch_dict to BEVDepth input format

        BEVDepth expects:
            - sweep_imgs: [B, num_sweeps, num_cams, C, H, W]
            - mats_dict: dict with sensor2ego_mats, intrin_mats, ida_mats, bda_mat
        """
        # Images: [B, C, H, W] -> [B, 1, 1, C, H, W]
        images = batch_dict['front0']  # [B, C, H, W]
        B, C, H, W = images.shape
        sweep_imgs = images.view(B, 1, 1, C, H, W)

        # mats_dict from dataset
        mats_dict = batch_dict['mats_dict']

        # Convert numpy arrays to tensors if needed
        for key in mats_dict:
            if isinstance(mats_dict[key], np.ndarray):
                mats_dict[key] = torch.from_numpy(mats_dict[key])
            mats_dict[key] = mats_dict[key].to(images.device)

        # Depth GT for supervision (if available)
        lidar_depth = batch_dict.get('depth_gt', None)
        if lidar_depth is not None:
            if isinstance(lidar_depth, np.ndarray):
                lidar_depth = torch.from_numpy(lidar_depth)
            # [B, 1, H, W] -> [B, 1, 1, H, W]
            if lidar_depth.dim() == 4:
                lidar_depth = lidar_depth.unsqueeze(2)  # [B, 1, 1, H, W]
            lidar_depth = lidar_depth.to(images.device)

        return sweep_imgs, mats_dict, lidar_depth

    def forward(self, batch_dict):
        # Move data to GPU
        batch_dict['front0'] = batch_dict['front0'].cuda()
        batch_dict['gt_boxes'] = batch_dict['gt_boxes'].cuda()

        # Prepare BEVDepth inputs
        sweep_imgs, mats_dict, lidar_depth = self.prepare_bevdepth_inputs(batch_dict)

        # Forward through BEVDepth
        # BaseLSSFPN returns (feature_map, depth) if is_return_depth=True
        is_return_depth = self.training and self.loss_cfg.get('DEPTH_SUPERVISION', False)
        bev_output = self.bevdepth(
            sweep_imgs=sweep_imgs,
            mats_dict=mats_dict,
            timestamps=None,
            is_return_depth=is_return_depth,
        )

        if is_return_depth and isinstance(bev_output, tuple):
            bev_feat, depth_pred = bev_output
            batch_dict['depth_pred'] = depth_pred
            batch_dict['lidar_depth'] = lidar_depth  # Store for loss computation
        else:
            bev_feat = bev_output

        # BEV Backbone + Neck processing (from BEVDepth bev_depth_head.py)
        if self.bev_backbone is not None:
            bev_feat = bev_feat.float()
            trunk_outs = [bev_feat]  # Keep initial features

            # Stem
            if self.bev_backbone.deep_stem:
                x = self.bev_backbone.stem(bev_feat)
            else:
                x = self.bev_backbone.conv1(bev_feat)
                x = self.bev_backbone.norm1(x)
                x = self.bev_backbone.relu(x)

            # Residual stages
            for i, layer_name in enumerate(self.bev_backbone.res_layers):
                res_layer = getattr(self.bev_backbone, layer_name)
                x = res_layer(x)
                if i in self.bev_backbone.out_indices:
                    trunk_outs.append(x)

            # FPN neck
            bev_feat = self.bev_neck(trunk_outs)

        # BEV feature: [B, C, Ny, Nx] - already in correct order for anchor head
        # BaseLSSFPN outputs [B, C, Y, X] = [B, C, Ny, Nx] (verified by runtime probe)
        # Anchor head expects spatial dims [H, W] = [Ny, Nx] - NO permute needed!
        batch_dict['camera_spatial_features'] = bev_feat  # DO NOT permute

        # Store depth GT for loss computation
        if lidar_depth is not None:
            batch_dict['lidar_depth'] = lidar_depth

        # Detection head
        batch_dict = self.dense_head(batch_dict)

        if self.training:
            return batch_dict
        else:
            batch_dict = self.post_processing(batch_dict)
            return batch_dict

    def loss(self, batch_dict):
        # Detection loss
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss = loss_rpn

        # Depth supervision loss (from BEVDepth base_exp.py)
        depth_loss = torch.tensor(0.0, device=loss.device)
        if self.loss_cfg.get('DEPTH_SUPERVISION', False):
            if 'depth_pred' in batch_dict and 'lidar_depth' in batch_dict:
                depth_pred = batch_dict['depth_pred']  # [B*N, D, H, W]
                depth_labels = batch_dict['lidar_depth']  # [B, 1, 1, H, W]

                # Reshape depth_labels: [B, 1, 1, H, W] -> [B, N, H, W] where N=1
                B = depth_labels.shape[0]
                depth_labels = depth_labels.view(B, 1, depth_labels.shape[3], depth_labels.shape[4])

                depth_loss = self.get_depth_loss(depth_labels, depth_pred)
                loss = loss + depth_loss

                if self.is_logging:
                    tb_dict['depth_loss'] = depth_loss.item()

        if self.is_logging:
            batch_dict['logging'] = dict()
            batch_dict['logging'].update(tb_dict)

        return loss

    def get_depth_loss(self, depth_labels, depth_preds):
        """
        Compute depth loss (from BEVDepth base_exp.py)

        Args:
            depth_labels: [B, N, H, W] GT depth values
            depth_preds: [B*N, D, h, w] predicted depth distribution
        """
        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(
            -1, self.depth_channels)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0

        with torch.cuda.amp.autocast(enabled=False):
            # Clamp predictions to prevent numerical instability in BCE
            # When softmax outputs very small values, BCE computes -log(small) -> large
            depth_preds_clamped = depth_preds[fg_mask].clamp(min=1e-4, max=1-1e-4)
            depth_loss = (F.binary_cross_entropy(
                depth_preds_clamped,
                depth_labels[fg_mask],
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum()))

        return self.depth_supervision_weight * depth_loss # 3.0 is the depth loss weight

    def get_downsampled_gt_depth(self, gt_depths):
        """
        Downsample GT depth and convert to one-hot (from BEVDepth base_exp.py)

        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(
            B * N,
            H // self.downsample_factor,
            self.downsample_factor,
            W // self.downsample_factor,
            self.downsample_factor,
            1,
        )
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(
            -1, self.downsample_factor * self.downsample_factor)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample_factor,
                                   W // self.downsample_factor)

        gt_depths = (gt_depths -
                     (self.dbound[0] - self.dbound[2])) / self.dbound[2]
        gt_depths = torch.where(
            (gt_depths < self.depth_channels + 1) & (gt_depths >= 0.0),
            gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(),
                              num_classes=self.depth_channels + 1).view(
                                  -1, self.depth_channels + 1)[:, 1:]

        return gt_depths.float()

    def post_processing(self, batch_dict):
        """Post-processing for inference"""
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        recall_dict = {}
        pred_dicts = []

        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_dict['batch_box_preds'].shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_box_preds'].shape.__len__() == 3
                batch_mask = index

            box_preds = batch_dict['batch_box_preds'][batch_mask]
            src_box_preds = box_preds

            if not isinstance(batch_dict['batch_cls_preds'], list):
                cls_preds = batch_dict['batch_cls_preds'][batch_mask]
                src_cls_preds = cls_preds
                assert cls_preds.shape[1] in [1, self.num_class]

                if not batch_dict['cls_preds_normalized']:
                    cls_preds = torch.sigmoid(cls_preds)
            else:
                cls_preds = [x[batch_mask] for x in batch_dict['batch_cls_preds']]
                src_cls_preds = cls_preds
                if not batch_dict['cls_preds_normalized']:
                    cls_preds = [torch.sigmoid(x) for x in cls_preds]

            if post_process_cfg.NMS_CONFIG.MULTI_CLASSES_NMS:
                if not isinstance(cls_preds, list):
                    cls_preds = [cls_preds]
                    multihead_label_mapping = [torch.arange(1, self.num_class, device=cls_preds[0].device)]
                else:
                    multihead_label_mapping = batch_dict['multihead_label_mapping']

                cur_start_idx = 0
                pred_scores, pred_labels, pred_boxes = [], [], []
                for cur_cls_preds, cur_label_mapping in zip(cls_preds, multihead_label_mapping):
                    assert cur_cls_preds.shape[1] == len(cur_label_mapping)
                    cur_box_preds = box_preds[cur_start_idx: cur_start_idx + cur_cls_preds.shape[0]]
                    cur_pred_scores, cur_pred_labels, cur_pred_boxes = model_nms_utils.multi_classes_nms(
                        cls_scores=cur_cls_preds, box_preds=cur_box_preds,
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=post_process_cfg.SCORE_THRESH
                    )
                    cur_pred_labels = cur_label_mapping[cur_pred_labels]
                    pred_scores.append(cur_pred_scores)
                    pred_labels.append(cur_pred_labels)
                    pred_boxes.append(cur_pred_boxes)
                    cur_start_idx += cur_cls_preds.shape[0]

                final_scores = torch.cat(pred_scores, dim=0)
                final_labels = torch.cat(pred_labels, dim=0)
                final_boxes = torch.cat(pred_boxes, dim=0)
            else:
                cls_preds, label_preds = torch.max(cls_preds, dim=-1)
                if batch_dict.get('has_class_labels', False):
                    label_key = 'roi_labels' if 'roi_labels' in batch_dict else 'batch_pred_labels'
                    label_preds = batch_dict[label_key][index]
                else:
                    label_preds = label_preds + 1

                selected, selected_scores = model_nms_utils.class_agnostic_nms(
                    box_scores=cls_preds, box_preds=box_preds,
                    nms_config=post_process_cfg.NMS_CONFIG,
                    score_thresh=post_process_cfg.SCORE_THRESH
                )

                if post_process_cfg.OUTPUT_RAW_SCORE:
                    max_cls_preds, _ = torch.max(src_cls_preds, dim=-1)
                    selected_scores = max_cls_preds[selected]

                final_scores = selected_scores
                final_labels = label_preds[selected]
                final_boxes = box_preds[selected]

            recall_dict = self.generate_recall_record(
                box_preds=final_boxes if 'rois' not in batch_dict else src_box_preds,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

            record_dict = {
                'pred_boxes': final_boxes,
                'pred_scores': final_scores,
                'pred_labels': final_labels
            }
            pred_dicts.append(record_dict)

        batch_dict['pred_dicts'] = pred_dicts
        batch_dict['recall_dict'] = recall_dict

        return batch_dict

    @staticmethod
    def generate_recall_record(box_preds, recall_dict, batch_index, data_dict=None, thresh_list=None):
        if 'gt_boxes' not in data_dict:
            return recall_dict

        rois = data_dict['rois'][batch_index] if 'rois' in data_dict else None
        gt_boxes = data_dict['gt_boxes'][batch_index]

        if recall_dict.__len__() == 0:
            recall_dict = {'gt': 0}
            for cur_thresh in thresh_list:
                recall_dict['roi_%s' % (str(cur_thresh))] = 0
                recall_dict['rcnn_%s' % (str(cur_thresh))] = 0

        cur_gt = gt_boxes
        k = cur_gt.__len__() - 1
        while k >= 0 and cur_gt[k].sum() == 0:
            k -= 1
        cur_gt = cur_gt[:k + 1]

        if cur_gt.shape[0] > 0:
            if box_preds.shape[0] > 0:
                iou3d_rcnn = iou3d_nms_utils.boxes_iou3d_gpu(box_preds[:, 0:7], cur_gt[:, 0:7])
            else:
                iou3d_rcnn = torch.zeros((0, cur_gt.shape[0]))

            if rois is not None:
                iou3d_roi = iou3d_nms_utils.boxes_iou3d_gpu(rois[:, 0:7], cur_gt[:, 0:7])

            for cur_thresh in thresh_list:
                if iou3d_rcnn.shape[0] == 0:
                    recall_dict['rcnn_%s' % str(cur_thresh)] += 0
                else:
                    rcnn_recalled = (iou3d_rcnn.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['rcnn_%s' % str(cur_thresh)] += rcnn_recalled
                if rois is not None:
                    roi_recalled = (iou3d_roi.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['roi_%s' % str(cur_thresh)] += roi_recalled

            recall_dict['gt'] += cur_gt.shape[0]

        return recall_dict
