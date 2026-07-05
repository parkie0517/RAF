import os
import torch
import torch.nn as nn
import numpy as np

from ops.iou3d_nms import iou3d_nms_utils
from utils.spconv_utils import find_all_spconv_keys
from models import backbone_2d, backbone_3d, head, roi_head
from models.backbone_2d import map_to_bev
from models.backbone_3d import pfe, vfe
from models.model_utils import model_nms_utils
from .utils import common_utils
tv = None
try:
    import cumm.tensorview as tv
except:
    pass

class VoxelGeneratorWrapper():
    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            self.spconv_ver = 1
        except:
            try:
                from spconv.utils import VoxelGenerator
                self.spconv_ver = 1
            except:
                from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
                self.spconv_ver = 2

        if self.spconv_ver == 1:
            self._voxel_generator = VoxelGenerator(
                voxel_size=vsize_xyz,
                point_cloud_range=coors_range_xyz,
                max_num_points=max_num_points_per_voxel,
                max_voxels=max_num_voxels
            )
        else:
            self._voxel_generator = VoxelGenerator(
                vsize_xyz=vsize_xyz,
                coors_range_xyz=coors_range_xyz,
                num_point_features=num_point_features,
                max_num_points_per_voxel=max_num_points_per_voxel,
                max_num_voxels=max_num_voxels
            )

    def generate(self, points):
        if self.spconv_ver == 1:
            voxel_output = self._voxel_generator.generate(points)
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output
        else:
            assert tv is not None, f"Unexpected error, library: 'cumm' wasn't imported properly."
            voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(points))
            tv_voxels, tv_coordinates, tv_num_points = voxel_output
            # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
            voxels = tv_voxels.numpy()
            coordinates = tv_coordinates.numpy()
            num_points = tv_num_points.numpy()
        return voxels, coordinates, num_points

class PointPillar_RLF_DT(nn.Module):
    def __init__(self, cfg, mode='train'):
        super().__init__()
        self.cfg = cfg
        self.model_cfg = cfg.MODEL
        self.dataset_cfg = cfg.DATASET
        self.mode_init = mode  # 'train' or 'test' (for pretrained loading control)

        # class
        self.num_class = 0
        self.class_names = []
        dict_label = self.cfg.DATASET.label.copy()
        list_for_pop = ['calib', 'onlyR', 'Label', 'consider_cls', 'consider_roi', 'remove_0_obj']
        for temp_key in list_for_pop:
            dict_label.pop(temp_key)
        self.dict_cls_name_to_id = dict()
        for k, v in dict_label.items():
            _, logit_idx, _, _ = v
            self.dict_cls_name_to_id[k] = logit_idx
            self.dict_cls_name_to_id['Background'] = 0
            if logit_idx > 0:
                self.num_class += 1
                self.class_names.append(k)
        # print(self.class_names)
        self.model_cfg.DENSE_HEAD.CLASS_NAMES_EACH_HEAD.append(self.class_names)
        # print(self.cfg.MODEL.DENSE_HEAD.CLASS_NAMES_EACH_HEAD)
        
        # Common params
        num_point_features = [self.dataset_cfg.ldr64.n_used,self.dataset_cfg.rdr_sparse.n_used]
        self.num_point_features = num_point_features
        point_cloud_range = np.array(self.dataset_cfg.roi.xyz)
        voxel_size = self.dataset_cfg.roi.voxel_size
        grid_size = (point_cloud_range[3:6] - point_cloud_range[0:3]) / np.array(voxel_size)
        grid_size = np.round(grid_size).astype(np.int64)
        model_info_dict = dict(
            module_list = [],
            num_rawpoint_features = num_point_features,
            num_point_features = num_point_features,
            grid_size = grid_size,
            point_cloud_range = point_cloud_range,
            voxel_size = voxel_size,
        )
        self.ldr_voxel_generator_train = VoxelGeneratorWrapper(
            vsize_xyz=voxel_size,
            coors_range_xyz=point_cloud_range,
            num_point_features=num_point_features[0],
            max_num_points_per_voxel=self.model_cfg.PRE_PROCESSING.MAX_POINTS_PER_VOXEL,
            max_num_voxels=self.model_cfg.PRE_PROCESSING.MAX_NUMBER_OF_VOXELS['train'],
        )
        self.ldr_voxel_generator_test = VoxelGeneratorWrapper(
            vsize_xyz=voxel_size,
            coors_range_xyz=point_cloud_range,
            num_point_features=num_point_features[0],
            max_num_points_per_voxel=self.model_cfg.PRE_PROCESSING.MAX_POINTS_PER_VOXEL,
            max_num_voxels=self.model_cfg.PRE_PROCESSING.MAX_NUMBER_OF_VOXELS['test'],
        )
        self.rdr_voxel_generator_train = VoxelGeneratorWrapper(
            vsize_xyz=voxel_size,
            coors_range_xyz=point_cloud_range,
            num_point_features=num_point_features[1],
            max_num_points_per_voxel=self.model_cfg.PRE_PROCESSING.MAX_POINTS_PER_VOXEL,
            max_num_voxels=self.model_cfg.PRE_PROCESSING.MAX_NUMBER_OF_VOXELS['train'],
        )
        self.rdr_voxel_generator_test = VoxelGeneratorWrapper(
            vsize_xyz=voxel_size,
            coors_range_xyz=point_cloud_range,
            num_point_features=num_point_features[1],
            max_num_points_per_voxel=self.model_cfg.PRE_PROCESSING.MAX_POINTS_PER_VOXEL,
            max_num_voxels=self.model_cfg.PRE_PROCESSING.MAX_NUMBER_OF_VOXELS['test'],
        )

        self.point_head = None
        # Build modules
        self.vfe = vfe.__all__[self.model_cfg.VFE.NAME](
            model_cfg=self.model_cfg.VFE,
            num_point_features=model_info_dict['num_rawpoint_features'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            voxel_size=model_info_dict['voxel_size'],
            grid_size=model_info_dict['grid_size'],
        )
        model_info_dict['num_point_features'] = self.vfe.get_output_feature_dim()
        self.backbone_3d = backbone_3d.__all__[self.model_cfg.BACKBONE_3D.NAME](
            model_cfg=self.model_cfg.BACKBONE_3D,
            input_channels=model_info_dict['num_rawpoint_features'][0],
            grid_size=model_info_dict['grid_size'],
            voxel_size=model_info_dict['voxel_size'],
            point_cloud_range=model_info_dict['point_cloud_range']
        )
        model_info_dict['num_point_features'] = self.backbone_3d.num_point_features
        model_info_dict['backbone_channels'] = self.backbone_3d.backbone_channels \
            if hasattr(self.backbone_3d, 'backbone_channels') else None
        self.point_head = head.__all__[self.model_cfg.POINT_HEAD.NAME](
            model_cfg=self.model_cfg.POINT_HEAD,
            input_channels= model_info_dict['num_point_features'],
            num_class=self.num_class if not self.model_cfg.POINT_HEAD.CLASS_AGNOSTIC else 1,
            predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False)
        )
        self.map_to_bev_module = map_to_bev.__all__[self.model_cfg.MAP_TO_BEV.NAME](
            model_cfg=self.model_cfg.MAP_TO_BEV,
            grid_size=model_info_dict['grid_size']
        )
        
        model_info_dict['num_bev_features'] = np.array(self.map_to_bev_module.num_bev_features).sum()
        self.backbone_2d = backbone_2d.__all__[self.model_cfg.BACKBONE_2D.NAME](
            model_cfg=self.model_cfg.BACKBONE_2D,
            input_channels=model_info_dict.get('num_bev_features', None)
        )
        model_info_dict['num_bev_features'] = self.backbone_2d.num_bev_features

        # Stage-based: Create BaseBEVBackbone_MGF_RT for stage 2 or 3
        # RT (Radar Teacher) backbone to generate Radar_plus feature (768ch)
        self.stage = cfg.GENERAL.STAGE
        self.backbone_2d_rt = None
        if self.stage in [2, 3]:
            # radar channel is the second element of num_bev_features [64, 64]
            radar_input_channels = self.map_to_bev_module.num_bev_features[1]  # 64ch
            self.backbone_2d_rt = backbone_2d.__all__[self.model_cfg.BACKBONE_2D_RT.NAME](
                model_cfg=self.model_cfg.BACKBONE_2D_RT,
                input_channels=radar_input_channels
            )

        self.dense_head = head.__all__[self.model_cfg.DENSE_HEAD.NAME](
            model_cfg=self.model_cfg.DENSE_HEAD,
            input_channels=model_info_dict['num_bev_features'] if 'num_bev_features' in model_info_dict else self.model_cfg.DENSE_HEAD.INPUT_FEATURES,
            num_class=self.num_class if not self.model_cfg.DENSE_HEAD.CLASS_AGNOSTIC else 1,
            class_names=self.class_names,
            grid_size=model_info_dict['grid_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False),
            voxel_size=model_info_dict.get('voxel_size', False),
            stage=self.stage  # Stage for feature selection in forward()
        )
        

        self.model_info_dict = model_info_dict
        
        # student model
        self.student = None

        # Pre-processor
        self.is_pre_processing = self.model_cfg.PRE_PROCESSING.get('VER', None)
        self.shuffle_points = self.model_cfg.PRE_PROCESSING.get('SHUFFLE_POINTS', False)
        self.transform_points_to_voxels = self.model_cfg.PRE_PROCESSING.get('TRANSFORM_POINTS_TO_VOXELS', False)
        self.TP = common_utils.AverageMeter()
        self.P = common_utils.AverageMeter()
        self.TP_FN = common_utils.AverageMeter()
        self.TP_FP_FN = common_utils.AverageMeter()
        self.is_logging = cfg.GENERAL.LOGGING.IS_LOGGING
        
        # Train mode only: Load pretrained weights and freeze
        # Test mode: Skip (full checkpoint loaded by main_eval or main_eval_del)
        if self.mode_init == 'train':
            self._load_pretrained_weights()  # Step 7
            self._freeze_weights()           # Step 10
        else:
            print(f"[Test mode] Skipping pretrained loading and freezing (stage {self.stage})")

    def _freeze_weights(self):
        """
        Stage 1: No freezing (train all weights)
        Stage 2: Freeze all except backbone_2d_rt
        Stage 3: Freeze all except student
        """
        if self.stage == 1:
            # Stage 1: All weights trainable
            return

        if self.stage == 2:
            # Stage 2: Freeze all except backbone_2d_rt
            for name, param in self.named_parameters():
                if name.startswith('backbone_2d_rt.'):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            # Count trainable parameters
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            frozen_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
            print(f"[Stage 2] Frozen: {frozen_params:,} params | Trainable (backbone_2d_rt): {trainable_params:,} params")

        if self.stage == 3:
            # Stage 3: Freeze all except student
            for name, param in self.named_parameters():
                if name.startswith('student.'):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            # Count trainable parameters
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            frozen_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
            print(f"[Stage 3] Frozen: {frozen_params:,} params | Trainable (student): {trainable_params:,} params")

    def _load_pretrained_weights(self):
        """
        Stage 1: No pretrained weights needed
        Stage 2: Load L4DR pretrained weights (backbone_2d_rt is randomly initialized)
        Stage 3: Load L4DR_DT pretrained weights (student is randomly initialized)
        """
        if self.stage == 1:
            return

        if self.stage == 2:
            pretrained_path = self.model_cfg.PRETRAINED.PATH
            if pretrained_path is None:
                raise ValueError("PRETRAINED.PATH must be specified for stage 2")
            if not os.path.exists(pretrained_path):
                raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_path}")

            print(f"[Stage 2] Loading pretrained L4DR weights from: {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location='cpu')

            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                pretrained_state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                pretrained_state_dict = checkpoint['state_dict']
            else:
                pretrained_state_dict = checkpoint

            # Get current model state dict
            model_state_dict = self.state_dict()

            # Check for missing keys (excluding backbone_2d_rt)
            missing_keys = []
            loaded_keys = []
            for key in model_state_dict.keys():
                if key.startswith('backbone_2d_rt.'):
                    # Skip backbone_2d_rt keys (expected to be missing in pretrained)
                    continue
                if key in pretrained_state_dict:
                    loaded_keys.append(key)
                else:
                    missing_keys.append(key)

            # Raise error if there are unexpected missing keys
            if len(missing_keys) > 0:
                raise RuntimeError(
                    f"[Stage 2] Missing keys in pretrained checkpoint (excluding backbone_2d_rt):\n"
                    f"{missing_keys}"
                )

            # Load pretrained weights (strict=False to allow missing backbone_2d_rt)
            self.load_state_dict(pretrained_state_dict, strict=False)
            print(f"[Stage 2] Successfully loaded {len(loaded_keys)} keys from pretrained checkpoint")
            print(f"[Stage 2] backbone_2d_rt ({sum(p.numel() for p in self.backbone_2d_rt.parameters())} params) will be trained from scratch")

        if self.stage == 3:
            pretrained_path = self.model_cfg.PRETRAINED.PATH
            if pretrained_path is None:
                raise ValueError("PRETRAINED.PATH must be specified for stage 3")
            if not os.path.exists(pretrained_path):
                raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_path}")

            print(f"[Stage 3] Loading pretrained L4DR_DT weights from: {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location='cpu')

            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                pretrained_state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                pretrained_state_dict = checkpoint['state_dict']
            else:
                pretrained_state_dict = checkpoint

            # Get current model state dict
            model_state_dict = self.state_dict()

            # Check for missing keys (excluding student)
            missing_keys = []
            loaded_keys = []
            for key in model_state_dict.keys():
                if key.startswith('student.'):
                    # Skip student keys (expected to be missing in pretrained)
                    continue
                if key in pretrained_state_dict:
                    loaded_keys.append(key)
                else:
                    missing_keys.append(key)

            # Raise error if there are unexpected missing keys
            if len(missing_keys) > 0:
                raise RuntimeError(
                    f"[Stage 3] Missing keys in pretrained checkpoint:\n"
                    f"{missing_keys}"
                )
            
            # Load pretrained weights (strict=False to allow missing backbone_2d_rt)
            self.load_state_dict(pretrained_state_dict, strict=False)
            print(f"[Stage 3] Successfully loaded {len(loaded_keys)} keys from pretrained checkpoint")
            print(f"[Stage 3] student ({sum(p.numel() for p in self.studnet.parameters())} params) will be trained from scratch")

    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'
    
    def pre_processor(self, batch_dict):
        if self.is_pre_processing is None:
            return batch_dict
        elif self.is_pre_processing == 'v1_0':
            # Shuffle (DataProcessor.shuffle_points)
            batched_rdr= batch_dict['rdr_sparse'].detach()
            batched_indices_rdr= batch_dict['batch_indices_rdr_sparse'].detach()
            list_points = []
            list_voxels = []
            list_voxel_coords = []
            list_voxel_num_points = []
            for batch_idx in range(batch_dict['batch_size']):
                temp_points = batched_rdr[torch.where(batched_indices_rdr == batch_idx)[0],:self.num_point_features[1]]
                
                if (self.shuffle_points) and (self.training):
                    shuffle_idx = np.random.permutation(temp_points.shape[0])
                    temp_points = temp_points[shuffle_idx,:]
                list_points.append(temp_points)
                
                if self.transform_points_to_voxels:
                    if self.training:
                        voxels, coordinates, num_points = self.rdr_voxel_generator_train.generate(temp_points.cpu().numpy())
                    else:
                        voxels, coordinates, num_points = self.rdr_voxel_generator_test.generate(temp_points.cpu().numpy())
                    voxel_batch_idx = np.full((coordinates.shape[0], 1), batch_idx, dtype=np.int64)
                    coordinates = np.concatenate((voxel_batch_idx, coordinates), axis=-1) # bzyx

                    list_voxels.append(voxels)
                    list_voxel_coords.append(coordinates)
                    list_voxel_num_points.append(num_points)
            
            batched_points = torch.cat(list_points, dim=0)
            batch_dict['radar_points'] = torch.cat((batched_indices_rdr.reshape(-1,1), batched_points), dim=1).cuda() # b, x, y, z, intensity
            batch_dict['radar_voxels'] = torch.from_numpy(np.concatenate(list_voxels, axis=0)).cuda()
            batch_dict['radar_voxel_coords'] = torch.from_numpy(np.concatenate(list_voxel_coords, axis=0)).cuda()
            batch_dict['radar_voxel_num_points'] = torch.from_numpy(np.concatenate(list_voxel_num_points, axis=0)).cuda()
            batch_dict['gt_boxes'] = batch_dict['gt_boxes'].cuda()
            batch_dict['points'] = batch_dict['radar_points']

            batched_ldr64 = batch_dict['ldr64']
            batched_indices_ldr64 = batch_dict['batch_indices_ldr64']
            list_points = []
            list_voxels = []
            list_voxel_coords = []
            list_voxel_num_points = []
            for batch_idx in range(batch_dict['batch_size']):
                temp_points = batched_ldr64[torch.where(batched_indices_ldr64 == batch_idx)[0],:self.num_point_features[0]]
                if (self.shuffle_points) and (self.training):
                    shuffle_idx = np.random.permutation(temp_points.shape[0])
                    temp_points = temp_points[shuffle_idx,:]
                list_points.append(temp_points)
                
                if self.transform_points_to_voxels:
                    if self.training:
                        voxels, coordinates, num_points = self.ldr_voxel_generator_train.generate(temp_points.cpu().numpy())
                    else:
                        voxels, coordinates, num_points = self.ldr_voxel_generator_test.generate(temp_points.cpu().numpy())
                    voxel_batch_idx = np.full((coordinates.shape[0], 1), batch_idx, dtype=np.int64)
                    coordinates = np.concatenate((voxel_batch_idx, coordinates), axis=-1) # bzyx

                    list_voxels.append(voxels)
                    list_voxel_coords.append(coordinates)
                    list_voxel_num_points.append(num_points)
            
            batched_points = torch.cat(list_points, dim=0)
            batch_dict['lidar_points'] = torch.cat((batched_indices_ldr64.reshape(-1,1), batched_points), dim=1).cuda() # b, x, y, z, intensity
            batch_dict['lidar_voxels'] = torch.from_numpy(np.concatenate(list_voxels, axis=0)).cuda()
            batch_dict['lidar_voxel_coords'] = torch.from_numpy(np.concatenate(list_voxel_coords, axis=0)).cuda()
            batch_dict['lidar_voxel_num_points'] = torch.from_numpy(np.concatenate(list_voxel_num_points, axis=0)).cuda()
            return batch_dict
        
    def forward(self, batch_dict):
        batch_dict = self.pre_processor(batch_dict)
        batch_dict = self.backbone_3d(batch_dict)
        batch_dict = self.point_head(batch_dict)
        pre_mask = batch_dict['point_cls_scores'] > self.model_cfg.PRE_PROCESSING.DENOISE_T

        
        #to keep inference when have few fore_radar_points
        if pre_mask.sum() < 10:
            pre_mask[:10] = 1
        
        extra_choice = torch.ones(batch_dict['point_cls_scores'][pre_mask].shape,dtype = bool)
        batch_dict['raw_rdr_sparse'] = batch_dict['rdr_sparse'][pre_mask]

        batch_dict['batch_indices_rdr_sparse'] = batch_dict['batch_indices_rdr_sparse'][pre_mask][extra_choice]
        try:
            batch_dict['rdr_sparse'] = torch.cat([batch_dict['rdr_sparse'][pre_mask][extra_choice],batch_dict['point_cls_scores'][pre_mask][extra_choice].reshape(-1,1)], dim=1)
        except:
            batch_dict['rdr_sparse'] = torch.cat([batch_dict['rdr_sparse'][pre_mask][extra_choice],batch_dict['point_cls_scores'][pre_mask][extra_choice].reshape(-1,1).detach().cpu()], dim=1)
        batch_dict = self.pre_processor(batch_dict)
        batch_dict = self.vfe(batch_dict)
        batch_dict = self.map_to_bev_module(batch_dict)

        # Stage-based backbone_2d execution
        if self.stage == 1:
            # Stage 1: Only BaseBEVBackbone_MGF (L4DR training)
            batch_dict = self.backbone_2d(batch_dict)
        elif self.stage == 2:
            # Stage 2: Only BaseBEVBackbone_MGF_RT (Radar Teacher training)
            # Skip backbone_2d (frozen L4DR), only run backbone_2d_rt
            batch_dict = self.backbone_2d_rt(batch_dict)
        elif self.stage == 3:
            # Stage 3: Both backbones (for KD to student)
            batch_dict = self.backbone_2d(batch_dict)      # -> spatial_features_2d (Fused)
            batch_dict = self.backbone_2d_rt(batch_dict)   # -> spatial_features_2d_rp (Radar+)

        batch_dict = self.dense_head(batch_dict)
        if self.training:
            return batch_dict
        else:

            batch_dict = self.post_processing(batch_dict)
            return batch_dict

    def loss(self, dict_item):
        loss_rpn, tb_dict = self.dense_head.get_loss()
        if self.point_head is not None:
            loss_point, tb_dict = self.point_head.get_loss(tb_dict)
        else:
            loss_point = 0
        loss = loss_rpn + loss_point 

        if self.is_logging:
            dict_item['logging'] = dict()
            dict_item['logging'].update(tb_dict)

        return loss

    def post_processing(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                                or [(B, num_boxes, num_class1), (B, num_boxes, num_class2) ...]
                multihead_label_mapping: [(num_class1), (num_class2), ...]
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                has_class_labels: True/False
                roi_labels: (B, num_rois)  1 .. num_classes
                batch_pred_labels: (B, num_boxes, 1)
        Returns:

        """
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
        else:
            gt_iou = box_preds.new_zeros(box_preds.shape[0])
        return recall_dict
