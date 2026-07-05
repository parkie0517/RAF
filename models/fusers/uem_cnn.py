import torch
import torch.nn as nn
import torch.nn.functional as F



class FeatureAlignmentMLP(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, output_dim=128, dropout=0.1):
        super(FeatureAlignmentMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)

class ConfHead(nn.Module):
    def __init__(self,
                 input_dim = 256,
                 ):
        super().__init__()
        self.conf_head = nn.Sequential(
            nn.Conv2d(input_dim, input_dim//2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(input_dim//2, 1, kernel_size=1), 
            nn.Sigmoid()
        )
        
        
    def forward(self, feature_map):
        """
        Input:
            - feature_map: torch.Tensor [B, C, H, W]
        Output:
            - conf_map: torch.Tensor [B, 1, H, W]
        """
        
        conf_head = self.conf_head(feature_map)
        return conf_head
    

class UEM_CNN(nn.Module):
    def __init__(self,
                 model_cfg,
                 point_cloud_range, # [ 0. , -6.4, -2. , 72. ,  6.4,  6. ]
                 voxel_size, # 0.4
                 encoder_stride, # [1, 2, 2]
                 uem_loss_dict,
                 **kwargs # {}
                 ):
        super().__init__()
        self.model_cfg = model_cfg
        self.voxel_size = voxel_size
        total_stride = 1
        for n in encoder_stride:
            total_stride *= n
        self.scaled_voxel_size = voxel_size * total_stride # 0.4 * 4 = 1.6
        self.x_min, self.y_min, self.z_min = point_cloud_range[0], point_cloud_range[1], point_cloud_range[2], # 0.0, -6.4, -2.0
        
        # latent space 보내는데 필요한 모델
        self.lidar_projector = FeatureAlignmentMLP()
        self.image_projector = FeatureAlignmentMLP()
        
        # confidence map 생성하는 head
        self.conf_head = ConfHead()

        # loss 관련
        self.uem_loss_dict = uem_loss_dict
        
        
        
    def getSimilarityLoss(self, tb_dict, climate_weight_tensor):
    
        climate_weight_tensor = climate_weight_tensor.to(self.similarity.device)
        climate_weights = climate_weight_tensor[self.valid_b] # [N_new]
        uem_loss = 1 - self.similarity # 0~2 사이로 변화
        uem_loss = uem_loss * climate_weights
        uem_loss = uem_loss.mean()
        
        uem_loss = uem_loss * self.uem_loss_dict['sim_loss_weight'] # uem에 맞는 거 갖고오기
        
        tb_dict.update({
            'uem_loss': uem_loss.item(),
        })
        return uem_loss, tb_dict
    
    
    def getConfidenceLoss(self, tb_dict, climate_weight_tensor):

        conf_map = self.conf_map.squeeze(1)  # [B, H, W]
        B, H, W = conf_map.shape
        device = conf_map.device

        # flatten conf_map and projected coordinates
        flat_conf = conf_map.view(-1) # [B, H, W] -> [B*H*W]
        index = (self.valid_b * (H * W) + self.y * W + self.x).to(device=device, dtype=torch.long)

        pseudo_gt = self.pseudo_gt.detach() # 상수로 취급해야 함
        sum_pseudo = torch.zeros(B * H * W, device=device) # 합 계산용
        count = torch.zeros_like(sum_pseudo) # count 계산용
        sum_pseudo = sum_pseudo.scatter_add(0, index, pseudo_gt)
        count = count.scatter_add(0, index, torch.ones_like(pseudo_gt))

        valid_mask = count > 0
        if not torch.any(valid_mask): # 전부 False일 경우 (즉, 입력된 lidar point cloud가 없을 경우)
            conf_loss = torch.tensor(0.0, device=device)
            tb_dict.update({'conf_loss': conf_loss.item()})
            return conf_loss, tb_dict
        avg_pseudo = torch.zeros_like(sum_pseudo)
        avg_pseudo[valid_mask] = sum_pseudo[valid_mask] / count[valid_mask]

        active_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(1) # 0이 아닌 index 모음
        selected_pred = flat_conf[active_indices]
        selected_target = avg_pseudo[active_indices]

        climate_weight_tensor = climate_weight_tensor.to(device)
        b_indices = active_indices // (H * W)
        weights = climate_weight_tensor[b_indices]

        bce = F.binary_cross_entropy(selected_pred, selected_target, reduction='none')
        conf_loss = (bce * weights).mean()
        conf_loss = conf_loss * self.uem_loss_dict['conf_loss_weight']

        tb_dict.update({'conf_loss': conf_loss.item()})
        return conf_loss, tb_dict
        
    def get_loss(self, tb_dict=None):
        # 날시에 따른 climate weight list 구하기
        climate_weight_list = []
        for climate in self.climate_list:
            if climate in ("normal", "overcast"):
                climate_weight_list.append(self.uem_loss_dict['positive'])
            else:
                climate_weight_list.append(self.uem_loss_dict['negative'])
        climate_weight_tensor = torch.tensor(climate_weight_list, dtype=torch.float32)
        
        # loss 계산하기
        sim_loss, tb_dict = self.getSimilarityLoss(tb_dict, climate_weight_tensor)
        conf_loss, tb_dict = self.getConfidenceLoss(tb_dict, climate_weight_tensor)
        uem_loss = sim_loss + conf_loss

        return uem_loss, tb_dict
       
    
    def forward(self, batch_dict):
        """
        Input:
            - feature map: torch.Tensor [B, 256, 40, 80]
            - batch_dict: dict
        Output:
            - conf_map: torch.Tensor [B, 1, 40, 80]
        """
        
        # 1. Data preparation
        feature_map = batch_dict['feature_map'] # [B, 256, 40, 80]
        lidar_feature = batch_dict['uem_lidar_feature'] # [N, 256]
        lidar_indices = batch_dict['uem_lidar_indices'] # [N, 4] # (B, Z, Y, X)
        self.climate_list = batch_dict['climate_list'] # loss 계산할 때 필요
        
        
        # 2. Caculate each voxel's center location (unit: meters)
        b, z_indices, y_indices, x_indices = lidar_indices[:, 0].long(), lidar_indices[:, 1], lidar_indices[:, 2], lidar_indices[:, 3]
        x_center = self.x_min + (x_indices.float() + 0.5) * self.scaled_voxel_size # NOTE 0.5를 더하는 이유는 voxel의 중심을 구하고 싶어서
        y_center = self.y_min + (y_indices.float() + 0.5) * self.scaled_voxel_size
        z_center = self.z_min + (z_indices.float() + 0.5) * self.scaled_voxel_size
        coords_center = torch.stack([x_center, y_center, z_center], dim=-1)  # [N, 3]
        coords_center = torch.cat([coords_center, torch.ones((coords_center.shape[0], 1), device=coords_center.device)], dim=-1) # [N, 3] -> [N, 4]
        

        # 3. Determine the feature map coordinates corresponding to the voxel centers
        # 3.1. calculate intrinsic (lidar->feature map) 
        intrinsic = batch_dict['camera2image'].clone() # lss 모듈 안에서 또 다시 필요하므로 clone하기
        intrinsic = intrinsic[0, 0] # 어짜피 배치 안에 모든 샘플의 intrinsic은 동일하다.
        
        _, _, _, ori_h, ori_w = batch_dict['front0'].shape # [320, 640]
        _, _, feat_h, feat_w = feature_map.shape # [40, 80]
        
        scale_w = ori_w/feat_w # 8.0
        scale_h = ori_h/feat_h # 8.0
        
        intrinsic[0, 0] /= scale_w  # fx
        intrinsic[0, 2] /= scale_w  # cx
        intrinsic[1, 1] /= scale_h  # fy
        intrinsic[1, 2] /= scale_h  # cy
        
        ldr2cam = batch_dict['lidar2camera'][0, 0] # [4, 4]
        cam2feat = intrinsic # [4, 4]
        ldr2feat = torch.matmul(cam2feat, ldr2cam) # [4, 4]
        

        # 3.2. calculate 2D coordinates
        proj_coords = coords_center @ ldr2feat.T # [N, 4]
        eps = 1e-6  # divide by zero 방지
        # depth = proj_coords[:, 2]
        proj_coords = proj_coords[:, :2] / (proj_coords[:, 2:3] + eps)  # [N, 2]
        
        u = proj_coords[:, 0]
        v = proj_coords[:, 1]
        valid_mask = (u >= 0) & (u < feat_w) & (v >= 0) & (v < feat_h) # feature map 경계 검사
        proj_coords = proj_coords[valid_mask] # [N_new, 2]
        
        
        # 4. Sample image features correspoding to the voxel centers
        valid_b = b[valid_mask] # [N_new]
        x = proj_coords[:, 0].long()  # [N_new]
        y = proj_coords[:, 1].long()  # [N_new]
        sampled_feat = feature_map[valid_b, :, y, x] # [N_new, 256]
        self.x, self.y = x, y
        self.valid_b = valid_b
        
        
        # 5. Calculate similarity
        # 5.1. lidar feature랑 image feature 동일한 latent space로 보내기
        lidar_feature = lidar_feature[valid_mask] # [N, 256] -> [N_new, 256]
        proj_lidar_feat = self.lidar_projector(lidar_feature) # [N_new, 256] -> [N_new, 128]
        proj_image_feat = self.image_projector(sampled_feat) # [N_new, 256] -> [N_new, 128]
        
        cosine_sim = F.cosine_similarity(proj_lidar_feat, proj_image_feat, dim=-1) # [N_new]
        self.similarity = cosine_sim # loss 계산할 때 사용
        self.pseudo_gt = (cosine_sim+1.0)/2.0 # 0~1 사이로 정규화 # 나중에 loss 계산 때 사용될 예정
        

        # 6. Compute confidence map
        self.conf_map = self.conf_head(feature_map) # [B, 1, H, W] # loss 계산할 때 필요함

        batch_dict['conf_map'] = self.conf_map
        return batch_dict