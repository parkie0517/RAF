from .pillars_backbone import PillarsBackbone
from .resnet_wrapper import ResNetFPN
from .base_bev_backbone_rt import BaseBEVBackbone_MGF_RT # KD
from .base_bev_backbone import BaseBEVBackbone, BaseBEVBackbone_MGF, BaseBEVBackbone_MF, BaseBEVBackbone_MF_cam, BaseBEVBackbone_MF_ASF, BaseBEVBackbone_MF_cam_samfusion
from .base_bev_backbone_l4dr import BaseBEVBackbone_MF_L4DR
from .voxel_to_bev import VOXEL_TO_BEV # hj

# rebuttal
from .base_bev_backbone_crn import BaseBEVBackbone_MF_RC  # fail
from .base_bev_backbone_crn_clr import BaseBEVBackbone_CRN_CLR
from .base_bev_backbone_crn_cr import BaseBEVBackbone_CRN_CR
from .base_bev_backbone_robu_cr import BaseBEVBackbone_Robu_CR
from .base_bev_backbone_robu_clr import BaseBEVBackbone_Robu_CLR
from .base_bev_backbone_robu_clr_scalar import BaseBEVBackbone_Robu_CLR_scalar


__all__ = {
    'PillarsBackbone': PillarsBackbone,
    'ResNetFPN': ResNetFPN,
    'BaseBEVBackbone': BaseBEVBackbone,
    'BaseBEVBackbone_MGF_RT' : BaseBEVBackbone_MGF_RT, # KD
    'BaseBEVBackbone_MGF' : BaseBEVBackbone_MGF,
    'BaseBEVBackbone_MF' : BaseBEVBackbone_MF,
    'BaseBEVBackbone_MF_cam' : BaseBEVBackbone_MF_cam,  # hj
    'BaseBEVBackbone_MF_ASF' : BaseBEVBackbone_MF_ASF,
    'VOXEL_TO_BEV' : VOXEL_TO_BEV, # hj
    'BaseBEVBackbone_MF_L4DR':BaseBEVBackbone_MF_L4DR,
    
    # rebuttal
    'BaseBEVBackbone_MF_RC': BaseBEVBackbone_MF_RC, # fail
    'BaseBEVBackbone_CRN_CLR': BaseBEVBackbone_CRN_CLR, # crn clr fusion
    'BaseBEVBackbone_CRN_CR': BaseBEVBackbone_CRN_CR,  # crn cr fusion
    'BaseBEVBackbone_Robu_CR':BaseBEVBackbone_Robu_CR, # robu cr fusion
    'BaseBEVBackbone_Robu_CLR':BaseBEVBackbone_Robu_CLR, # robu clr fusion
    'BaseBEVBackbone_Robu_CLR_scalar':BaseBEVBackbone_Robu_CLR_scalar,
    
    # eccv
    'BaseBEVBackbone_MF_cam_samfusion':BaseBEVBackbone_MF_cam_samfusion,
}
