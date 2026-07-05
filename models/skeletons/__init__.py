'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
* eddited: HJ Park
'''

from .rdr_base import RadarBase
from .ldr_base import LidarBase
from .pvrcnn_pp import PVRCNNPlusPlus
from .pp_radar import PointPillar_RADAR
from .pp_rlf import PointPillar_RLF
from .pp_rlf_dt import PointPillar_RLF_DT # KD
from .pp_crlf import PointPillar_CRLF                       # hj
from .pp_crlf_uem import PointPillar_CRLF_UEM               # hj
from .pp_crlf_uem_cnn import PointPillar_CRLF_UEM_CNN
from .voxel_lrf import Voxel_LRF                            # hj
from .voxel_l import Voxel_L                            # hj
from .voxel_clrf_uem import Voxel_CLRF_UEM                  #  hj
from .second_net import SECONDNet
from .pp_lidar import PointPillar
from .pp_inter import PointPillar_InterF
from .rtnh_lidar import RTNH_L
from .rtnh_lr import RTNH_LR
from .student import Student # KD (file removed)
from .student_bevdepth import StudentBEVDepth  # BEVDepth cam-only

# rubuttal
# from .crn import CRN
from .pp_crlf_crn import PointPillar_CRLF_CRN  # crn
from .pp_crlf_robu import PointPillar_CRLF_Robu  # crn


def build_skeleton(cfg, mode='train'):
    skeleton_name = cfg.MODEL.SKELETON
    # Skeletons that accept mode parameter (for test mode pretrained loading control)
    mode_aware_skeletons = ['PointPillar_RLF_DT']

    if skeleton_name in mode_aware_skeletons:
        return __all__[skeleton_name](cfg, mode=mode)
    else:
        return __all__[skeleton_name](cfg)

__all__ = {
    'RadarBase': RadarBase,
    'LidarBase': LidarBase,
    'PVRCNNPlusPlus': PVRCNNPlusPlus,
    'SECONDNet': SECONDNet,
    'PointPillar':PointPillar,
    'PointPillar_RADAR' : PointPillar_RADAR,
    'PointPillar_RLF': PointPillar_RLF,
    'PointPillar_RLF_DT': PointPillar_RLF_DT, # KD
    'PointPillar_CRLF': PointPillar_CRLF,                   # hj
    'PointPillar_CRLF_UEM' : PointPillar_CRLF_UEM,          # hj
    'PointPillar_CRLF_UEM_CNN' : PointPillar_CRLF_UEM_CNN,  # hj
    'Voxel_LRF': Voxel_LRF,                                 # hj
    'Voxel_L' : Voxel_L,                                    # hj
    'Voxel_CLRF_UEM' : Voxel_CLRF_UEM,                      # hj
    'PointPillar_InterF' : PointPillar_InterF,
    'RTNH_L' : RTNH_L,
    'RTNH_LR' : RTNH_LR,
    'Student': Student, # KD (file removed)
    'StudentBEVDepth': StudentBEVDepth,  # BEVDepth cam-only
    
    # rbuttal
    # 'CRN': CRN, # fail
    'PointPillar_CRLF_CRN':PointPillar_CRLF_CRN,
    'PointPillar_CRLF_Robu':PointPillar_CRLF_Robu,
}
