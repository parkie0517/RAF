# RAF


## Abstract
Robust 3D object detection in adverse weather conditions is challenging due to sensor limitations. Although combining complementary modalities such as LiDAR and 4D RADAR has shown promise, the sparsity of these sensors becomes apparent in adverse weather with reduced reflections, leading to objects with few or no point cloud returns.
To address this limitation, camera sensors provide visual cues even when LiDAR and RADAR signals are weakened. However, cameras themselves are also vulnerable to adverse weather, where some regions become unreliable due to snow or rain occluding the camera lens. While some camera-fusion methods designed for adverse weather learn to weigh image regions via confidence maps, these maps receive no direct supervision and are learned solely through the detection loss.
We introduce **Reliability-Aware Fusion (RAF)**, which explicitly supervises per-pixel reliability estimation and provides a direct learning signal for identifying and suppressing unreliable visual cues. Our framework leverages pretrained LiDAR-RADAR networks, keeping their backbones frozen while training only the added camera branch, BEV fusion encoder, and detection head.
Extensive experiments on the K-Radar and VoD datasets demonstrate that integrating RAF consistently improves detection accuracy over LiDAR-RADAR baselines, achieving improvements of up to **+6.5 AP<sub>BEV</sub>** and **+7.4 AP<sub>3D</sub>**.

## Installation

For setup, please refer to the official [K-Radar](https://github.com/kaist-avelab/K-Radar) GitHub repository.

## Acknowledgements

We thank the authors of [K-Radar](https://github.com/kaist-avelab/K-Radar) and [L4DR](https://github.com/ylwhxht/L4DR) for releasing their datasets and codebases, which provide valuable resources for 3D object detection research in adverse weather.

## Citation

If you find our work useful, please cite:

```bibtex
@inproceedings{park2026raf,
  title={RAF: Reliability-Aware Fusion for Robust 3D Object Detection in Adverse Weather},
  author={Park, Heejun and Jeong, Jaeseok and Yoon, Kuk-Jin},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2026},
  note={To appear}
}
```
