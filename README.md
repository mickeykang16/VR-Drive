<div align="center">
<img src="assets/logo.png" width="400">
<h1>VR-Drive</h1>
<h3>Viewpoint-Robust End-to-End Driving with Feed-Forward 3D Gaussian Splatting</h3>

[Hoonhee Cho](https://chohoonhee.github.io)<sup>1 &ast;</sup>, Jae-Young Kang<sup>1 &ast;</sup>, [Giwon Lee](https://giwonlee00.github.io)<sup>1 &ast;</sup>, Hyemin Yang<sup>1 &ast;</sup>, Heejun Park<sup>1</sup>, Seokwoo Jung<sup>2</sup>, [Kuk-Jin Yoon](https://vi.kaist.ac.kr)<sup>1 ✉</sup>

<sup>1</sup>KAIST, <sup>2</sup>42dot  
<sup>&ast;</sup>Equal contribution. <sup>✉</sup>Corresponding author.




[![VR-Drive Paper](https://img.shields.io/badge/Paper-VR--Drive-2b9348.svg?logo=readme)](https://openreview.net/pdf/a9479d1d90a2762ee248170d1f5844228e68a116.pdf)&nbsp;
[![VR-Drive Project Page](https://img.shields.io/badge/Project%20Page-VR--Drive-blue.svg?logo=githubpages)](https://vrdriveneurips.github.io/)&nbsp;




</div>

## News
* **` Jan. 4th, 2026`:** We release the initial version of our code for nuScenes, accompanied by comprehensive documentation and training/evaluation scripts.


## Table of Contents
- [Getting Started](#getting-started)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)


<div align="center"><b>Truncated Diffusion Policy.</b>
<img src="assets/truncated_diffusion_policy.png" />
<b>Pipeline of DiffusionDrive. DiffusionDrive is highly flexible to integrate with onboard sensor data and existing perception modules.</b>
<img src="assets/pipeline.png" />
</div>






## Getting Started

- [Getting started from nuScenes environment preparation](https://github.com/swc-17/SparseDrive/blob/main/docs/quick_start.md)
- [Training and Evaluation](docs/train_eval.md)


## Checkpoint
> Results on NAVSIM


| Method | Model Size | Backbone | PDMS | Weight Download |
| :---: | :---: | :---: | :---:  | :---: |
| DiffusionDrive | 60M | [ResNet-34](https://huggingface.co/timm/resnet34.a1_in1k) | [88.1](https://github.com/hustvl/DiffusionDrive/releases/download/DiffusionDrive_88p1_PDMS_Eval_file/diffusiondrive_88p1_PDMS.csv) | [Hugging Face](https://huggingface.co/hustvl/DiffusionDrive) |

> Results on nuScenes


| Method | Backbone | Weight | Log | L2 (m) 1s | L2 (m) 2s | L2 (m) 3s | L2 (m) Avg | Col. (%) 1s | Col. (%) 2s | Col. (%) 3s | Col. (%) Avg |
| :---: | :---: | :---: | :---: | :---: | :---: | :---:| :---: | :---: | :---: | :---: | :---: |
| DiffusionDrive | ResNet-50 | [HF](https://huggingface.co/hustvl/DiffusionDrive) | [Github](https://github.com/hustvl/DiffusionDrive/releases/download/DiffusionDrive_nuScenes/diffusiondrive_stage2.log.log) |  0.27 | 0.54  | 0.90 |0.57 | 0.03  | 0.05 | 0.16 | 0.08  |




## Acknowledgement
VR-Drive builds upon and is strongly influenced by several outstanding open-source projects, including [DiffusionDrive](https://github.com/hustvl/DiffusionDrive), [SparseDrive](https://github.com/swc-17/SparseDrive), [DrivingForward](https://github.com/fangzhou2000/DrivingForward).

## Citation
If you find DiffusionDrive is useful in your research or applications, please consider giving us a star 🌟 and citing it by the following BibTeX entry.

```bibtex
 @article{diffusiondrive,
  title={DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving},
  author={Bencheng Liao and Shaoyu Chen and Haoran Yin and Bo Jiang and Cheng Wang and Sixu Yan and Xinbang Zhang and Xiangyu Li and Ying Zhang and Qian Zhang and Xinggang Wang},
  journal={arXiv preprint arXiv:2411.15139},
  year={2024}
}
```
