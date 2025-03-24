## [Project page (ICLR 2025)](https://modgs.github.io/) | [Paper (ArXiv)](https://arxiv.org/abs/2406.00434) | [OpenReview](https://openreview.net/forum?id=2prShxdLkX)
## TODO List:
- [x] Core techniques codes of proposed MoDGS.
- [ ] Training Codes(preprocessing and training).
- [ ] Rendering instruction 
- [ ] Self-collected data
- [ ] Codes Branch dealing with moving camera scenes.(e.g, Davis)

## 
Hi, this is Qingming. So sorry for the late release. I’m currently very busy(for my exam), but you can refer to our core codes located in `train_PointTrackGS.py`(3D-aware initialization and Ordinal depth loss) and `pointTrack.py`( 3D Flow pairs extraction). I will provide detailed processing instructions later when I have more time(one or two week). Thank you for your understanding! 


## Prepare your environment

1. prepare your env:

- Create the Conda virtual environment using the command below (Python and PyTorch versions higher than the specified ones should work fine):
```bash
conda create -n MoDGS python=3.7

pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
```

2. **Install dependencies**:
   - Follow [GaussianFlow](https://github.com/Zerg-Overmind/diff-gaussian-rasterization) to install the Gaussian rasterizer.
   - Follow the [official 3D-GS repo](https://github.com/graphdeco-inria/gaussian-splatting) to install Simple-KNN.
   - Install other packages using the command below:
   
```bash
pip install -r requirements.txt
```

---


## Preprocessing Dataset

### General Steps

There are several steps required to prepare your dataset for MoDGS training:
- Depth estimation
- Exhaustive flow estimation
- Scale alignment
- 3D Flow extraction

Detailed instructions for this part will be updated later.

<!-- To preprocess your dataset, use the script below: -->
you can check our code below. however, please note that I haven’t cleaned or fully tested it yet, so it may contain bugs. Thank you for your understanding!
```
./scripts/preprocess_selfmadeData.sh
```

**Preprocessed Dataset Example**: A sample dataset is provided here: [OneDrive](https://cuhko365-my.sharepoint.com/:f:/g/personal/224045018_link_cuhk_edu_cn/Er9SWOlAYx5EhK6qUPS8QsUBVPCpfEo7gqXt_6l1at68BA?e=dIKUDs)

<!-- After preprocessing, the file structure will look like this:

```shell
├── data
│   ├── D-NeRF 
│   │   ├── hook
│   │   ├── standup 
│   │   ├── ...
│   ├── NeRF-DS
│   │   ├── as
│   │   ├── basin
│   │   ├── ...
│   ├── HyperNeRF
│   │   ├── interp
│   │   ├── misc
│   │   ├── vrig
``` -->

---

## Training

### Configuration Setup: 3D-Aware Initialization Stage

### Stage 1: 3D-Aware Initialization

Run the following script:

```shell
./scripts/run_stage1.sh
```

### Stage 2: Optimizing the 4D Representation via Rendering and Ordinal Depth Losses

Run the following script:

```shell
./scripts/run_stage2.sh
```

---

## Rendering Scenes from Checkpoints

Once the training is complete, you can load the checkpoint and render scenes:

```shell
# Spiral cameras
./scripts/render_vis_cam.sh
```

---

## Metric Evaluation

To be updated.

---
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
## BibTex

```
@inproceedings{
liu2025modgs,
title={Mo{DGS}: Dynamic Gaussian Splatting from Casually-captured Monocular Videos with Depth Priors},
author={Qingming LIU and Yuan Liu and Jiepeng Wang and Xianqiang Lyu and Peng Wang and Wenping Wang and Junhui Hou},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=2prShxdLkX}
}
```

## Acknowledgements
And thanks to the authors of [Geowizard](https://github.com/fuxiao0719/GeoWizard), 
 [NSFF](https://www.cs.cornell.edu/~zl548/NSFF/), [3D Gaussians](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/), [GaussianFlow](https://github.com/Zerg-Overmind/diff-gaussian-rasterization), [D-GS](https://github.com/ingra14m/Deformable-3D-Gaussians)  and [Omnimotion](https://omnimotion.github.io/) for their excellent code, please consider also cite their paper:

```
@Article{kerbl3Dgaussians,
      author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {42},
      month        = {July},
      year         = {2023},
      url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```
```
@inproceedings{wang2023omnimotion,
  title     = {Tracking Everything Everywhere All at Once},
  author    = {Wang, Qianqian and Chang, Yen-Yu and Cai, Ruojin and Li, Zhengqi and Hariharan, Bharath and Holynski, Aleksander and Snavely, Noah},
  booktitle = {International Conference on Computer Vision},
  year      = {2023}
}
```

```
@article{yang2023deformable3dgs,
    title={Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction},
    author={Yang, Ziyi and Gao, Xinyu and Zhou, Wen and Jiao, Shaohui and Zhang, Yuqing and Jin, Xiaogang},
    journal={arXiv preprint arXiv:2309.13101},
    year={2023}
}
```
```
@article{gao2024gaussianflow,
  author    = {Quankai Gao and Qiangeng Xu and Zhe Cao and Ben Mildenhall and Wenchao Ma and Le Chen and Danhang Tang and Ulrich Neumann},
  title     = {GaussianFlow: Splatting Gaussian Dynamics for 4D Content Creation},
  journal   = {},
  year      = {2024},
}
```
```
@inproceedings{fu2024geowizard,
  title={GeoWizard: Unleashing the Diffusion Priors for 3D Geometry Estimation from a Single Image},
  author={Fu, Xiao and Yin, Wei and Hu, Mu and Wang, Kaixuan and Ma, Yuexin and Tan, Ping and Shen, Shaojie and Lin, Dahua and Long, Xiaoxiao},
  booktitle={ECCV},
  year={2024}
}
```
```
@InProceedings{li2020neural,
  title={Neural Scene Flow Fields for Space-Time View Synthesis of Dynamic Scenes},
  author={Li, Zhengqi and Niklaus, Simon and Snavely, Noah and Wang, Oliver},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2021}
}
```
