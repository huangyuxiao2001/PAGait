# PAGait: Parsing-Aware Multi-modal Gait Recognition
[![DOI](https://zenodo.org/badge/1238402679.svg)](https://doi.org/10.5281/zenodo.20257768)
[![Manuscript Status](https://img.shields.io/badge/Manuscript-Under%20Review-blue)](https://github.com/huangyuxiao2001/PAGait)
[![Code](https://img.shields.io/badge/Code-PyTorch-green)](https://pytorch.org/)
[![Framework](https://img.shields.io/badge/Framework-OpenGait-orange)](https://github.com/ShiqiYu/OpenGait)

This repository provides the official implementation of **PAGait**, a multi-modal gait recognition framework based on **silhouette sequences** and **human parsing sequences**.

PAGait aims to improve gait recognition robustness under complex scenarios such as clothing changes, occlusions, and background variations. It introduces two key modules:

- **Region-aware Adaptive Modulation (RAM)**, which models region-level differences by adaptively modulating parsing features according to different target body regions.
- **Cross-modal Consistency Enhancement (CCE)**, which enhances consistent cross-modal responses between silhouette and human parsing modalities while suppressing modality-dominant activations.

The implementation is built based on [OpenGait](https://github.com/ShiqiYu/OpenGait).

---

## 1. Introduction

Gait recognition aims to identify individuals according to their walking patterns. Existing silhouette-based methods have achieved promising performance in controlled scenarios. However, their robustness may degrade under clothing changes, occlusions, and complex real-world environments.

To address these challenges, PAGait introduces human parsing sequences as an additional modality to provide fine-grained body-part semantic information. Instead of simply concatenating silhouette and parsing features, PAGait explicitly models regional differences and cross-modal response consistency.

The overall pipeline consists of the following stages:

1. Preprocessing of silhouette and human parsing sequences;
2. First-stage feature encoding with two independent CNN branches;
3. Parallel feature modeling with RAM and CCE;
4. Second-stage feature encoding;
5. Feature post-processing with temporal pooling, HPP, fully connected layers, and BNNeck.

---

## 2. Main Modules

### 2.1 Region-aware Adaptive Modulation

The Region-aware Adaptive Modulation module uses human parsing masks to construct coarse body regions and adaptively modulates parsing features according to different target regions.

The coarse body regions are defined as follows:

| Region | Semantic Parts |
| :--- | :--- |
| Upper region | Head |
| Middle region | Torso and upper limbs |
| Lower region | Lower limbs and feet |

For each target region, RAM enhances target-region parsing features and suppresses non-target regions. The region-modulated parsing features are then used to generate region-conditioned fusion weights for silhouette and parsing features.

### 2.2 Cross-modal Consistency Enhancement

The Cross-modal Consistency Enhancement module models response consistency between silhouette and human parsing modalities. It emphasizes feature elements with balanced modality-normalized responses and suppresses modality-dominant activations, improving the stability and robustness of multi-modal feature fusion.

---

## 3. Repository Structure

```text
PAGait/
├── configs/
│   ├── PAGait_Gait3D.yaml
│   ├── PAGait_CCPG.yaml
│   ├── PAGait_MultiSubjectD.yaml
│   ├── PAGait_MultiSubjectP.yaml
│   └── PAGait_MultiSubjectS.yaml
├── datasets/
│   └── README.md
├── docs/
├── opengait/
│   ├── data/
│   ├── evaluation/
│   ├── modeling/
│   ├── utils/
│   └── main.py
├── output/
├── train.sh
├── test.sh
├── requirements.txt
└── README.md
```

---

# 4. Requirements

The code is implemented with PyTorch and OpenGait.

Recommended environment:

```text
Python 3.8
PyTorch 1.11.0
CUDA 11.3
torchvision 0.12.0
torchaudio 0.11.0
```

We provide the conda environment configuration file:

```text
opengait.yaml
```

Create the environment using:

```bash
conda env create -f opengait.yaml
```

Activate the environment:

```bash
conda activate gait
```

You can also follow the environment setup instructions of [OpenGait](https://github.com/ShiqiYu/OpenGait).

## 5. Data Preparation

The original datasets are **not redistributed** in this repository due to license and redistribution restrictions. Please download the datasets from the official providers.

### 5.1 Dataset Links and References

| Dataset | Official Link | Reference DOI |
| :--- | :--- | :--- |
| Gait3D | https://gait3d.github.io | `10.1109/CVPR52688.2022.01959` |
| CCPG | https://github.com/BNU-IVC/CCPG | `10.1109/CVPR52729.2023.01328` |
| MultiSubjects | https://huggingface.co/datasets/Henu-Software/Henu-MultiSubjects | `10.1016/j.cviu.2024.104193` |


Please cite the corresponding dataset papers if you use these datasets.

### 5.2 Human Parsing Sequences

Human parsing sequences are generated frame by frame from the original RGB videos or image sequences. In this project, human parsing maps are used as an additional modality together with silhouette sequences.

The generated parsing maps should be aligned with the corresponding silhouette sequences.

The expected input resolution is:

```text
64 × 44
```

### 5.3 Expected Data Format

Please organize the data following the OpenGait format. A typical structure is shown below:

```text
datasets/
├── Gait3D/
│   ├── silhouettes/
│   └── parsings/
├── CCPG/
│   ├── silhouettes/
│   └── parsings/
└── MultiSubjects-Gait/
    ├── silhouettes/
    └── parsings/
```

The exact data path should be configured in the corresponding YAML configuration file under the `configs/` directory.

---

## 6. Training

### 6.1 Train on Gait3D

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --nproc_per_node=1 \
    opengait/main.py \
    --cfgs ./configs/PAGait_Gait3D.yaml \
    --phase train \
    --log_to_file
```

### 6.2 Train on CCPG

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --nproc_per_node=1 \
    opengait/main.py \
    --cfgs ./configs/PAGait_CCPG.yaml \
    --phase train \
    --log_to_file
```

### 6.3 Train on MultiSubjects-Gait

For the MultiSubjects-Gait dataset, we evaluate the model on three subsets: D, P, and S.

Train on the D subset:

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --nproc_per_node=1 \
    opengait/main.py \
    --cfgs ./configs/PAGait_MultiSubjectD.yaml \
    --phase train \
    --log_to_file
```

Train on the P subset:

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --nproc_per_node=1 \
    opengait/main.py \
    --cfgs ./configs/PAGait_MultiSubjectP.yaml \
    --phase train \
    --log_to_file
```

Train on the S subset:

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --nproc_per_node=1 \
    opengait/main.py \
    --cfgs ./configs/PAGait_MultiSubjectS.yaml \
    --phase train \
    --log_to_file
```

---

## 7. Evaluation

### 7.1 Evaluate on Gait3D

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --nproc_per_node=1 \
    opengait/main.py \
    --cfgs ./configs/PAGait_Gait3D.yaml \
    --phase test \
    --log_to_file
```

### 7.2 Evaluate on CCPG

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --nproc_per_node=1 \
    opengait/main.py \
    --cfgs ./configs/PAGait_CCPG.yaml \
    --phase test \
    --log_to_file
```

### 7.3 Evaluate on MultiSubjects-Gait

Evaluate on the D subset:

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --nproc_per_node=1 \
    opengait/main.py \
    --cfgs ./configs/PAGait_MultiSubjectD.yaml \
    --phase test \
    --log_to_file
```

Evaluate on the P subset:

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --nproc_per_node=1 \
    opengait/main.py \
    --cfgs ./configs/PAGait_MultiSubjectP.yaml \
    --phase test \
    --log_to_file
```

Evaluate on the S subset:

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --nproc_per_node=1 \
    opengait/main.py \
    --cfgs ./configs/PAGait_MultiSubjectS.yaml \
    --phase test \
    --log_to_file
```

---



## 8. Citation

If you find this repository useful for your research, please cite our paper:

```bibtex
@article{han2026pagait,
  title={PAGait: Region-Aware Modulation and Cross-Modal Consistency Enhancement for Multi-Modal Gait Recognition},
  author={Han, Zhijie and Huang, Yuxiao and Wang, Yalu and Zhao, Yanxiang},
  journal={Under review},
  year={2026}
}
```

---

## 9. Acknowledgements

This project is built upon [OpenGait](https://github.com/ShiqiYu/OpenGait).

We thank the authors of OpenGait, DeepGaitV2, CDGNet, Gait3D, CCPG, MultiSubjects, and other related gait recognition works for their valuable contributions to the community.

---

## 10. License

This project is released for academic research purposes only.

Please follow the licenses of the original datasets and the OpenGait framework. The datasets used in this project are not redistributed in this repository due to license and redistribution restrictions.

---

## 11. Contact

For questions, please contact:

```text
Yuxiao Huang
Email: 15515952990@163.com
GitHub: https://github.com/huangyuxiao2001
```