
# PAGait: Region-Aware Modulation and Cross-Modal Consistency Enhancement for Multi-Modal Gait Recognition

[![Manuscript Status](https://img.shields.io/badge/Manuscript-Under%20Review-blue)](https://github.com/lpl8848/BAPnP_Solver)

This repository contains the MATLAB simulations and C++ implementation for the paper:

> **"BAPnP: A Barycentric Affine Invariant Linear Solver for Robust and Efficient Perspective-$n$-Point Pose Estimation"**  
> *Under review at **The Visual Computer** (Springer).*

# PAGait: Region-Aware Modulation and Cross-Modal Consistency Enhancement for Multi-Modal Gait Recognition

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)]()
[![Status](https://img.shields.io/badge/Status-Under%20Review-orange.svg)]()

Official implementation of the paper:

> **PAGait: Region-Aware Modulation and Cross-Modal Consistency Enhancement for Multi-Modal Gait Recognition**  
> Under review.

---

# 1. Introduction

This repository provides the official implementation of PAGait, a multi-modal gait recognition framework designed for robust fusion of silhouette sequences and human parsing sequences.

The repository contains:

- Training and testing code
- Network architecture implementation
- SAM module and consistency enhancement module
- Dataset preprocessing scripts
- Visualization tools
- Configuration files and checkpoints

Main features of PAGait:

- Structure-aware Adaptive Modulation (SAM)
- Cross-modal consistency enhancement
- Multi-modal feature fusion
- Robust gait representation learning under challenging conditions

---

# 2. Repository Structure

```text
PAGait/
├── configs/                    # Configuration files
├── datasets/                   # Dataset preprocessing scripts
├── modeling/                   # Model implementation
│   ├── backbone/               # Backbone networks
│   ├── modules/                # SAM and fusion modules
│   ├── losses/                 # Loss functions
│   └── heads/                  # Classification heads
├── tools/
│   ├── train.py                # Training script
│   ├── test.py                 # Evaluation script
│   └── visualization.py        # Visualization tools
├── output/                     # Logs and checkpoints
├── README.md
└── requirements.txt
```

---

# 3. Dataset Preparation

## 3.1 Supported Datasets

The framework supports the following public gait datasets:

- Gait3D
- CASIA-B
- OUMVLP
- GREW
- SUSTech1K

---

## 3.2 Dataset Directory Structure

Example:

```text
datasets/
├── Gait3D/
│   ├── silhouettes/
│   ├── parsing/
│   └── split/
├── CASIA-B/
└── OUMVLP/
```

Each dataset should contain:

- Silhouette sequences
- Human parsing sequences
- Training/testing split files

---

## 3.3 Human Parsing Generation

Human parsing sequences can be generated using off-the-shelf human parsing models.

Example workflow:

1. Extract RGB frames
2. Run human parsing model
3. Save parsing masks
4. Convert to training format

---

## 3.4 Data Preprocessing

The preprocessing scripts are located in:

```text
datasets/
```

Example:

```bash
python datasets/preprocess_gait3d.py
```

---

# 4. Installation

## 4.1 Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.7+ (recommended)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 5. Training

## 5.1 Single GPU Training

```bash
python tools/train.py \
    --config configs/gait3d.yaml
```

---

## 5.2 Multi-GPU Training

```bash
torchrun --nproc_per_node=4 tools/train.py \
    --config configs/gait3d.yaml
```

---

# 6. Evaluation

```bash
python tools/test.py \
    --config configs/gait3d.yaml \
    --checkpoint output/model_best.pth
```

---

# 7. Visualization

We provide visualization tools for:

- Attention maps
- Cross-modal response consistency
- Feature activation statistics
- Region-aware modulation analysis

Example:

```bash
python tools/visualization.py \
    --config configs/gait3d.yaml \
    --checkpoint output/model_best.pth
```

---

# 8. Experimental Results

## Gait3D

| Method | Rank-1 | mAP |
|---|---|---|
| Baseline | -- | -- |
| PAGait | -- | -- |

---

## CASIA-B

| Condition | NM | BG | CL |
|---|---|---|---|
| PAGait | -- | -- | -- |

---

# 9. Citation

```bibtex
@article{pagait2026,
  title={PAGait: Region-Aware Modulation and Cross-Modal Consistency Enhancement for Multi-Modal Gait Recognition},
  author={Author Name},
  journal={Under Review},
  year={2026}
}
```

---

# 10. Acknowledgements

This repository is built upon several excellent open-source gait recognition projects, including:

- GaitSet
- OpenGait
- GaitPart
- GaitGL

We sincerely thank the authors for their contributions.

---

# 11. License

This project is released under the MIT License.

---

# 12. Contact

For questions or collaborations, please open an issue or contact:

```text
your_email@example.com
```