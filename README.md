
# PAGait: Region-Aware Modulation and Cross-Modal Consistency Enhancement for Multi-Modal Gait Recognition

[![Python](https://img.shields.io/badge/Python-3.8-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-1.11.0-ee4c2c.svg)]()
[![CUDA](https://img.shields.io/badge/CUDA-11.3-76B900.svg)]()
[![Status](https://img.shields.io/badge/Status-Under%20Review-orange.svg)]()

Official implementation of PAGait.

> **PAGait: Region-Aware Modulation and Cross-Modal Consistency Enhancement for Multi-Modal Gait Recognition**  
> Under review.

---

# 1. Introduction

This repository provides the official implementation of PAGait, a multi-modal gait recognition framework designed for robust fusion of silhouette sequences and human parsing sequences.

The repository contains:

- Training and testing code
- Network architecture implementation
- RAM module and CCE module
- Dataset preprocessing scripts
- Configuration files 

Main features of PAGait:

- Region-aware Adaptive Modulation (SAM)
- Cross-modal consistency enhancement (CCE)
- Robust gait representation learning under challenging conditions

---

# 2. Repository Structure

```text
PAGait/
├── configs/                    # Training configuration files
├── datasets/                   # Dataset split files and preprocessing scripts
├── opengait/                   # Main framework
│   ├── data/                   # Data loading and dataset processing
│   ├── evaluation/             # Evaluation and testing
│   ├── modeling/               # Network architectures and modules
│   ├── utils/                  # Utility functions
│   └── main.py                 # Program entry
├── README.md
├── train.sh                    # Training script
└── test.sh                     # Evaluation script
````

---

# 3. Dataset Preparation

## 3.1 Supported Datasets

The framework currently supports the following public gait datasets:

| Dataset | Official Link |
|---|---|
| Gait3D | https://gait3d.github.io |
| CCPG | https://github.com/BNU-IVC/CCPG |
| MultiSubjects-Gait | https://huggingface.co/datasets/Henu-Software/Henu-MultiSubjects |

Please download the datasets from the official project pages and organize them according to the required directory structure.

---

## 3.2 Dataset Split Files and Preprocessing Scripts
- The official training/testing split files for **Gait3D** and **CCPG** are provided by the original dataset authors.
- The split files for **MultiSubjects** are provided in this repository.

Example:

```text
datasets/
├── Gait3D/
│   └── Gait3D.json
├── CCPG/
│   └── CCPG.json
├── MultiSubjects-Gait/
│   ├── MultiSubjects-D.json
│   ├── MultiSubjects-P.json
│   └── MultiSubjects-S.json
└── ln_sil_parsing.py
```

Each dataset should contain the corresponding training/testing split files.

## 3.3 Human Parsing Generation

For the Gait3D dataset, the official human parsing data is publicly available.

For the CCPG and MultiSubjects-Gait datasets, human parsing sequences can be generated using the official CDGNet parsing model:

- CDGNet-Parsing: https://github.com/Gait3D/CDGNet-Parsing

Please follow the instructions in the official repository to:

1. Extract RGB frames from gait sequences
2. Run the CDGNet parsing model
3. Generate human parsing masks
4. Convert parsing results into the required training format

After generation, please keep the same directory structure as the original dataset.


## 3.4 Data Preprocessing

After preparing the silhouette data and human parsing data, you need to link the two modalities into a unified directory structure for training.

The preprocessing script is located in:

```text
datasets/ln_sil_parsing.py

python datasets/ln_sil_parsing.py \
    --parsing_data_path /path/to/parsing_data \
    --silhouette_data_path /path/to/silhouette_data \
    --output_path /path/to/output_data

# 4. Installation

## 4.1 Requirements

The project is tested with the following environment:

| Package | Version |
|---|---|
| Python | 3.8 |
| PyTorch | 1.11.0 |
| CUDA | 11.3 |
| torchvision | 0.12.0 |
| torchaudio | 0.11.0 |

---

## 4.2 Install Dependencies

We provide the complete conda environment configuration file:

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


# 5. Training

Run the training script using:

```bash
MASTER_ADDR=localhost \
MASTER_PORT=12355 \
RANK=x \
WORLD_SIZE=x \
CUDA_VISIBLE_DEVICES=x \
python -m torch.distributed.launch \
    --nproc_per_node=x \
    PAGait/main.py \
    --cfgs ./configs/PAGait/Dataset-name.yaml \
    --phase train \
    --log_to_file
```

### Arguments

| Argument | Description |
|---|---|
| `RANK` | Rank of current node |
| `WORLD_SIZE` | Total number of nodes |
| `CUDA_VISIBLE_DEVICES` | GPU IDs used for training |
| `--nproc_per_node` | Number of GPUs used for training |
| `--cfgs` | Path to configuration file |
| `--phase train` | Training mode |

Replace `Dataset-name.yaml` with the corresponding dataset configuration file, for example:

```text
configs/PAGait_Gait3D.yaml
configs/PAGait_CCPG.yaml
configs/PAGait_MultiSubjectD.yaml
configs/PAGait_MultiSubjectP.yaml
configs/PAGait_MultiSubjectS.yaml
```

# 6. Evaluation

Run the evaluation script using:

```bash
MASTER_ADDR=localhost \
MASTER_PORT=12355 \
RANK=x \
WORLD_SIZE=x \
CUDA_VISIBLE_DEVICES=x,x \
python -m torch.distributed.launch \
    --nproc_per_node=x \
    PAGait/main.py \
    --cfgs ./configs/PAGait/Dataset-name.yaml \
    --phase test \
    --log_to_file
```

### Arguments

| Argument | Description |
|---|---|
| `RANK` | Rank of current node |
| `WORLD_SIZE` | Total number of nodes |
| `CUDA_VISIBLE_DEVICES` | GPU IDs used for evaluation |
| `--nproc_per_node` | Number of GPUs used for evaluation |
| `--cfgs` | Path to configuration file |
| `--phase test` | Evaluation mode |
| `--log_to_file` | Save logs to file |

Replace `Dataset-name.yaml` with the corresponding dataset configuration file.

# 9. Citation

If you find this repository useful for your research, please cite:

```bibtex
@article{pagait2026,
  title={PAGait: Region-Aware Modulation and Cross-Modal Consistency Enhancement for Multi-Modal Gait Recognition},
  author={Han, Zhijie and Huang, Yuxiao and Wang, Yalu and Zhao, Yanxiang and Guo, Li},
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

We sincerely thank the authors for their valuable contributions to the gait recognition community.

---

# 11. Contact

For questions, discussions, or collaborations, please open an issue or contact:

```text
15515952990@163.com
```