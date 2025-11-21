<div align="center">

# 🫀 CoroSAM

### *Interactive Coronary Artery Segmentation with Segment Anything Model*

[📄 Paper](https://www.sciencedirect.com/science/article/pii/S0169260725005887) • [🚀 Getting Started](#installation) • [💾 Pretrained Models](#pretrained-checkpoints) • [🎮 GUI](#gui-application)

---

**CoroSAM** is a deep learning framework for **interactive coronary artery segmentation** in coronary angiograms, built on a computationally efficient SAM-based architecture with custom convolutional adapters.

*This is the official implementation of the paper published in Computer Methods and Programs in Biomedicine.*

</div>

---

## 📑 Table of Contents

- [Installation](#installation)
- [Pretrained checkpoints](#pretrained-checkpoints)
- [ARCADE dataset](#arcade-dataset-preparation)
- [Preprocessing](#preprocessing)
- [Training](#training)
- [Testing](#testing)
- [Testing on different datasets](#testing-on-different-datasets)
- [GUI application](#gui-application)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

---

## Installation

### 1. Create virtual environment & install PyTorch

First, install PyTorch following the [official installation guide](https://pytorch.org/get-started/locally/).

**Recommended version:** `torch==2.6.0+cu124`

### 2. Clone repository

```bash
git clone https://github.com/mife-git/corosam.git
cd corosam
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Pretrained checkpoints

### External

Download and place in `checkpoints/Pretrained/`:

| Model | Source | Path |
|-------|--------|------|
| **LiteMedSAM** | [GitHub](https://github.com/bowang-lab/MedSAM/tree/LiteMedSAM) | `checkpoints/Pretrained/lite_medsam.pth` |
| **SAMMed2D** | [GitHub](https://github.com/OpenGVLab/SAM-Med2D) | `checkpoints/Pretrained/sam-med2d_b.pth` |

### CoroSAM

Our pretrained CoroSAM model trained on ARCADE is available here:

📥 **[Download CoroSAM Checkpoint](https://drive.google.com/file/d/1wgZ4ZojzC9bea92EVavQDGaH8ijGcawC/view?usp=sharing)**

Save as: `checkpoints/CoroSAM/CoroSAM_Final_Training.pt`

---

## ARCADE dataset preparation

### Download

1. Download ARCADE from [Zenodo](https://zenodo.org/records/10390295)
2. Extract to your workspace
3. Use only the `syntax` subset for this project

```
arcade/
└── syntax/
    ├── train/
    ├── val/
    └── test/
```

---

## Preprocessing

Transform ARCADE COCO annotations into training-ready format.

### 1. Configure paths

Edit `preprocessing_config.yaml`:

```yaml
dataset_root: "C:/path/to/arcade/syntax"
seed: 2025
```

### 2. Run preprocessing pipeline

```bash
# Step 1: Convert COCO to binary masks
python preprocessing/convert_coco_to_binary_masks.py

# Step 2: Merge train+val and apply augmentation
python preprocessing/data_augmentation.py

# Step 3: Create k-fold splits
python preprocessing/split_dataset.py
```

**Output structure:**
```
syntax/
├── train/              # Original train set
├── val/                # Original val set
├── test/               # Test set
├── train_all/          # Merged train+val
│   ├── images/
│   ├── annotations/
│   ├── images_augmented/
│   └── annotations_augmented/
└── kf_split/           # 5-fold cross-validation
    ├── set1/
    ├── set2/
    └── ...
```

---

## Training

Train CoroSAM on your data with flexible configurations.

### Configuration

Edit `train_config.yaml`:

```yaml
# Dataset
dataset_root: "C:/path/to/arcade/syntax"
k_fold_path: "C:/path/to/arcade/syntax/kf_split"

# Model
model_name: "LiteMedSAM"
exp_name: "CoroSAM_Training"

# Adapters
use_adapters: true
use_conv_adapters: true
channel_reduction: 0.25

# Training
n_folds: 5        # 5-fold CV or set to 1 for single run
epochs: 25
batch_size: 4
lr: 0.0005

# Logging
use_wandb: true
proj_name: "CoroSAM"
```

### Run training

**K-fold cross-validation:**
```bash
python training/train.py --config train_config.yaml
```

**Single training run:**
```yaml
n_folds: 1
train_path: "C:/path/to/arcade/syntax/train_all"
val_path: "C:/path/to/arcade/syntax/test"
```

---

## Testing

Comprehensive evaluation with detailed metrics and visualizations.

### Configure testing

Edit `test_config.yaml`:

```yaml
# Model
model_name: "LiteMedSAM"
checkpoint: "checkpoints/CoroSAM/corosam_pretrained.pth"

# Dataset
test_path: "C:/path/to/arcade/syntax/test"
results_path: "results/CoroSAM_ARCADE_Test"

# Options
save_predictions: true  # Save visualization images
```

### Run testing

```bash
python testing/test.py --config test_config.yaml
```

---

## Testing on different datasets

CoroSAM can be evaluated on **any custom dataset**!

### Requirements

Your dataset must follow the ARCADE preprocessing output structure:

```
dataset_name/
└── test/  (or any folder name)
    ├── images/
    │   ├── dataset_name_1.png
    │   ├── dataset_name_2.png
    │   └── ...
    └── annotations/
        ├── dataset_name_1_gt.png
        ├── dataset_name_2_gt.png
        └── ...
```

### Quick test

```yaml
# test_config.yaml
test_path: "path/to/your_dataset/test"
checkpoint: "checkpoints/CoroSAM/corosam_pretrained.pth"
```

```bash
python testing/test.py --config test_config.yaml
```

---

## GUI application

Interactive segmentation with a user-friendly interface.

### Launch GUI

```bash
python gui/gui_corosam.py
```

---

## Citation

If you find CoroSAM useful in your research, please cite our paper:

```bibtex
@article{corosam2025,
  title={CoroSAM: adaptation of the Segment Anything Model for interactive segmentation in Coronary angiograms},
  journal={Computer Methods and Programs in Biomedicine},
  year={2025},
  publisher={Elsevier},
  doi={10.1016/j.cmpb.2025.108587},
  url={https://www.sciencedirect.com/science/article/pii/S0169260725005887}
}
```

---

## Acknowledgments

This project builds upon excellent open-source work:

- **Segment Anything Model (SAM)**: [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)
- **MedSAM**: [bowang-lab/MedSAM](https://github.com/bowang-lab/MedSAM)
- **SAM-Med2D**: [OpenGVLab/SAM-Med2D](https://github.com/OpenGVLab/SAM-Med2D)
