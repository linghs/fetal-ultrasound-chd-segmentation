# fetal-ultrasound-chd-segmentation

[中文说明 (Chinese README)](README_CN.md)

## Paper Title

`A View-Specific Dual-Task Framework for Fetal Heart UltraSound  Analysis`

## Overview

This repository provides baseline scripts for the FETUS 2026 fetal heart ultrasound challenge:

- Segmentation training: `FPN + mit_b5` (trained independently for each view)
- Classification training: `Xception encoder + classification head` (trained independently for each view)
- Inference: segmentation only, classification only, or both

## Dataset Source

Official challenge dataset page:  
[FETUS 2026 Challenge - Dataset](http://119.29.231.17:90/data.html)

## Dataset Structure

According to the official format, the training set can be organized as:

```text
train/
├── image/   # trainning images' file             001.h5/
│   ├── 001.h5                                       └──image # size: [512, 512, 3]
│   ├── 002.h5                                          # one of the four views
│   ├── 003.h5
│   └── ...
│
├── label/   # trainning labels' file         001_label.h5/
│   ├── 001_label.h5                                 ├──mask # size: [512, 512]
│   ├── 002_label.h5                                 └──classification label
│   ├── 003_label.h5                                    # [0,1,0,0,0,1,0]
│   └── ...                                             # 0: normal
│                                                       # 1: abnormal
└── train.txt  # trianning file's list
```

### Required H5 fields used by scripts

- Image file (e.g., `001.h5`) should include:
  - `image`: `shape=[512, 512, 3]`
  - `view`: one of `1/2/3/4`
- Label file (e.g., `001_label.h5`) should include:
  - `mask`: segmentation mask
  - `label`: 7-dim multi-label CHD vector

## Dependencies

Recommended: Python 3.10+

Main packages:

- `torch`
- `numpy`
- `h5py`
- `tqdm`
- `scipy`
- `scikit-learn`
- `segmentation-models-pytorch`

Install from `requirements.txt` (recommended):

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install torch numpy h5py tqdm scipy scikit-learn segmentation-models-pytorch
```

## Training

### Run Python scripts directly

Segmentation (4 views):

```bash
python train_segmentation.py \
  --images_dir /path/to/train/image \
  --labels_dir /path/to/train/label \
  --save_dir checkpoints \
  --epochs 100 \
  --batch_size 8 \
  --lr 5e-5 \
  --views 1 2 3 4
```

Classification (4 views):

```bash
python train_classification.py \
  --images_dir /path/to/train/image \
  --labels_dir /path/to/train/label \
  --save_dir checkpoints \
  --epochs 100 \
  --batch_size 16 \
  --lr 1e-4 \
  --views 1 2 3 4
```

### Run one-click training script

```bash
bash run_train.sh
```

> Note: `run_train.sh` currently defaults to `/root/autodl-tmp/_FETUS_data/train/images` and `/root/autodl-tmp/_FETUS_data/train/labels`.  
> If your folders are `train/image` and `train/label`, update `IMAGES_DIR` and `LABELS_DIR` in `run_train.sh`.

## Inference

Run prediction script directly:

```bash
python predict.py \
  --images_dir /path/to/val/image \
  --output_dir /path/to/output/preds \
  --checkpoints_dir checkpoints \
  --task both \
  --views 1 2 3 4 \
  --batch_size 1
```

Or use the wrapper script:

```bash
bash run_predict.sh /path/to/val/image /path/to/output/preds checkpoints
```

## Output Format

Each predicted sample is saved as one `h5` file (e.g., `123.h5`) and may contain:

- `mask`: segmentation output (`uint8`)
- `label`: 7-dim classification output (`uint8`, threshold default `0.5`)

`--task` options:

- `seg`: segmentation only
- `cls`: classification only
- `both`: segmentation + classification

## View and Class Settings

- View IDs: `1=4CH, 2=LVOT, 3=RVOT, 4=3VT`
- View-specific segmentation classes: `VIEW_CLASSES` in `train_segmentation.py`
- View-specific CHD subsets for classification: `CLS_ALLOWED` in `train_classification.py`

## Repository Files

- `train_segmentation.py`: segmentation training
- `train_classification.py`: classification training
- `predict.py`: inference and output writing
- `run_train.sh`: full training pipeline
- `run_predict.sh`: prediction wrapper

