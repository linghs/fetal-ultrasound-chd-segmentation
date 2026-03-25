# fetal-ultrasound-chd-segmentation（中文说明）

[English README](README.md)

## 论文题目

`\title{A View-Specific Dual-Task Framework for Fetal Heart UltraSound  Analysis}`

## 简介

本项目提供 FETUS 2026 胎儿心脏超声任务的基线脚本，包括：

- 分割训练：`FPN + mit_b5`（按视图独立训练）
- 分类训练：`Xception encoder + classification head`（按视图独立训练）
- 推理预测：支持分割、分类或两者同时输出

## 数据来源

官方页面：  
[FETUS 2026 Challenge - Dataset](http://119.29.231.17:90/data.html)

## 数据格式

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

## 快速使用

安装依赖（推荐）：

```bash
pip install -r requirements.txt
```

训练：

```bash
bash run_train.sh
```

预测：

```bash
bash run_predict.sh /path/to/val/image /path/to/output/preds checkpoints
```
