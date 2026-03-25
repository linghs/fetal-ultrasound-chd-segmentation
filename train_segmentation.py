#!/usr/bin/env python3
"""
Fetus2026 Challenge - 分割训练脚本
模型: FPN + MIT-B5
每个视图独立训练
"""

import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from pathlib import Path
from tqdm import tqdm
import argparse
import json
from scipy import ndimage

# ============================================================================
# 配置
# ============================================================================

VIEW_CLASSES = {
    1: [0, 1, 2, 3, 4, 5, 6, 7],      # 4CH: 8类
    2: [0, 1, 2, 4, 8],               # LVOT: 5类
    3: [0, 6, 8, 9, 10, 11, 12],      # RVOT: 7类
    4: [0, 9, 12, 13, 14],            # 3VT: 5类
}

VIEW_NAMES = {1: "4CH", 2: "LVOT", 3: "RVOT", 4: "3VT"}

ANATOMY_NAMES = {
    0: "Background", 1: "Left Atrium", 2: "Left Ventricle",
    3: "Right Atrium", 4: "Right Ventricle", 5: "Heart (Whole)",
    6: "Descending Aorta", 7: "Thoracic Cavity", 8: "Ascending Aorta",
    9: "Main Pulmonary Artery", 10: "Left Pulmonary Artery",
    11: "Right Pulmonary Artery", 12: "Superior Vena Cava",
    13: "Aortic Transverse Arch", 14: "Trachea",
}


# ============================================================================
# 数据集
# ============================================================================

class FetusSegmentationDataset(Dataset):
    def __init__(self, images_dir: str, labels_dir: str, view_id: int):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.view_id = view_id
        self.original_classes = VIEW_CLASSES[view_id]
        self.num_classes = len(self.original_classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.original_classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}
        self.samples = self._collect_samples()
        print(f"View {view_id} ({VIEW_NAMES[view_id]}): {len(self.samples)} samples, {self.num_classes} classes")

    def _collect_samples(self):
        samples = []
        for label_path in self.labels_dir.glob("*_label.h5"):
            img_id = int(label_path.stem.replace("_label", ""))
            img_path = self.images_dir / f"{img_id}.h5"
            if not img_path.exists():
                continue
            try:
                with h5py.File(img_path, 'r') as f:
                    view = int(np.array(f['view']).flat[0])
                    if view == self.view_id:
                        samples.append({
                            'img_id': img_id,
                            'img_path': str(img_path),
                            'label_path': str(label_path),
                        })
            except Exception as e:
                print(f"Error reading {img_path}: {e}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        with h5py.File(sample['img_path'], 'r') as f:
            image = f['image'][()]
        with h5py.File(sample['label_path'], 'r') as f:
            mask = f['mask'][()]

        mapped_mask = np.zeros_like(mask, dtype=np.int64)
        for orig_class, new_idx in self.class_to_idx.items():
            mapped_mask[mask == orig_class] = new_idx

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mapped_mask).long()
        return {'image': image, 'mask': mask, 'img_id': sample['img_id']}


# ============================================================================
# 损失函数 & 评估指标
# ============================================================================

class DiceLoss(nn.Module):
    def __init__(self, num_classes: int, smooth: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.softmax(pred, dim=1)
        target_one_hot = torch.zeros_like(pred)
        target_one_hot.scatter_(1, target.unsqueeze(1), 1)
        dice_scores = []
        for c in range(1, self.num_classes):
            pred_c = pred[:, c]
            target_c = target_one_hot[:, c]
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            dice_scores.append((2.0 * intersection + self.smooth) / (union + self.smooth))
        if dice_scores:
            return 1.0 - torch.stack(dice_scores).mean()
        return torch.tensor(0.0, device=pred.device)


class CombinedLoss(nn.Module):
    def __init__(self, num_classes: int, ce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(num_classes)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        return self.ce_weight * self.ce_loss(pred, target) + self.dice_weight * self.dice_loss(pred, target)


def compute_dice_score(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> dict:
    pred_classes = torch.argmax(pred, dim=1)
    dice_scores = {}
    for c in range(num_classes):
        pred_c = (pred_classes == c).float()
        target_c = (target == c).float()
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        dice_scores[c] = ((2.0 * intersection) / union).item() if union > 0 else (1.0 if intersection == 0 else 0.0)
    fg_dice = [v for k, v in dice_scores.items() if k > 0]
    dice_scores['mean'] = np.mean(fg_dice) if fg_dice else 0.0
    return dice_scores


def nsd_binary(pred_bin: np.ndarray, gt_bin: np.ndarray, tol: float = 2.0) -> float:
    if pred_bin.sum() == 0 and gt_bin.sum() == 0:
        return 1.0
    if pred_bin.sum() == 0 or gt_bin.sum() == 0:
        return 0.0
    pred_boundary = pred_bin ^ ndimage.binary_erosion(pred_bin)
    gt_boundary = gt_bin ^ ndimage.binary_erosion(gt_bin)
    if pred_boundary.sum() == 0:
        pred_boundary = pred_bin
    if gt_boundary.sum() == 0:
        gt_boundary = gt_bin
    gt_dist = ndimage.distance_transform_edt(~gt_boundary)
    pred_dist = ndimage.distance_transform_edt(~pred_boundary)
    pred_to_gt = gt_dist[pred_boundary.astype(bool)]
    gt_to_pred = pred_dist[gt_boundary.astype(bool)]
    total = len(pred_to_gt) + len(gt_to_pred)
    if total == 0:
        return 1.0
    return float(((pred_to_gt <= tol).sum() + (gt_to_pred <= tol).sum()) / total)


# ============================================================================
# 训练 & 验证
# ============================================================================

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, num_classes):
    model.train()
    total_loss = 0
    all_dice = []
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        with torch.no_grad():
            dice = compute_dice_score(outputs, masks, num_classes)
            all_dice.append(dice['mean'])
        pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{dice['mean']:.4f}")
    return total_loss / len(dataloader), np.mean(all_dice)


def validate(model, dataloader, criterion, device, epoch, num_classes):
    model.eval()
    total_loss = 0
    C = num_classes
    dice_sum = np.zeros(C - 1, dtype=np.float64)
    nsd_sum = np.zeros(C - 1, dtype=np.float64)
    cnt = np.zeros(C - 1, dtype=np.int64)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
    with torch.no_grad():
        for batch in pbar:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            pred_mask = outputs.argmax(dim=1)
            for b in range(images.shape[0]):
                pm = pred_mask[b].cpu().numpy().astype(np.int32)
                gt = masks[b].cpu().numpy().astype(np.int32)
                for cls in range(1, C):
                    pred_bin = (pm == cls)
                    gt_bin = (gt == cls)
                    if pred_bin.sum() + gt_bin.sum() == 0:
                        continue
                    inter = (pred_bin & gt_bin).sum()
                    union = pred_bin.sum() + gt_bin.sum()
                    dice_sum[cls - 1] += (2.0 * inter) / (union + 1e-8)
                    nsd_sum[cls - 1] += nsd_binary(pred_bin, gt_bin)
                    cnt[cls - 1] += 1
            valid = cnt > 0
            cur_dice = float(100.0 * dice_sum[valid].sum() / max(cnt[valid].sum(), 1))
            cur_nsd = float(100.0 * nsd_sum[valid].sum() / max(cnt[valid].sum(), 1))
            pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{cur_dice:.2f}", nsd=f"{cur_nsd:.2f}")

    dice_class = 100.0 * dice_sum / np.maximum(cnt, 1)
    nsd_class = 100.0 * nsd_sum / np.maximum(cnt, 1)
    valid = cnt > 0
    mean_dice = float(dice_class[valid].mean()) if valid.any() else 0.0
    mean_nsd = float(nsd_class[valid].mean()) if valid.any() else 0.0
    return {
        'loss': total_loss / len(dataloader),
        'dice_class': dice_class, 'nsd_class': nsd_class, 'cnt': cnt,
        'mean_dice': mean_dice, 'mean_nsd': mean_nsd,
        'score': (mean_dice + mean_nsd) / 2.0,
    }


# ============================================================================
# 单视图训练
# ============================================================================

def train_view(view_id: int, args):
    print("=" * 70)
    print(f"Training View {view_id} ({VIEW_NAMES[view_id]})")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    full_ds = FetusSegmentationDataset(args.images_dir, args.labels_dir, view_id)
    if len(full_ds) == 0:
        print(f"No samples for View {view_id}, skipping.")
        return None

    n_train = int(0.8 * len(full_ds))
    n_val = len(full_ds) - n_train
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    num_classes = full_ds.num_classes
    model = smp.FPN(
        encoder_name='mit_b5', encoder_weights='imagenet',
        in_channels=3, classes=num_classes, activation=None,
    ).to(device)
    print(f"Model: FPN + mit_b5, Classes: {num_classes}")
    print(f"  {[ANATOMY_NAMES[c] for c in full_ds.original_classes]}")

    criterion = CombinedLoss(num_classes)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_score = 0.0
    history = {'train_loss': [], 'train_dice': [],
               'val_loss': [], 'val_dice': [], 'val_nsd': [], 'val_score': []}

    save_dir = Path(args.save_dir) / f"view_{view_id}_{VIEW_NAMES[view_id]}"
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_dice = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, num_classes)
        val = validate(model, val_loader, criterion, device, epoch, num_classes)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['val_loss'].append(val['loss'])
        history['val_dice'].append(val['mean_dice'])
        history['val_nsd'].append(val['mean_nsd'])
        history['val_score'].append(val['score'])

        print(f"  Train Loss: {train_loss:.4f}  Dice: {train_dice:.4f}")
        print(f"  Val   Dice: {val['mean_dice']:.2f}%  NSD: {val['mean_nsd']:.2f}%  Score: {val['score']:.2f}%")
        for c in range(num_classes - 1):
            if val['cnt'][c] > 0:
                name = ANATOMY_NAMES[full_ds.original_classes[c + 1]]
                print(f"    {name}: Dice={val['dice_class'][c]:.2f}% NSD={val['nsd_class'][c]:.2f}%")

        if val['score'] > best_score:
            best_score = val['score']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val['mean_dice'], 'val_nsd': val['mean_nsd'],
                'val_score': val['score'], 'val_loss': val['loss'],
                'num_classes': num_classes,
                'original_classes': full_ds.original_classes,
                'class_to_idx': full_ds.class_to_idx,
            }, save_dir / 'best_model.pth')
            print(f"  *** Best model saved! Score: {best_score:.2f}% ***")

    with open(save_dir / 'history.json', 'w') as f:
        json.dump({k: [float(v) for v in vs] for k, vs in history.items()}, f, indent=2)
    with open(save_dir / 'config.json', 'w') as f:
        json.dump({
            'view_id': view_id, 'view_name': VIEW_NAMES[view_id],
            'num_classes': num_classes,
            'original_classes': full_ds.original_classes,
            'class_names': [ANATOMY_NAMES[c] for c in full_ds.original_classes],
            'encoder': 'mit_b5', 'architecture': 'FPN',
            'best_score': float(best_score), 'epochs': args.epochs,
        }, f, indent=2)

    print(f"\nView {view_id} done. Best Score: {best_score:.2f}%")
    return best_score


# ============================================================================
# main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Segmentation Training (FPN + MIT-B5)')
    parser.add_argument('--images_dir', type=str,
                        default='/root/autodl-tmp/_FETUS_data/train/images')
    parser.add_argument('--labels_dir', type=str,
                        default='/root/autodl-tmp/_FETUS_data/train/labels')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--views', type=int, nargs='+', default=[1, 2, 3, 4])
    args = parser.parse_args()

    print("=" * 70)
    print("Fetus2026 - Segmentation Training (FPN + MIT-B5)")
    print("=" * 70)
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    results = {}
    for view_id in args.views:
        if view_id not in VIEW_CLASSES:
            continue
        score = train_view(view_id, args)
        if score is not None:
            results[view_id] = score

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    for vid, sc in results.items():
        print(f"  View {vid} ({VIEW_NAMES[vid]}): {sc:.2f}%")
    if results:
        print(f"  Average: {np.mean(list(results.values())):.2f}%")


if __name__ == '__main__':
    main()
