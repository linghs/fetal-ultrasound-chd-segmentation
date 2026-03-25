#!/usr/bin/env python3
"""
Fetus2026 Challenge - CHD 分类训练脚本
模型: Xception encoder + Classification Head
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
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

# ============================================================================
# 配置
# ============================================================================

VIEW_NAMES = {1: "4CH", 2: "LVOT", 3: "RVOT", 4: "3VT"}

CHD_NAMES = {
    0: "Ventricular Septal Defect (VSD)",
    1: "Atrioventricular Valve Stenosis/Atresia (AVV)",
    2: "Aortic Hypoplasia (AH)",
    3: "Aortic Valve Stenosis (AS)",
    4: "Double Outlet Right Ventricle (DORV)",
    5: "Pulmonary Valve Stenosis (PS)",
    6: "Right Aortic Arch (RAA)",
}

NUM_CHD_CLASSES = 7

CLS_ALLOWED = {
    1: [0, 1],        # 4CH: VSD, AVV
    2: [0, 2, 3],     # LVOT: VSD, AH, AS
    3: [4, 5],        # RVOT: DORV, PS
    4: [2, 5, 6],     # 3VT: AH, PS, RAA
}


# ============================================================================
# 数据集
# ============================================================================

class FetusClassificationDataset(Dataset):
    def __init__(self, images_dir: str, labels_dir: str, view_id: int):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.view_id = view_id
        self.allowed_chd_classes = CLS_ALLOWED.get(view_id, list(range(NUM_CHD_CLASSES)))
        self.num_classes = len(self.allowed_chd_classes)
        self.original_to_new_idx = {orig: new for new, orig in enumerate(self.allowed_chd_classes)}
        self.samples = self._collect_samples()
        print(f"View {view_id} ({VIEW_NAMES[view_id]}): {len(self.samples)} samples, "
              f"{self.num_classes} CHD classes: {[CHD_NAMES[c] for c in self.allowed_chd_classes]}")

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
            chd_full = f['label'][()]
        chd_label = np.array([chd_full[c] for c in self.allowed_chd_classes], dtype=np.float32)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return {'image': image, 'label': torch.from_numpy(chd_label), 'img_id': sample['img_id']}


class SubsetDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


# ============================================================================
# 分层划分
# ============================================================================

def stratified_split(dataset, test_size=0.2, random_state=42):
    n = len(dataset)
    allowed = dataset.allowed_chd_classes

    all_labels = np.stack([dataset[i]['label'].numpy() for i in range(n)])
    pos_counts = all_labels.sum(axis=0)

    print(f"\nLabel distribution: total={n}, "
          f"normal={int((all_labels.sum(1) == 0).sum())}, "
          f"abnormal={int((all_labels.sum(1) > 0).sum())}")
    for i, c in enumerate(allowed):
        print(f"  CHD {c} ({CHD_NAMES[c]}): {int(pos_counts[i])} positive")

    train_idx, val_idx = set(), set()

    normal = np.where(all_labels.sum(1) == 0)[0]
    if len(normal) > 0:
        tr, va = train_test_split(normal, test_size=test_size, random_state=random_state)
        train_idx.update(tr)
        val_idx.update(va)

    for i, orig in enumerate(allowed):
        pos = np.where(all_labels[:, i] == 1)[0]
        if len(pos) == 0:
            continue
        if len(pos) <= 2:
            if len(pos) == 1:
                train_idx.update(pos)
            else:
                train_idx.add(pos[0])
                val_idx.add(pos[1])
        else:
            tr, va = train_test_split(pos, test_size=test_size, random_state=random_state + i)
            train_idx.update(tr)
            val_idx.update(va)

    train_idx = sorted(train_idx)
    val_idx = sorted(val_idx)

    # 确保验证集每个 CHD 类型至少有 1 个正样本
    for i, orig in enumerate(allowed):
        if pos_counts[i] > 0:
            val_pos = sum(1 for idx in val_idx if all_labels[idx, i] == 1)
            if val_pos == 0:
                candidates = [idx for idx in train_idx if all_labels[idx, i] == 1]
                if candidates:
                    move = candidates[0]
                    train_idx.remove(move)
                    val_idx.append(move)
                    val_idx = sorted(val_idx)

    print(f"Split: train={len(train_idx)}, val={len(val_idx)}")
    return train_idx, val_idx


# ============================================================================
# 模型
# ============================================================================

class ClassificationModel(nn.Module):
    def __init__(self, encoder_name: str, num_classes: int):
        super().__init__()
        self.encoder = smp.encoders.get_encoder(
            encoder_name, in_channels=3, depth=5, weights='imagenet')
        out_ch = self.encoder.out_channels[-1]
        self.classifier = smp.base.ClassificationHead(
            in_channels=out_ch, classes=num_classes,
            pooling="avg", dropout=0.2, activation=None)

    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features[-1])


# ============================================================================
# 评估指标
# ============================================================================

def compute_metrics(pred: torch.Tensor, target: torch.Tensor, threshold=0.5):
    probs = torch.sigmoid(pred)
    pred_bin = (probs >= threshold).float()
    probs_np = probs.detach().cpu().numpy()
    pred_np = pred_bin.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    num_cls = target.shape[1]
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': [], 'ap': []}

    for c in range(num_cls):
        pc, tc, prob_c = pred_np[:, c], target_np[:, c], probs_np[:, c]
        tp = ((pc == 1) & (tc == 1)).sum()
        fp = ((pc == 1) & (tc == 0)).sum()
        fn = ((pc == 0) & (tc == 1)).sum()
        tn = ((pc == 0) & (tc == 0)).sum()
        acc = (tp + tn) / (tp + fp + fn + tn + 1e-8)
        prec = tp / (tp + fp + 1e-8) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn + 1e-8) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec + 1e-8) if (prec + rec) > 0 else 0.0
        try:
            auc = roc_auc_score(tc, prob_c) if tc.sum() > 0 and (tc == 0).sum() > 0 else 0.0
        except Exception:
            auc = 0.0
        try:
            ap = average_precision_score(tc, prob_c) if tc.sum() > 0 else 0.0
        except Exception:
            ap = 0.0
        metrics['accuracy'].append(acc)
        metrics['precision'].append(prec)
        metrics['recall'].append(rec)
        metrics['f1'].append(f1)
        metrics['auc'].append(auc)
        metrics['ap'].append(ap)

    for key in list(metrics.keys()):
        metrics[f'{key}_mean'] = np.mean(metrics[key]) if metrics[key] else 0.0
    return metrics


# ============================================================================
# 训练 & 验证
# ============================================================================

def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    all_f1, all_auc = [], []
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        optimizer.zero_grad()
        out = model(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        with torch.no_grad():
            m = compute_metrics(out, labels)
            all_f1.append(m['f1_mean'])
            all_auc.append(m['auc_mean'])
        pbar.set_postfix(loss=f"{loss.item():.4f}", f1=f"{m['f1_mean']:.4f}")
    return total_loss / len(loader), {'f1': np.mean(all_f1), 'auc': np.mean(all_auc)}


def validate(model, loader, criterion, device, epoch, num_classes, allowed):
    model.eval()
    total_loss = 0
    all_f1, all_auc = [], []
    cls_f1 = [[] for _ in range(num_classes)]
    cls_auc = [[] for _ in range(num_classes)]

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]")
    with torch.no_grad():
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            out = model(images)
            loss = criterion(out, labels)
            total_loss += loss.item()
            m = compute_metrics(out, labels)
            all_f1.append(m['f1_mean'])
            all_auc.append(m['auc_mean'])
            for c in range(num_classes):
                if len(m['f1']) > c:
                    cls_f1[c].append(m['f1'][c])
                    cls_auc[c].append(m['auc'][c])
            pbar.set_postfix(loss=f"{loss.item():.4f}", f1=f"{m['f1_mean']:.4f}")

    avg_f1 = np.mean(all_f1)
    avg_auc = np.mean(all_auc)
    return {
        'loss': total_loss / len(loader),
        'f1': avg_f1, 'auc': avg_auc, 'score': avg_f1,
        'cls_f1': [np.mean(v) if v else 0.0 for v in cls_f1],
        'cls_auc': [np.mean(v) if v else 0.0 for v in cls_auc],
    }


# ============================================================================
# 单视图训练
# ============================================================================

def train_view(view_id: int, args):
    print("=" * 70)
    print(f"Training View {view_id} ({VIEW_NAMES[view_id]}) Classification")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    full_ds = FetusClassificationDataset(args.images_dir, args.labels_dir, view_id)
    if len(full_ds) == 0:
        print(f"No samples for View {view_id}, skipping.")
        return None

    # 类别权重
    all_labels = np.stack([full_ds[i]['label'].numpy() for i in range(len(full_ds))])
    pos_counts = all_labels.sum(axis=0)
    neg_counts = len(all_labels) - pos_counts
    pos_weights = torch.tensor(neg_counts / (pos_counts + 1e-8), dtype=torch.float32)

    train_idx, val_idx = stratified_split(full_ds, test_size=args.val_split)
    train_ds = SubsetDataset(full_ds, train_idx)
    val_ds = SubsetDataset(full_ds, val_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    num_classes = full_ds.num_classes
    model = ClassificationModel('xception', num_classes).to(device)
    print(f"Model: Xception, Classes: {num_classes}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_score = 0.0
    history = {'train_loss': [], 'train_f1': [], 'train_auc': [],
               'val_loss': [], 'val_f1': [], 'val_auc': [], 'val_score': []}

    save_dir = Path(args.save_dir) / f"classification_view_{view_id}_{VIEW_NAMES[view_id]}"
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_m = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch)
        val = validate(model, val_loader, criterion, device, epoch,
                       num_classes, full_ds.allowed_chd_classes)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_f1'].append(train_m['f1'])
        history['train_auc'].append(train_m['auc'])
        history['val_loss'].append(val['loss'])
        history['val_f1'].append(val['f1'])
        history['val_auc'].append(val['auc'])
        history['val_score'].append(val['score'])

        print(f"  Train Loss: {train_loss:.4f}  F1: {train_m['f1']:.4f}")
        print(f"  Val   F1: {val['f1']:.4f}  AUC: {val['auc']:.4f}  Score: {val['score']:.4f}")
        for i, c in enumerate(full_ds.allowed_chd_classes):
            print(f"    {CHD_NAMES[c]}: F1={val['cls_f1'][i]:.4f} AUC={val['cls_auc'][i]:.4f}")

        if val['score'] > best_score:
            best_score = val['score']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val['f1'], 'val_auc': val['auc'],
                'val_score': val['score'], 'val_loss': val['loss'],
                'num_classes': num_classes,
                'allowed_chd_classes': full_ds.allowed_chd_classes,
                'pos_weights': pos_weights,
            }, save_dir / 'best_model.pth')
            print(f"  *** Best model saved! Score: {best_score:.4f} ***")

    with open(save_dir / 'history.json', 'w') as f:
        json.dump({k: [float(v) for v in vs] for k, vs in history.items()}, f, indent=2)
    with open(save_dir / 'config.json', 'w') as f:
        json.dump({
            'view_id': view_id, 'view_name': VIEW_NAMES[view_id],
            'num_classes': num_classes,
            'allowed_chd_classes': full_ds.allowed_chd_classes,
            'chd_names': {c: CHD_NAMES[c] for c in full_ds.allowed_chd_classes},
            'encoder': 'xception',
            'best_score': float(best_score), 'epochs': args.epochs,
        }, f, indent=2)

    print(f"\nView {view_id} done. Best Score: {best_score:.4f}")
    return best_score


# ============================================================================
# main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Classification Training (Xception)')
    parser.add_argument('--images_dir', type=str,
                        default='/root/autodl-tmp/_FETUS_data/train/images')
    parser.add_argument('--labels_dir', type=str,
                        default='/root/autodl-tmp/_FETUS_data/train/labels')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--views', type=int, nargs='+', default=[1, 2, 3, 4])
    args = parser.parse_args()

    print("=" * 70)
    print("Fetus2026 - Classification Training (Xception)")
    print("=" * 70)
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    results = {}
    for view_id in args.views:
        if view_id not in VIEW_NAMES:
            continue
        score = train_view(view_id, args)
        if score is not None:
            results[view_id] = score

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    for vid, sc in results.items():
        print(f"  View {vid} ({VIEW_NAMES[vid]}): {sc:.4f}")
    if results:
        print(f"  Average: {np.mean(list(results.values())):.4f}")


if __name__ == '__main__':
    main()
