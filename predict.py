#!/usr/bin/env python3
"""
Fetus2026 Challenge - 验证集预测脚本
分割: FPN + MIT-B5
分类: Xception
"""

import os
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from pathlib import Path
from tqdm import tqdm
import argparse
import json

# ============================================================================
# 配置
# ============================================================================

VIEW_NAMES = {1: "4CH", 2: "LVOT", 3: "RVOT", 4: "3VT"}

VIEW_CLASSES = {
    1: [0, 1, 2, 3, 4, 5, 6, 7],
    2: [0, 1, 2, 4, 8],
    3: [0, 6, 8, 9, 10, 11, 12],
    4: [0, 9, 12, 13, 14],
}

CLS_ALLOWED = {
    1: [0, 1],
    2: [0, 2, 3],
    3: [4, 5],
    4: [2, 5, 6],
}

NUM_CHD_CLASSES = 7


# ============================================================================
# 数据集
# ============================================================================

class PredictionDataset(Dataset):
    def __init__(self, images_dir: str, image_ids: list = None):
        self.images_dir = Path(images_dir)
        if image_ids is None:
            files = sorted(self.images_dir.glob("*.h5"), key=lambda x: int(x.stem))
            self.image_ids = [int(f.stem) for f in files]
        else:
            self.image_ids = sorted(image_ids)
        print(f"Found {len(self.image_ids)} images to predict")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        with h5py.File(self.images_dir / f"{img_id}.h5", 'r') as f:
            image = f['image'][()]
            view = int(np.array(f['view']).flat[0])
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return {'image': image, 'img_id': img_id, 'view': view}


# ============================================================================
# 模型加载
# ============================================================================

def load_seg_model(checkpoint_path: str, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    num_classes = ckpt['num_classes']
    original_classes = ckpt['original_classes']
    class_to_idx = ckpt['class_to_idx']

    config_path = Path(checkpoint_path).parent / 'config.json'
    encoder = 'mit_b5'
    if config_path.exists():
        with open(config_path) as f:
            encoder = json.load(f).get('encoder', 'mit_b5')

    model = smp.FPN(
        encoder_name=encoder, encoder_weights=None,
        in_channels=3, classes=num_classes, activation=None,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    return model, original_classes, class_to_idx


def load_cls_model(checkpoint_path: str, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    num_classes = ckpt['num_classes']
    allowed = ckpt.get('allowed_chd_classes', [0, 1])

    config_path = Path(checkpoint_path).parent / 'config.json'
    encoder = 'xception'
    if config_path.exists():
        with open(config_path) as f:
            encoder = json.load(f).get('encoder', 'xception')

    enc = smp.encoders.get_encoder(encoder, in_channels=3, depth=5, weights=None)
    out_ch = enc.out_channels[-1]

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = enc
            self.classifier = smp.base.ClassificationHead(
                in_channels=out_ch, classes=num_classes,
                pooling="avg", dropout=0.2, activation=None)
        def forward(self, x):
            return self.classifier(self.encoder(x)[-1])

    model = _Model().to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    return model, allowed


def find_model(checkpoints_dir: str, view_id: int, task: str):
    base = Path(checkpoints_dir)
    if task == 'seg':
        pattern = f"view_{view_id}_*"
    else:
        pattern = f"classification_view_{view_id}_*"
    for d in base.glob(pattern):
        p = d / "best_model.pth"
        if p.exists():
            return p
    return None


# ============================================================================
# 保存 & 预测
# ============================================================================

def save_pred_h5(path: str, mask=None, label=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    existing_mask, existing_label = None, None
    if os.path.exists(path):
        try:
            with h5py.File(path, 'r') as f:
                if 'mask' in f:
                    existing_mask = f['mask'][:]
                if 'label' in f:
                    existing_label = f['label'][:]
        except Exception:
            pass
    final_mask = mask if mask is not None else existing_mask
    final_label = label if label is not None else existing_label
    with h5py.File(path, 'w') as f:
        if final_mask is not None:
            f.create_dataset('mask', data=final_mask.astype(np.uint8), compression='gzip')
        if final_label is not None:
            f.create_dataset('label', data=final_label.astype(np.uint8), compression='gzip')


def predict_seg(model, loader, device, view_id, original_classes, class_to_idx):
    model.eval()
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    results = {}
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Seg prediction (View {view_id})"):
            images = batch['image'].to(device)
            views = batch['view']
            mask = (views == view_id)
            if not mask.any():
                continue
            out = model(images[mask])
            preds = torch.argmax(out, dim=1).cpu().numpy()
            ids = [batch['img_id'][i] for i in range(len(views)) if mask[i]]
            for i, img_id in enumerate(ids):
                orig_mask = np.zeros_like(preds[i], dtype=np.uint8)
                for new_idx, orig_cls in idx_to_class.items():
                    orig_mask[preds[i] == new_idx] = orig_cls
                results[img_id if isinstance(img_id, int) else img_id.item()] = orig_mask
    return results


def predict_cls(model, loader, device, view_id, allowed, threshold=0.5):
    model.eval()
    orig_to_new = {orig: new for new, orig in enumerate(allowed)}
    results = {}
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Cls prediction (View {view_id})"):
            images = batch['image'].to(device)
            views = batch['view']
            mask = (views == view_id)
            if not mask.any():
                continue
            out = model(images[mask])
            probs = torch.sigmoid(out)
            pred_labels = (probs >= threshold).long().cpu().numpy()
            ids = [batch['img_id'][i] for i in range(len(views)) if mask[i]]
            for i, img_id in enumerate(ids):
                full_label = np.zeros(NUM_CHD_CLASSES, dtype=np.uint8)
                for orig_idx, new_idx in orig_to_new.items():
                    full_label[orig_idx] = pred_labels[i][new_idx]
                results[img_id if isinstance(img_id, int) else img_id.item()] = full_label
    return results


# ============================================================================
# main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Fetus2026 Prediction (FPN+MIT-B5 / Xception)')
    parser.add_argument('--images_dir', type=str, default='data/val/images')
    parser.add_argument('--output_dir', type=str, default='data/val/preds')
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints')
    parser.add_argument('--task', choices=['seg', 'cls', 'both'], default='both')
    parser.add_argument('--views', type=int, nargs='+', default=[1, 2, 3, 4])
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--cls_threshold', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    print("=" * 70)
    print("Fetus2026 - Prediction")
    print("=" * 70)
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"  Using: {device}")

    dataset = PredictionDataset(args.images_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # ---- 分割 ----
    if args.task in ('seg', 'both'):
        print("\n--- Segmentation (FPN + MIT-B5) ---")
        for vid in args.views:
            mp = find_model(args.checkpoints_dir, vid, 'seg')
            if mp is None:
                print(f"  View {vid}: model not found, skipping")
                continue
            print(f"  View {vid}: loading {mp}")
            model, orig_cls, cls2idx = load_seg_model(str(mp), device)
            seg_res = predict_seg(model, loader, device, vid, orig_cls, cls2idx)
            print(f"  View {vid}: {len(seg_res)} predictions")
            for img_id, mask in seg_res.items():
                all_results.setdefault(img_id, {})['mask'] = mask

    # ---- 分类 ----
    if args.task in ('cls', 'both'):
        print("\n--- Classification (Xception) ---")
        for vid in args.views:
            mp = find_model(args.checkpoints_dir, vid, 'cls')
            if mp is None:
                print(f"  View {vid}: model not found, skipping")
                continue
            print(f"  View {vid}: loading {mp}")
            model, allowed = load_cls_model(str(mp), device)
            cls_res = predict_cls(model, loader, device, vid, allowed, args.cls_threshold)
            print(f"  View {vid}: {len(cls_res)} predictions")
            for img_id, label in cls_res.items():
                all_results.setdefault(img_id, {})['label'] = label

    # ---- 保存 ----
    print(f"\nSaving {len(all_results)} predictions to {args.output_dir}")
    n_mask, n_label, n_both = 0, 0, 0
    for img_id, res in tqdm(all_results.items(), desc="Saving"):
        m = res.get('mask')
        l = res.get('label')
        if m is not None:
            n_mask += 1
        if l is not None:
            n_label += 1
        if m is not None and l is not None:
            n_both += 1
        save_pred_h5(str(output_dir / f"{img_id}.h5"), mask=m, label=l)

    print(f"\nDone! mask={n_mask}, label={n_label}, both={n_both}")


if __name__ == '__main__':
    main()
