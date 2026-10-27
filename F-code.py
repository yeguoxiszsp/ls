# -*- coding: utf-8 -*-
"""
LDG-CNN for Latin dance posture recognition.
Leakage-free protocol (response to Reviewer 1, Comment 1):
  1) Stratified 5-fold split is performed on the ORIGINAL images (1,200).
  2) Augmentation is applied ONLY to the training partition of each fold.
  3) Validation/test partitions contain original images only.
  4) No original image or its augmented copies appear in both train and test.
"""

import os, time, copy, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------------
# 0. Reproducibility
# -----------------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
set_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------
# 1. Hyperparameters (Sec. 4.3)
# -----------------------------
NUM_CLASSES   = 6
NUM_EPOCHS    = 100
BATCH_SIZE    = 64
LR            = 1e-4
WEIGHT_DECAY  = 1e-3
K_NEIGHBORS   = 9
IMG_SIZE      = 224
FEAT_H = FEAT_W = 16          # after 3 max-poolings: 224 -> 112 -> 56 -> 28? see note below
N_FOLDS       = 5
PATIENCE      = 10            # early stopping on val loss

# NOTE on spatial size: 224 with three 2x2 max-pools gives 28x28, not 16x16.
# The paper states H=W=16. To match the paper exactly, use stride-2 convs or
# an extra pooling. Here we keep the paper's stated H=W=16 by using
# kernel=3, stride=2 in the first conv block (see CNN backbone below).

# -----------------------------
# 2. Dataset: original images + fold-aware augmentation
# -----------------------------
CLASS_NAMES = ["ChaCha", "Samba", "Rumba", "PasoDoble", "Jive", "P6"]

class LatinDanceDataset(Dataset):
    """
    image_paths: list of paths to ORIGINAL images
    labels:      list of int labels
    transform:   augmentation pipeline (train) or deterministic (val/test)
    """
    def __init__(self, image_paths, labels, transform):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = self.transform(img)
        return img, self.labels[idx]

def build_file_index(raw_root):
    """Scan data/raw/<class>/*.jpg and return paths + labels."""
    paths, labels = [], []
    for ci, cname in enumerate(CLASS_NAMES):
        cdir = os.path.join(raw_root, cname)
        for fname in sorted(os.listdir(cdir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(cdir, fname))
                labels.append(ci)
    return paths, labels

# Augmentation ONLY for training partitions (Sec. 4.1)
train_transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.2)),  # ~ random scaling 0.8-1.2x
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3),
])

# Validation / test: original images only, deterministic resize
eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3),
])

# -----------------------------
# 3. LDG-CNN model (Sec. 3)
# -----------------------------
class LDG_CNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, k=K_NEIGHBORS):
        super().__init__()
        self.k = k

        # CNN backbone: three conv blocks, output 16x16x256 to match paper
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),   # 224 -> 112
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),       # 112 -> 56
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 56 -> 28
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),      # 28 -> 14
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # 14 -> 7
            nn.BatchNorm2d(256), nn.ReLU(),
        )
        # After above: 7x7. To reach 16x16 as in paper, adjust strides.
        # For clarity and paper consistency we instead use the paper's exact blocks:
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2,2),   # 112
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2,2), # 56
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2,2),# 28
        )
        # 224 -> 28x28. If 16x16 is required, add AdaptiveAvgPool2d((16,16)).
        self.pool_to_16 = nn.AdaptiveAvgPool2d((FEAT_H, FEAT_W))  # -> 16x16x256

        # LDG: graph conv weight (C -> C_out). Paper says output dim 128 for z.
        self.gcn_out = 128
        self.weight = nn.Parameter(torch.empty(256, self.gcn_out))
        nn.init.xavier_uniform_(self.weight)

        # Classifier: g (256) + z (128) = 384 -> 6
        self.fc = nn.Linear(256 + self.gcn_out, num_classes)

    def ldg_forward(self, feat):
        """
        feat: [B, C, H, W] with H=W=16, C=256.
        Build sparse k-NN graph in feature space, apply graph convolution.
        """
        B, C, H, W = feat.shape
        n = H * W
        x = feat.view(B, C, n).transpose(1, 2)          # [B, n, C]

        # cosine similarity
        x_norm = F.normalize(x, dim=2)                   # [B, n, C]
        S = torch.bmm(x_norm, x_norm.transpose(1, 2))    # [B, n, n]

        # hard k-NN selection (differentiable via STE in backward)
        k = min(self.k, n - 1)
        topk_val, topk_idx = torch.topk(S, k=k + 1, dim=-1)  # include self
        # build sparse adjacency A (including self-loop)
        A = torch.zeros_like(S)
        A.scatter_(-1, topk_idx, 1.0)                    # [B, n, n], at most k+1 per row

        # straight-through: forward uses hard A, backward uses soft weights
        soft = torch.softmax(topk_val / 1.0, dim=-1)     # [B, n, k+1]
        A_soft = torch.zeros_like(S).scatter_(-1, topk_idx, soft)
        A = A + (A_soft - A_soft.detach())               # STE

        # symmetric normalization D^{-1/2} A D^{-1/2}
        deg = A.sum(-1).clamp(min=1e-6)                  # [B, n]
        d_inv_sqrt = deg.pow(-0.5)
        A_norm = A * d_inv_sqrt.unsqueeze(-1) * d_inv_sqrt.unsqueeze(1)

        # graph convolution: Z = ReLU(A_norm @ X @ W)
        Z = torch.relu(torch.bmm(A_norm, torch.bmm(x, self.weight.unsqueeze(0).expand(B, -1, -1))))
        z = Z.mean(dim=1)                                # [B, gcn_out]
        return z

    def forward(self, x):
        feat = self.cnn(x)                               # [B,256,28,28]
        feat = self.pool_to_16(feat)                     # [B,256,16,16]
        g = feat.mean(dim=(2, 3))                        # [B,256] GAP
        z = self.ldg_forward(feat)                       # [B,128] LDG
        h = torch.cat([g, z], dim=1)                     # [B,384]
        return self.fc(h)

# -----------------------------
# 4. Leakage-free 5-fold CV
# -----------------------------
def run_fold(fold, tr_paths, tr_labels, va_paths, va_labels, te_paths, te_labels):
    tr_ds = LatinDanceDataset(tr_paths, tr_labels, train_transform)
    va_ds = LatinDanceDataset(va_paths, va_labels, eval_transform)
    te_ds = LatinDanceDataset(te_paths, te_labels, eval_transform)

    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    te_loader = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = LDG_CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_val, best_state, wait = float('inf'), None, 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        for imgs, labels in tr_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward(); optimizer.step()
        scheduler.step()

        # validation
        model.eval(); vloss = 0.0
        with torch.no_grad():
            for imgs, labels in va_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                vloss += criterion(model(imgs), labels).item() * imgs.size(0)
        vloss /= len(va_ds)
        if vloss < best_val:
            best_val, best_state, wait = vloss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval(); preds, gts = [], []
    with torch.no_grad():
        for imgs, labels in te_loader:
            imgs = imgs.to(device)
            preds.extend(model(imgs).argmax(1).cpu().numpy())
            gts.extend(labels.numpy())
    acc = (np.array(preds) == np.array(gts)).mean()
    rep = classification_report(gts, preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    print(f"[Fold {fold}] test acc={acc:.4f}  macro-F1={rep['macro avg']['f1-score']:.4f}")
    return acc, rep, preds, gts

def main(raw_root="./data/raw"):
    paths, labels = build_file_index(raw_root)
    paths, labels = np.array(paths), np.array(labels)
    assert len(paths) == 1200, f"Expected 1200 original images, got {len(paths)}"

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    accs, all_preds, all_gts = [], [], []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(paths, labels), start=1):
        # split train into train/val (val is original images only)
        tr_paths, tr_labels = paths[tr_idx], labels[tr_idx]
        te_paths, te_labels = paths[te_idx], labels[te_idx]
        # inner stratified split for val
        skf_in = StratifiedKFold(n_splits=5, shuffle=True, random_state=fold)
        inner_tr, inner_va = next(iter(skf_in.split(tr_paths, tr_labels)))
        va_paths, va_labels = tr_paths[inner_va], tr_labels[inner_va]
        tr_paths, tr_labels = tr_paths[inner_tr], tr_labels[inner_tr]

        acc, rep, preds, gts = run_fold(fold, tr_paths, tr_labels,
                                        va_paths, va_labels,
                                        te_paths, te_labels)
        accs.append(acc); all_preds.extend(preds); all_gts.extend(gts)

    print(f"\n5-fold CV: {np.mean(accs)*100:.2f} ± {np.std(accs)*100:.2f} (%)")
    print(classification_report(all_gts, all_preds, target_names=CLASS_NAMES, digits=4))

if __name__ == "__main__":
    main()