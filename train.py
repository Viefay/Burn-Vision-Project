"""
╔═════════════════════════════════════════════════════════════════╗
║          Burn Wound Segmentation — Training Script              ║
║                     BurnVision Project                          ║
╠═════════════════════════════════════════════════════════════════╣
║  Models (5):                                                    ║
║   1. unet_resnet34       — UNet + ResNet34 (Baseline)           ║
║   2. unet_densenet169    — UNet + DenseNet169 (Baseline 2)      ║
║   3. unet_efficientb4    — UNet + EfficientNet-B4               ║
║   4. unetpp_efficientb4  — UNet++ + EfficientNet-B4 (Proposed)  ║
║   5. deeplabv3p_effb4    — DeepLabV3+ + EfficientNet-B4 (Best)  ║
╠═════════════════════════════════════════════════════════════════╣
║  Cara run (1 by 1):                                             ║
║   python train.py --model unet_resnet34                         ║
║   python train.py --model unet_densenet169                      ║
║   python train.py --model unet_efficientb4                      ║
║   python train.py --model unetpp_efficientb4                    ║
║   python train.py --model deeplabv3p_effb4                      ║
║   python train.py --model all   ← train semua sekaligus         ║
╚═════════════════════════════════════════════════════════════════╝
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import cv2
from pathlib import Path, PureWindowsPath

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ================================================================
# CLASS DEFINITIONS (warna RGB → class ID)
# ================================================================
CLASS_NAMES = [
    "normal",                # 0  — #AEFCAE
    "superficial_thickness", # 1  — #F8BDC9
    "mid_deep_thickness",    # 2  — #FF007C
    "full_thickness",        # 3  — #FDFDFD
    "medical_object",        # 4  — #F8F810
    "tangan_pemeriksa",      # 5  — #1BCEF0
    "background",            # 6  — #2323FA
]
NUM_CLASSES = len(CLASS_NAMES)  # 7

COLOR_MAP: dict = {
    (0xAE, 0xFC, 0xAE): 0,
    (0xF8, 0xBD, 0xC9): 1,
    (0xFF, 0x00, 0x7C): 2,
    (0xFD, 0xFD, 0xFD): 3,
    (0xF8, 0xF8, 0x10): 4,
    (0x1B, 0xCE, 0xF0): 5,
    (0x23, 0x23, 0xFA): 6,
}

IGNORE_CLASSES = [4, 5, 6]
EVAL_CLASSES   = [c for c in range(NUM_CLASSES) if c not in IGNORE_CLASSES]  # [0,1,2,3]

_COLORS_ARRAY  = np.array(list(COLOR_MAP.keys()),   dtype=np.uint8)
_IDS_ARRAY     = np.array(list(COLOR_MAP.values()), dtype=np.int64)
LABEL_TO_COLOR = {v: k for k, v in COLOR_MAP.items()}

VIS_COLORS = [
    "#AEFCAE", "#F8BDC9", "#FF007C", "#CCCCCC",
    "#F8F810", "#1BCEF0", "#2323FA",
]


def rgb_mask_to_label(mask_rgb: np.ndarray) -> np.ndarray:
    h, w, _ = mask_rgb.shape
    flat    = mask_rgb.reshape(-1, 3).astype(np.int32)
    diffs   = flat[:, None, :] - _COLORS_ARRAY[None, :, :].astype(np.int32)
    dist    = (diffs ** 2).sum(axis=-1)
    nearest = dist.argmin(axis=-1)
    return _IDS_ARRAY[nearest].reshape(h, w).astype(np.uint8)


# ================================================================
#  CONFIG — EDIT DI SINI UNTUK MENGUBAH PATH & HYPERPARAMETER
# ================================================================
class Config:

    # [PATH] Sesuaikan path di bawah ini dengan lokasi project
    DATA_DIR   = r"D:\Thesis Luka Bakar\BurnVision Project\DATASET\Training"
    RAW_DIR    = os.path.join(DATA_DIR, "RAW_DATA")
    SEG_DIR    = os.path.join(DATA_DIR, "SEGMENTED_DATA")
    XLSX_PATH  = os.path.join(DATA_DIR, "FIlesPath.xlsx")
    OUTPUT_DIR = r"D:\Thesis Luka Bakar\BurnVision Project\outputs"

    IMG_SIZE   = 320      # ukuran input gambar (resize). Coba: 256, 320, 384, 512
    BATCH_SIZE = 16        # kurangi jika VRAM OOM, naikkan jika VRAM cukup. Coba: 4, 8, 16
    NUM_EPOCHS = 150      # jumlah epoch maksimum. Coba: 100, 150, 200
    LR         = 1e-4     # learning rate awal. Coba: 3e-4, 1e-4, 5e-5
    WEIGHT_DECAY      = 5e-4   # L2 regularisasi. Coba: 1e-4, 5e-4, 1e-3
    VAL_SPLIT         = 0.2    # proporsi validasi. Coba: 0.15, 0.2, 0.25
    SEED              = 42

    # ----------------------------------------------------------
    # [FINE-TUNE STRATEGY]
    # ----------------------------------------------------------
    ENCODER_FREEZE_EPOCHS = 15   # freeze encoder N epoch pertama.
                                  # Coba: 10, 15, 20
                                  # Lebih tinggi = decoder belajar lebih dulu
    UNFREEZE_LR_FACTOR    = 0.1  # LR setelah unfreeze = LR * faktor ini
                                  # Coba: 0.05, 0.1, 0.3
                                  # EfficientNet: gunakan 0.05 (lebih konservatif)

    # ----------------------------------------------------------
    # [REGULARISASI]
    # ----------------------------------------------------------
    GRAD_CLIP        = 1.0    # gradient clipping. Coba: 0.5, 1.0, 2.0
    LABEL_SMOOTHING  = 0.05   # label smoothing CE. Coba: 0.0, 0.05, 0.1
    DECODER_DROPOUT  = 0.1    # dropout decoder (UNet, UNet++, FPN, MAnet)
                               # Coba: 0.1, 0.2, 0.3
    DECODER_DROPOUT_EFF = 0.2 # dropout khusus model berbasis EfficientNet
                               # Coba: 0.2, 0.3

    # ----------------------------------------------------------
    # [EARLY STOPPING]
    # ----------------------------------------------------------
    EARLY_STOP_PATIENCE = 40  # stop jika tidak ada improvement N epoch
                               # Coba: 25, 40, 50
                               # Harus > ENCODER_FREEZE_EPOCHS

    # ----------------------------------------------------------
    # [LOSS FUNCTION WEIGHTS]
    # 0.34*Focal + 0.33*CE + 0.33*Dice = 1.0
    # ----------------------------------------------------------
    LOSS_FOCAL_W = 0.34   # bobot Focal Loss.  Coba: 0.4, 0.34, 0.25
    LOSS_CE_W    = 0.33   # bobot CrossEntropy. Coba: 0.3, 0.33, 0.4
    LOSS_DICE_W  = 0.33   # bobot Dice Loss.   Coba: 0.3, 0.33, 0.5
    FOCAL_GAMMA  = 2.0    # gamma Focal Loss. Coba: 1.5, 2.0, 3.0
                           # Lebih tinggi = lebih fokus ke kelas sulit

    # ----------------------------------------------------------
    # [MISC]
    # ----------------------------------------------------------
    NUM_WORKERS    = 0 if os.name == "nt" else 4
    SAVE_VIS_EVERY = 15   # simpan visualisasi prediksi setiap N epoch
    DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"


# ================================================================
# MODEL REGISTRY
# Tambah atau hapus model di sini
# Format: "nama_run": ("arsitektur", "encoder")
# Arsitektur yang tersedia di smp:
#   Unet, UnetPlusPlus, DeepLabV3Plus, FPN, MAnet, Linknet, PSPNet
# Encoder populer:
#   resnet34, resnet50, densenet169, efficientnet-b4,
#   efficientnet-b3, mit_b2, resnext50_32x4d
# ================================================================
MODEL_REGISTRY = {
    "unet_resnet34":      ("Unet",         "resnet34"),
    "unet_densenet169":   ("Unet",         "densenet169"),
    "unet_efficientb4":   ("Unet",         "efficientnet-b4"),
    "unetpp_efficientb4": ("UnetPlusPlus", "efficientnet-b4"),
    "deeplabv3p_effb4":   ("DeepLabV3Plus","efficientnet-b4"),
}


# ================================================================
# EARLY STOPPING
# ================================================================
class EarlyStopping:
    def __init__(self, patience=25, min_delta=1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best      = None
        self.stop      = False

    def __call__(self, score: float) -> bool:
        if self.best is None or score > self.best + self.min_delta:
            self.best    = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


# ================================================================
# DATASET
# ================================================================
def _fix_path(p: str, raw_dir: str, seg_dir: str) -> str:
    fname = PureWindowsPath(p).name
    if "RAW" in fname.upper() or "_raw" in fname:
        return os.path.join(raw_dir, fname)
    return os.path.join(seg_dir, fname)


def load_pairs(cfg: Config):
    raw_dir = cfg.RAW_DIR
    seg_dir = cfg.SEG_DIR

    if os.path.exists(cfg.XLSX_PATH):
        df = pd.read_excel(cfg.XLSX_PATH).iloc[:, :2]
        df.columns = ["RawPath", "SegPath"]
        df["RawPath"] = df["RawPath"].apply(lambda p: _fix_path(p, raw_dir, seg_dir))
        df["SegPath"] = df["SegPath"].apply(lambda p: _fix_path(p, raw_dir, seg_dir))
        df = df[df["RawPath"].apply(os.path.exists) & df["SegPath"].apply(os.path.exists)]
        df = df.reset_index(drop=True)
        print(f"Loaded {len(df)} valid pairs from XLSX")
        return df

    raw_files = {Path(f).stem.replace("_raw", ""): f
                 for f in os.listdir(raw_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))}
    seg_files = {Path(f).stem: f
                 for f in os.listdir(seg_dir) if f.lower().endswith(".png")}
    pairs = []
    for key, raw_fname in raw_files.items():
        if key in seg_files:
            pairs.append({
                "RawPath": os.path.join(raw_dir, raw_fname),
                "SegPath": os.path.join(seg_dir, seg_files[key]),
            })
    df = pd.DataFrame(pairs)
    print(f"Loaded {len(df)} valid pairs by filename matching")
    return df


class BurnWoundDataset(Dataset):
    def __init__(self, raw_paths, seg_paths, transform=None):
        self.raw_paths = raw_paths
        self.seg_paths = seg_paths
        self.transform = transform

    def __len__(self):
        return len(self.raw_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.raw_paths[idx])
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {self.raw_paths[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_bgr = cv2.imread(self.seg_paths[idx])
        if mask_bgr is None:
            raise FileNotFoundError(f"Cannot read mask: {self.seg_paths[idx]}")
        mask = rgb_mask_to_label(cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB))

        if self.transform:
            out   = self.transform(image=image, mask=mask)
            image = out["image"]
            mask  = out["mask"]
        return image, mask.long()


def get_transforms(img_size: int, is_train: bool):
    norm = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    if is_train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(scale=(0.85, 1.15), translate_percent=(-0.1, 0.1),
                     rotate=(-20, 20), p=0.5),
            A.ElasticTransform(alpha=1, sigma=30, p=0.2),
            A.GridDistortion(num_steps=5, distort_limit=0.15, p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
            A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15,
                                 val_shift_limit=8, p=0.3),
            A.GaussNoise(std_range=(0.01, 0.02), p=0.2),
            A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(16, 24),
                            hole_width_range=(16, 24), p=0.2),
            norm,
            ToTensorV2(),
        ], is_check_shapes=False)
    return A.Compose([
        A.Resize(img_size, img_size),
        norm,
        ToTensorV2(),
    ], is_check_shapes=False)


# ================================================================
# LOSS
# ================================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, class_weights=None):
        super().__init__()
        self.gamma   = gamma
        self.weights = class_weights

    def forward(self, pred, target):
        ce = nn.functional.cross_entropy(pred, target, weight=self.weights, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class CombinedLoss(nn.Module):
    def __init__(self, cfg: Config, class_weights=None):
        super().__init__()
        self.focal_w = cfg.LOSS_FOCAL_W
        self.ce_w    = cfg.LOSS_CE_W
        self.dice_w  = cfg.LOSS_DICE_W
        self.focal   = FocalLoss(gamma=cfg.FOCAL_GAMMA, class_weights=class_weights)
        self.ce      = nn.CrossEntropyLoss(weight=class_weights,
                                            label_smoothing=cfg.LABEL_SMOOTHING)
        self.dice    = smp.losses.DiceLoss(mode="multiclass", from_logits=True)

    def forward(self, pred, target):
        return (self.focal_w * self.focal(pred, target)
                + self.ce_w  * self.ce(pred, target)
                + self.dice_w * self.dice(pred, target))


# ================================================================
# METRICS
# ================================================================
def compute_metrics(preds_logits, targets):
    preds    = torch.argmax(preds_logits, dim=1)
    eps      = 1e-8
    dice_sum = iou_sum = 0.0

    for c in EVAL_CLASSES:
        pred_c = (preds == c).float()
        tgt_c  = (targets == c).float()
        TP = (pred_c * tgt_c).sum().item()
        FP = (pred_c * (1 - tgt_c)).sum().item()
        FN = ((1 - pred_c) * tgt_c).sum().item()
        dice_sum += (2 * TP) / (2 * TP + FP + FN + eps)
        iou_sum  += TP / (TP + FP + FN + eps)

    ignore_mask = torch.zeros_like(targets, dtype=torch.bool)
    for c in IGNORE_CLASSES:
        ignore_mask |= (targets == c)
    eval_pixels = ~ignore_mask
    accuracy = ((preds[eval_pixels] == targets[eval_pixels]).float().mean().item()
                if eval_pixels.sum() > 0 else 0.0)

    n = len(EVAL_CLASSES)
    return {"dice": dice_sum / n, "iou": iou_sum / n, "accuracy": accuracy}


def compute_class_weights(seg_paths: list, device: str) -> torch.Tensor:
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    for p in seg_paths:
        mask_bgr = cv2.imread(p)
        if mask_bgr is None:
            continue
        labels = rgb_mask_to_label(cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB))
        for c in range(NUM_CLASSES):
            counts[c] += (labels == c).sum()
    counts  = np.maximum(counts, 1)
    weights = 1.0 / counts
    weights = weights / weights.sum() * NUM_CLASSES
    print("  Class weights:", {CLASS_NAMES[i]: f"{weights[i]:.4f}" for i in range(NUM_CLASSES)})
    return torch.tensor(weights, dtype=torch.float32).to(device)


def print_class_distribution(seg_paths: list):
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    for p in seg_paths:
        mask_bgr = cv2.imread(p)
        if mask_bgr is None:
            continue
        labels = rgb_mask_to_label(cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB))
        for c in range(NUM_CLASSES):
            counts[c] += (labels == c).sum()
    total = counts.sum()
    print("\n  Class distribution (training set):")
    for c in range(NUM_CLASSES):
        bar = "█" * int(50 * counts[c] / max(total, 1))
        print(f"  {CLASS_NAMES[c]:<25} {100*counts[c]/total:5.1f}%  {bar}")
    print()


def _accumulate(acc, metrics):
    for k, v in metrics.items():
        acc[k] = acc.get(k, 0.0) + v


def _average(acc, n):
    return {k: v / n for k, v in acc.items()}


# ================================================================
# TRAIN / VAL LOOPS
# ================================================================
def run_epoch(model, loader, loss_fn, device, optimizer=None, grad_clip=1.0):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss  = 0.0
    acc_metrics = {}
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for images, masks in tqdm(loader, desc="train" if is_train else "val ", leave=False):
            images, masks = images.to(device), masks.to(device)
            if is_train:
                optimizer.zero_grad()
            preds = model(images)
            loss  = loss_fn(preds, masks)
            if is_train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()
            total_loss += loss.item()
            _accumulate(acc_metrics, compute_metrics(preds.detach(), masks))

    n = len(loader)
    return total_loss / n, _average(acc_metrics, n)


# ================================================================
# MODEL BUILDER
# ================================================================
def build_model(arch_name: str, encoder_name: str, cfg: Config) -> nn.Module:
    """
    Bangun model berdasarkan nama arsitektur dan encoder.
    Semua arsitektur diambil dari segmentation_models_pytorch (smp).
    """
    is_efficientnet = "efficientnet" in encoder_name
    dropout = cfg.DECODER_DROPOUT_EFF if is_efficientnet else cfg.DECODER_DROPOUT

    common_kwargs = dict(
        encoder_name    = encoder_name,
        encoder_weights = "imagenet",
        in_channels     = 3,
        classes         = NUM_CLASSES,
        activation      = None,
    )

    if arch_name == "Unet":
        model = smp.Unet(
            **common_kwargs,
            decoder_dropout  = dropout,
            decoder_channels = (256, 128, 64, 32, 16),
        )
    elif arch_name == "UnetPlusPlus":
        model = smp.UnetPlusPlus(
            **common_kwargs,
            decoder_dropout  = dropout,
            decoder_channels = (256, 128, 64, 32, 16),
        )
    elif arch_name == "DeepLabV3Plus":
        model = smp.DeepLabV3Plus(
            **common_kwargs,
            decoder_channels      = 256,   # ASPP output channels
            decoder_atrous_rates  = (12, 24, 36),  # ASPP dilation rates
                                                     # Coba: (6,12,18), (12,24,36)
        )
    elif arch_name == "FPN":
        model = smp.FPN(
            **common_kwargs,
            decoder_dropout       = dropout,
            decoder_merge_policy  = "add",  # "add" atau "cat"
        )
    elif arch_name == "MAnet":
        model = smp.MAnet(
            **common_kwargs,
            decoder_channels      = (256, 128, 64, 32, 16),
            decoder_pab_channels  = 64,
        )
    else:
        raise ValueError(f"Arsitektur tidak dikenal: {arch_name}. "
                         f"Pilihan: Unet, UnetPlusPlus, DeepLabV3Plus, FPN, MAnet")

    return model


# ================================================================
# TRAINING ORCHESTRATOR
# ================================================================
def train_model(run_name: str, arch_name: str, encoder_name: str,
                df_train, df_val, cfg: Config):

    print(f"\n{'='*62}")
    print(f"  Model    : {run_name.upper()}")
    print(f"  Arch     : {arch_name}  |  Encoder: {encoder_name}")
    print(f"  Device   : {cfg.DEVICE}")
    print(f"  Train    : {len(df_train)}  |  Val: {len(df_val)}")
    print(f"  Epochs   : {cfg.NUM_EPOCHS}  |  Batch: {cfg.BATCH_SIZE}  "
          f"|  LR: {cfg.LR:.0e}  |  IMG: {cfg.IMG_SIZE}px")
    print(f"{'='*62}")

    print_class_distribution(df_train["SegPath"].tolist())

    train_ds = BurnWoundDataset(
        df_train["RawPath"].tolist(), df_train["SegPath"].tolist(),
        transform=get_transforms(cfg.IMG_SIZE, is_train=True),
    )
    val_ds = BurnWoundDataset(
        df_val["RawPath"].tolist(), df_val["SegPath"].tolist(),
        transform=get_transforms(cfg.IMG_SIZE, is_train=False),
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE,
                              shuffle=True,  num_workers=cfg.NUM_WORKERS,
                              pin_memory=(cfg.DEVICE == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE,
                              shuffle=False, num_workers=cfg.NUM_WORKERS,
                              pin_memory=(cfg.DEVICE == "cuda"))

    model = build_model(arch_name, encoder_name, cfg).to(cfg.DEVICE)

    total_p    = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"  Parameters: {total_p:.1f}M total  |  {trainable_p:.1f}M trainable")

    # Freeze encoder
    for param in model.encoder.parameters():
        param.requires_grad = False
    frozen_p = sum(p.numel() for p in model.encoder.parameters()) / 1e6
    print(f"  Encoder frozen ({frozen_p:.1f}M params) for first "
          f"{cfg.ENCODER_FREEZE_EPOCHS} epochs")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max  = cfg.NUM_EPOCHS - cfg.ENCODER_FREEZE_EPOCHS,
        eta_min = 1e-6,
    )

    class_weights  = compute_class_weights(df_train["SegPath"].tolist(), cfg.DEVICE)
    loss_fn        = CombinedLoss(cfg, class_weights=class_weights)
    early_stopping = EarlyStopping(patience=cfg.EARLY_STOP_PATIENCE)

    best_val_dice = 0.0
    history = {k: [] for k in
               ["train_loss", "val_loss", "train_dice", "val_dice",
                "train_iou",  "val_iou",  "train_acc",  "val_acc"]}

    # ----------------------------------------------------------
    # [OUTPUT PATH] subfolder per model di dalam OUTPUT_DIR
    # ----------------------------------------------------------
    out_dir = os.path.join(cfg.OUTPUT_DIR, run_name)
    os.makedirs(out_dir, exist_ok=True)

    save_data_diagnostic(train_ds, out_dir, run_name, n=4)

    for epoch in range(1, cfg.NUM_EPOCHS + 1):

        # Unfreeze encoder setelah N epoch
        if epoch == cfg.ENCODER_FREEZE_EPOCHS + 1:
            for param in model.encoder.parameters():
                param.requires_grad = True
            is_eff = "efficientnet" in encoder_name
            unfreeze_lr = cfg.LR * (0.05 if is_eff else cfg.UNFREEZE_LR_FACTOR)
            optimizer = optim.AdamW(model.parameters(),
                                    lr=unfreeze_lr, weight_decay=cfg.WEIGHT_DECAY)
            scheduler = CosineAnnealingLR(optimizer,
                                          T_max=cfg.NUM_EPOCHS - epoch,
                                          eta_min=1e-6)
            print(f"  Epoch {epoch}: Encoder unfrozen — LR reset to {unfreeze_lr:.2e}")

        t_loss, t_m = run_epoch(model, train_loader, loss_fn,
                                cfg.DEVICE, optimizer, cfg.GRAD_CLIP)
        v_loss, v_m = run_epoch(model, val_loader, loss_fn, cfg.DEVICE)
        scheduler.step()

        for key in ["loss", "dice", "iou", "acc"]:
            k = key if key != "acc" else "accuracy"
            history[f"train_{key}"].append(t_loss if key == "loss" else t_m[k])
            history[f"val_{key}"].append(v_loss   if key == "loss" else v_m[k])

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch [{epoch:03d}/{cfg.NUM_EPOCHS}] "
              f"Loss {t_loss:.4f}/{v_loss:.4f}  "
              f"Dice {t_m['dice']:.4f}/{v_m['dice']:.4f}  "
              f"IoU {t_m['iou']:.4f}/{v_m['iou']:.4f}  "
              f"Acc {t_m['accuracy']:.4f}/{v_m['accuracy']:.4f}  "
              f"LR {lr_now:.2e}")

        if v_m["dice"] > best_val_dice:
            best_val_dice = v_m["dice"]
            torch.save(model.state_dict(),
                       os.path.join(out_dir, f"best_{run_name}.pth"))
            print(f"  -> Best model saved (Val Dice: {best_val_dice:.4f})")

        if epoch == 1 or epoch % cfg.SAVE_VIS_EVERY == 0:
            save_prediction_samples(model, val_loader, cfg.DEVICE,
                                    out_dir, run_name, n_samples=4, epoch=epoch)

        if early_stopping(v_m["dice"]):
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(no improvement for {cfg.EARLY_STOP_PATIENCE} epochs)")
            break

    torch.save(model.state_dict(), os.path.join(out_dir, f"final_{run_name}.pth"))
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    _save_curves(history, run_name, out_dir)

    # Final evaluation dengan best checkpoint
    print(f"\n  Loading best checkpoint for final evaluation…")
    model.load_state_dict(torch.load(
        os.path.join(out_dir, f"best_{run_name}.pth"), map_location=cfg.DEVICE))
    save_prediction_samples(model, val_loader, cfg.DEVICE,
                            out_dir, run_name, n_samples=6, epoch="best")
    dice_pc, iou_pc, conf_mat = compute_class_metrics(model, val_loader, cfg.DEVICE)
    save_class_metrics_table(dice_pc, iou_pc, run_name, out_dir)
    save_confusion_matrix(conf_mat, run_name, out_dir)

    print(f"\n  Best Val Dice [{run_name}]: {best_val_dice:.4f}")
    return history, best_val_dice


# ================================================================
# VISUALIZATION
# ================================================================
def label_to_rgb(label_map: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    for cls_id, color in LABEL_TO_COLOR.items():
        rgb[label_map == cls_id] = color
    rgb[label_map == 3] = (180, 180, 180)  # full_thickness lebih visible
    return rgb


_IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _denorm(t: torch.Tensor) -> np.ndarray:
    img = t.cpu().float().permute(1, 2, 0).numpy()
    return ((img * _IMG_STD + _IMG_MEAN).clip(0, 1) * 255).astype(np.uint8)


def _overlay(img: np.ndarray, pred: np.ndarray, alpha=0.45) -> np.ndarray:
    return (img * (1 - alpha) + pred * alpha).astype(np.uint8)


def _make_legend_handles():
    handles = []
    for name, hex_col in zip(CLASS_NAMES, VIS_COLORS):
        r, g, b = (int(hex_col[i:i+2], 16) / 255 for i in (1, 3, 5))
        handles.append(mpatches.Patch(facecolor=(r, g, b), edgecolor="black",
                                      linewidth=0.5, label=name))
    return handles


def save_data_diagnostic(dataset, out_dir, run_name, n=4):
    n = min(n, len(dataset))
    _, axes = plt.subplots(n, 2, figsize=(8, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    axes[0, 0].set_title("Raw image", fontsize=11, fontweight="bold")
    axes[0, 1].set_title("Ground-truth mask", fontsize=11, fontweight="bold")
    for i in range(n):
        img_t, mask_t = dataset[i]
        axes[i, 0].imshow(_denorm(img_t))
        axes[i, 1].imshow(label_to_rgb(mask_t.numpy().astype(np.uint8)))
        for col in range(2):
            axes[i, col].axis("off")
    fig = axes[0, 0].get_figure()
    fig.legend(handles=_make_legend_handles(), loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.01), fontsize=8, title="Class Legend")
    plt.suptitle(f"{run_name} — Data Diagnostic", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, f"{run_name}_data_diagnostic.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Data diagnostic saved → {path}")


def save_prediction_samples(model, val_loader, device, out_dir,
                             run_name, n_samples=6, epoch="final"):
    model.eval()
    imgs, gts, preds_rgb = [], [], []
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            pred_cls = torch.argmax(model(images), dim=1)
            for i in range(images.size(0)):
                if len(imgs) >= n_samples:
                    break
                imgs.append(_denorm(images[i]))
                gts.append(label_to_rgb(masks[i].cpu().numpy()))
                preds_rgb.append(label_to_rgb(pred_cls[i].cpu().numpy()))
            if len(imgs) >= n_samples:
                break

    n = len(imgs)
    _, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    for col, title in enumerate(["Original", "Ground Truth", "Prediction", "Overlay"]):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold")
    for row in range(n):
        axes[row, 0].imshow(imgs[row])
        axes[row, 1].imshow(gts[row])
        axes[row, 2].imshow(preds_rgb[row])
        axes[row, 3].imshow(_overlay(imgs[row], preds_rgb[row]))
        for col in range(4):
            axes[row, col].axis("off")
    fig = axes[0, 0].get_figure()
    fig.legend(handles=_make_legend_handles(), loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.01), fontsize=8, title="Class Legend")
    epoch_label = f"Epoch {epoch}" if isinstance(epoch, int) else epoch.capitalize()
    plt.suptitle(f"{run_name} — {epoch_label}", fontsize=14, fontweight="bold", y=1.005)
    plt.tight_layout()
    path = os.path.join(out_dir, f"{run_name}_pred_ep{epoch}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Prediction samples saved → {path}")


def compute_class_metrics(model, val_loader, device):
    model.eval()
    eps = 1e-8
    TP  = np.zeros(NUM_CLASSES)
    FP  = np.zeros(NUM_CLASSES)
    FN  = np.zeros(NUM_CLASSES)
    conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            pred_cls = torch.argmax(model(images), dim=1)
            p_np = pred_cls.cpu().numpy().ravel()
            t_np = masks.cpu().numpy().ravel()
            for t_c in range(NUM_CLASSES):
                for p_c in range(NUM_CLASSES):
                    conf_mat[t_c, p_c] += int(((t_np == t_c) & (p_np == p_c)).sum())
            for c in range(NUM_CLASSES):
                pc = (pred_cls == c).float()
                tc = (masks    == c).float()
                TP[c] += (pc * tc).sum().item()
                FP[c] += (pc * (1 - tc)).sum().item()
                FN[c] += ((1 - pc) * tc).sum().item()
    dice_pc = (2 * TP) / (2 * TP + FP + FN + eps)
    iou_pc  = TP / (TP + FP + FN + eps)
    return dice_pc, iou_pc, conf_mat


def save_class_metrics_table(dice_per_class, iou_per_class, run_name, out_dir):
    print(f"\n  Per-class metrics [{run_name}]:")
    print(f"  {'Class':<25} {'Dice':>8} {'IoU':>8}")
    print(f"  {'-'*43}")
    for c in range(NUM_CLASSES):
        marker = "  * (eval)" if c in EVAL_CLASSES else "    (ignored)"
        print(f"  {CLASS_NAMES[c]:<25} {dice_per_class[c]:>8.4f} "
              f"{iou_per_class[c]:>8.4f}{marker}")
    eval_dice = [dice_per_class[c] for c in EVAL_CLASSES]
    eval_iou  = [iou_per_class[c]  for c in EVAL_CLASSES]
    print(f"  {'-'*43}")
    print(f"  {'Mean (eval classes)':<25} {np.mean(eval_dice):>8.4f} "
          f"{np.mean(eval_iou):>8.4f}")

    x = np.arange(NUM_CLASSES)
    w = 0.35
    _, ax = plt.subplots(figsize=(12, 5))
    bars1 = ax.bar(x - w/2, dice_per_class, w, label="Dice", color="steelblue")
    bars2 = ax.bar(x + w/2, iou_per_class,  w, label="IoU",  color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.1)
    ax.set_title(f"{run_name} — Per-class Dice & IoU", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                f"{h:.3f}", ha="center", va="bottom", fontsize=7)
    for c in IGNORE_CLASSES:
        ax.axvspan(c - 0.5, c + 0.5, alpha=0.08, color="gray")
    plt.tight_layout()
    path = os.path.join(out_dir, f"{run_name}_class_metrics.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Per-class chart saved → {path}")


def save_confusion_matrix(conf_mat, run_name, out_dir):
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = conf_mat.astype(float) / conf_mat.sum(axis=1, keepdims=True)
        norm = np.nan_to_num(norm)
    _, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(CLASS_NAMES, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Ground Truth")
    ax.set_title(f"{run_name} — Confusion Matrix (normalised)", fontweight="bold")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, f"{norm[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if norm[i,j] > 0.6 else "black")
    plt.tight_layout()
    path = os.path.join(out_dir, f"{run_name}_confusion_matrix.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Confusion matrix saved → {path}")


# ================================================================
# PLOTTING
# ================================================================
def _save_curves(history, run_name, out_dir):
    _, axes = plt.subplots(2, 2, figsize=(14, 10))
    for (key, title), ax in zip(
        [("loss","Loss"),("dice","Dice"),("iou","IoU"),("acc","Accuracy")],
        axes.ravel()
    ):
        ax.plot(history[f"train_{key}"], label="Train")
        ax.plot(history[f"val_{key}"],   label="Val")
        ax.set_title(f"{run_name} — {title}")
        ax.set_xlabel("Epoch"); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{run_name}_curves.png"), dpi=150)
    plt.close()


def _save_comparison(results: dict, out_dir: str):
    names  = list(results.keys())
    values = list(results.values())
    colors = ["steelblue", "darkorange", "mediumseagreen",
              "mediumpurple", "crimson"][:len(names)]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(names, values, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_ylabel("Best Val Dice (mean eval classes)", fontsize=12)
    ax.set_title("Model Comparison — Best Validation Dice", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1); ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelsize=9)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.015,
                f"{v:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=11)

    # Garis target
    ax.axhline(y=0.5, color="red",  linestyle="--", alpha=0.5, label="Target min (0.50)")
    ax.axhline(y=0.7, color="green",linestyle="--", alpha=0.5, label="Target good (0.70)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(out_dir, "model_comparison.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Comparison chart saved → {path}")


# ================================================================
# MAIN
# ================================================================
def parse_args():
    valid_models = list(MODEL_REGISTRY.keys()) + ["all"]
    p = argparse.ArgumentParser(
        description="BurnVision — Burn Wound Segmentation Training",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Contoh penggunaan:
  python train.py --model unet_resnet34
  python train.py --model unetpp_efficientb4 --epochs 200 --batch_size 8
  python train.py --model deeplabv3p_effb4 --img_size 384
  python train.py --model all

Model tersedia:
  unet_resnet34       UNet + ResNet34        (Baseline)
  unet_densenet169    UNet + DenseNet169     (Baseline 2)
  unet_efficientb4    UNet + EfficientNet-B4
  unetpp_efficientb4  UNet++ + EfficientNet-B4  (Proposed)
  deeplabv3p_effb4    DeepLabV3+ + EfficientNet-B4  (Best candidate)
  all                 Train semua 5 model
        """
    )
    p.add_argument("--model",      choices=valid_models, default="all",
                   help="Model yang akan ditraining")
    p.add_argument("--data_dir",   default=None,
                   help="Override DATA_DIR (default: dari Config)")
    p.add_argument("--output_dir", default=None,
                   help="Override OUTPUT_DIR (default: dari Config)")
    p.add_argument("--img_size",   type=int,   default=None,
                   help="Ukuran gambar input (default: 320)")
    p.add_argument("--batch_size", type=int,   default=None,
                   help="Batch size (default: 8)")
    p.add_argument("--epochs",     type=int,   default=None,
                   help="Jumlah epoch (default: 150)")
    p.add_argument("--lr",         type=float, default=None,
                   help="Learning rate (default: 1e-4)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = Config()

    # Override dari CLI jika diberikan
    if args.data_dir:
        cfg.DATA_DIR  = args.data_dir
        cfg.RAW_DIR   = os.path.join(cfg.DATA_DIR, "RAW_DATA")
        cfg.SEG_DIR   = os.path.join(cfg.DATA_DIR, "SEGMENTED_DATA")
        cfg.XLSX_PATH = os.path.join(cfg.DATA_DIR, "FIlesPath.xlsx")
    if args.output_dir:  cfg.OUTPUT_DIR = args.output_dir
    if args.img_size:    cfg.IMG_SIZE   = args.img_size
    if args.batch_size:  cfg.BATCH_SIZE = args.batch_size
    if args.epochs:      cfg.NUM_EPOCHS = args.epochs
    if args.lr:          cfg.LR         = args.lr

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    print(f"\nDevice : {cfg.DEVICE}")
    if cfg.DEVICE == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"Img sz : {cfg.IMG_SIZE}  |  Batch: {cfg.BATCH_SIZE}  "
          f"|  Epochs: {cfg.NUM_EPOCHS}  |  LR: {cfg.LR:.0e}")
    print(f"Classes: {NUM_CLASSES} — {CLASS_NAMES}")

    df = load_pairs(cfg)
    if len(df) == 0:
        raise RuntimeError("No valid image pairs found. Cek DATA_DIR & struktur folder.")

    df_train, df_val = train_test_split(df, test_size=cfg.VAL_SPLIT,
                                        random_state=cfg.SEED)
    df_train = df_train.reset_index(drop=True)
    df_val   = df_val.reset_index(drop=True)

    # Pilih model yang akan ditraining
    if args.model == "all":
        models_to_run = MODEL_REGISTRY
    else:
        arch, enc = MODEL_REGISTRY[args.model]
        models_to_run = {args.model: (arch, enc)}

    print(f"\nModels to train ({len(models_to_run)}): {list(models_to_run.keys())}")

    results = {}
    for run_name, (arch_name, encoder_name) in models_to_run.items():
        _, best_dice = train_model(run_name, arch_name, encoder_name,
                                   df_train, df_val, cfg)
        results[run_name] = best_dice

    # Summary
    print(f"\n{'='*62}")
    print("  TRAINING COMPLETE — SUMMARY")
    print(f"{'='*62}")
    for name, dice in sorted(results.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(30 * dice)
        print(f"  {name:<25} Dice = {dice:.4f}  {bar}")

    best_model = max(results, key=results.get)
    print(f"\n  🏆 Best Model : {best_model}  (Dice = {results[best_model]:.4f})")

    with open(os.path.join(cfg.OUTPUT_DIR, "results_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results JSON  → {cfg.OUTPUT_DIR}/results_summary.json")

    if len(results) > 1:
        _save_comparison(results, cfg.OUTPUT_DIR)


if __name__ == "__main__":
    main()