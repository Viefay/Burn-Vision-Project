"""
Burn Wound Segmentation - Inference Script (7-class)
Usage:
  python predict.py --image foto_luka.jpg --model_type unet --model_path outputs/unet/best_unet.pth
  python predict.py --image foto_luka.jpg --model_type densenet169_unet --model_path outputs/densenet169_unet/best_densenet169_unet.pth
"""

import os
import argparse
import numpy as np
import cv2
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ================================================================
# CLASS DEFINITIONS (harus sama persis dengan train.py)
# ================================================================
CLASS_NAMES = [
    "normal",                # 0
    "superficial_thickness", # 1
    "mid_deep_thickness",    # 2
    "full_thickness",        # 3
    "medical_object",        # 4
    "tangan_pemeriksa",      # 5
    "background",            # 6
]
NUM_CLASSES = len(CLASS_NAMES)

# Warna RGB untuk visualisasi tiap kelas (sesuai COLOR_MAP di train.py)
CLASS_COLORS = np.array([
    [0xAE, 0xFC, 0xAE],  # 0 normal             — hijau muda
    [0xF8, 0xBD, 0xC9],  # 1 superficial         — merah muda
    [0xFF, 0x00, 0x7C],  # 2 mid-deep            — merah terang
    [0xFD, 0xFD, 0xFD],  # 3 full thickness      — putih
    [0xF8, 0xF8, 0x10],  # 4 medical_object      — kuning
    [0x1B, 0xCE, 0xF0],  # 5 tangan_pemeriksa    — cyan
    [0x23, 0x23, 0xFA],  # 6 background          — biru
], dtype=np.uint8)

# Kelas luka yang relevan untuk statistik
WOUND_CLASSES = [1, 2, 3]  # superficial, mid-deep, full thickness
IGNORE_CLASSES = [4, 5, 6]

ENCODERS = {
    "unet":             "resnet34",
    "densenet169_unet": "densenet169",
}


def load_model(model_type: str, model_path: str, device: str):
    encoder = ENCODERS[model_type]
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,   # 7 kelas
        activation=None,
    )
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def preprocess(image_path: str, img_size: int = 256):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original = image.copy()

    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    tensor = transform(image=image)["image"].unsqueeze(0)
    return tensor, original


def predict(model, tensor, device: str):
    """Kembalikan label map (H, W) dengan nilai 0–6."""
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)                        # (1, 7, H, W)
        label_map = torch.argmax(logits, dim=1)       # (1, H, W)
    return label_map.squeeze(0).cpu().numpy().astype(np.uint8)  # (H, W)


def label_to_color(label_map: np.ndarray) -> np.ndarray:
    """Konversi label map (H, W) → gambar RGB (H, W, 3)."""
    color_img = CLASS_COLORS[label_map]  # vectorized indexing
    return color_img


def compute_burn_stats(label_map: np.ndarray) -> dict:
    total_px = label_map.size
    ignore_mask = np.zeros(label_map.shape, dtype=bool)
    for c in IGNORE_CLASSES:
        ignore_mask |= (label_map == c)
    skin_px = int((~ignore_mask).sum())

    stats = {"total_pixels": total_px, "evaluable_skin_pixels": skin_px}
    for c in range(NUM_CLASSES):
        px = int((label_map == c).sum())
        pct = px / total_px * 100
        stats[f"class_{c}_{CLASS_NAMES[c]}_pixels"] = px
        stats[f"class_{c}_{CLASS_NAMES[c]}_pct"]    = round(pct, 2)

    wound_px = sum(int((label_map == c).sum()) for c in WOUND_CLASSES)
    stats["total_wound_pixels"] = wound_px
    stats["wound_pct_of_skin"]  = round(wound_px / skin_px * 100, 2) if skin_px > 0 else 0.0
    return stats


def save_result(original, label_map, stats, output_path: str):
    orig_resized  = cv2.resize(original, (label_map.shape[1], label_map.shape[0]))
    color_mask    = label_to_color(label_map)

    # Overlay: blend original + color mask
    overlay = (orig_resized * 0.5 + color_mask * 0.5).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(orig_resized)
    axes[0].set_title("Input Image", fontsize=13)
    axes[0].axis("off")

    axes[1].imshow(color_mask)
    axes[1].set_title("Segmentation Map", fontsize=13)
    axes[1].axis("off")
    legend_patches = [
        Patch(color=CLASS_COLORS[c] / 255.0, label=f"{c}: {CLASS_NAMES[c]}")
        for c in range(NUM_CLASSES)
    ]
    axes[1].legend(handles=legend_patches, loc="lower right",
                   fontsize=7, framealpha=0.8)

    axes[2].imshow(overlay)
    wound_pct = stats.get("wound_pct_of_skin", 0)
    axes[2].set_title(f"Overlay\nWound area: {wound_pct:.2f}% of skin", fontsize=13)
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_mask_png(label_map: np.ndarray, output_path: str):
    """Simpan label map sebagai PNG grayscale (nilai 0–6)."""
    cv2.imwrite(output_path, label_map)


def main():
    parser = argparse.ArgumentParser(description="Burn Wound Segmentation Inference (7-class)")
    parser.add_argument("--image",      required=True,  help="Path to input image (JPG/PNG)")
    parser.add_argument("--model_type", required=True,  choices=["unet", "densenet169_unet"])
    parser.add_argument("--model_path", required=True,  help="Path to trained .pth weights file")
    parser.add_argument("--output_dir", default=".",    help="Directory to save results")
    parser.add_argument("--img_size",   type=int, default=256)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device     : {device}")
    print(f"Model type : {args.model_type}")
    print(f"Weights    : {args.model_path}")
    print(f"Image      : {args.image}")

    model            = load_model(args.model_type, args.model_path, device)
    tensor, original = preprocess(args.image, args.img_size)
    label_map        = predict(model, tensor, device)
    stats            = compute_burn_stats(label_map)

    os.makedirs(args.output_dir, exist_ok=True)
    base_name    = os.path.splitext(os.path.basename(args.image))[0]
    result_path  = os.path.join(args.output_dir, f"{base_name}_result.png")
    mask_path    = os.path.join(args.output_dir, f"{base_name}_mask.png")

    save_result(original, label_map, stats, result_path)
    save_mask_png(label_map, mask_path)

    print(f"\nResults saved:")
    print(f"  Visualization : {result_path}")
    print(f"  Label map     : {mask_path}")
    print(f"\nSegmentation stats:")
    for k, v in stats.items():
        print(f"  {k:<45}: {v}")


if __name__ == "__main__":
    main()
