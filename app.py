"""
BurnVision Flask API
Menghubungkan GUI HTML dengan predict.py untuk inferensi nyata.

Install dependencies:
    pip install flask flask-cors

Jalankan:
    python app.py

Lalu buka browser: http://localhost:5000
"""

import os
import sys
import json
import base64
import tempfile
import traceback
from pathlib import Path
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS

# ── Coba import library ML (opsional, fallback ke simulasi jika tidak ada) ──
try:
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
    ML_AVAILABLE = True
    print("[BurnVision] ✅ ML libraries loaded (numpy, torch, smp, albumentations)")
except ImportError as e:
    ML_AVAILABLE = False
    print(f"[BurnVision] ⚠️  ML library tidak tersedia: {e}")
    print("[BurnVision]    Mode simulasi aktif (untuk testing GUI)")

# ── Konstanta (harus sama dengan train.py) ──────────────────────────────────
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

CLASS_COLORS = [
    [0xAE, 0xFC, 0xAE],  # 0 normal
    [0xF8, 0xBD, 0xC9],  # 1 superficial
    [0xFF, 0x00, 0x7C],  # 2 mid-deep
    [0xFD, 0xFD, 0xFD],  # 3 full thickness
    [0xF8, 0xF8, 0x10],  # 4 medical_object
    [0x1B, 0xCE, 0xF0],  # 5 tangan_pemeriksa
    [0x23, 0x23, 0xFA],  # 6 background
]

WOUND_CLASSES  = [1, 2, 3]
IGNORE_CLASSES = [4, 5, 6]

ENCODERS = {
    "unet":             "resnet34",
    "densenet169_unet": "densenet169",
}

# ── Flask app ────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

OUTPUT_DIR = Path("hasil")
OUTPUT_DIR.mkdir(exist_ok=True)

# Cache model agar tidak reload setiap request
_model_cache = {}


# ── Helper functions ─────────────────────────────────────────────────────────

def load_model_cached(model_type: str, model_path: str, device: str):
    """Load model sekali, cache untuk request berikutnya."""
    key = f"{model_type}:{model_path}"
    if key not in _model_cache:
        encoder = ENCODERS[model_type]
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=None,
            in_channels=3,
            classes=NUM_CLASSES,
            activation=None,
        )
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        _model_cache[key] = model
        print(f"[BurnVision] Model loaded & cached: {key}")
    return _model_cache[key]


def preprocess_image(image_bytes: bytes, img_size: int):
    """Konversi bytes gambar → tensor siap inferensi + original numpy array."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Gagal membaca gambar. Pastikan format JPG/PNG.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original = image.copy()

    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    tensor = transform(image=image)["image"].unsqueeze(0)
    return tensor, original


def run_inference(model, tensor, device: str):
    """Jalankan forward pass, kembalikan label map (H, W) nilai 0–6."""
    tensor = tensor.to(device)
    with torch.no_grad():
        logits    = model(tensor)
        label_map = torch.argmax(logits, dim=1)
    return label_map.squeeze(0).cpu().numpy().astype(np.uint8)


def label_to_color(label_map: np.ndarray) -> np.ndarray:
    colors = np.array(CLASS_COLORS, dtype=np.uint8)
    return colors[label_map]


def compute_stats(label_map: np.ndarray) -> dict:
    total_px   = label_map.size
    ignore_px  = sum(int((label_map == c).sum()) for c in IGNORE_CLASSES)
    skin_px    = total_px - ignore_px
    wound_px   = sum(int((label_map == c).sum()) for c in WOUND_CLASSES)
    wound_pct  = round(wound_px / skin_px * 100, 2) if skin_px > 0 else 0.0

    per_class = []
    for c in range(NUM_CLASSES):
        px  = int((label_map == c).sum())
        pct = round(px / total_px * 100, 2)
        per_class.append({"class": c, "name": CLASS_NAMES[c], "pixels": px, "pct": pct})

    return {
        "total_pixels":           total_px,
        "evaluable_skin_pixels":  skin_px,
        "total_wound_pixels":     wound_px,
        "wound_pct_of_skin":      wound_pct,
        "per_class":              per_class,
    }


def numpy_to_b64(arr: np.ndarray, ext: str = ".png") -> str:
    """Konversi numpy RGB array → base64 string untuk dikirim ke frontend."""
    arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    ok, buf  = cv2.imencode(ext, arr_bgr)
    if not ok:
        raise RuntimeError("Gagal encode gambar ke PNG")
    return base64.b64encode(buf).decode("utf-8")


def simulate_inference(img_size: int) -> np.ndarray:
    """
    Fallback simulasi jika ML library tidak tersedia.
    Menghasilkan label map acak yang lebih realistis (bukan pola lingkaran).
    """
    np_mod = __import__("numpy")
    rng = np_mod.random.default_rng(42)
    label_map = np_mod.zeros((img_size, img_size), dtype=np_mod.uint8)

    # Simulasi: sebagian besar normal, patch luka di tengah
    label_map[:] = 0  # normal
    cx, cy = img_size // 2, img_size // 2
    r1, r2, r3 = img_size // 5, img_size // 3, img_size * 2 // 5

    for y in range(img_size):
        for x in range(img_size):
            d = ((x - cx)**2 + (y - cy)**2) ** 0.5
            if d < r1:
                label_map[y, x] = 3   # full thickness (inti)
            elif d < r2:
                label_map[y, x] = 2   # mid-deep
            elif d < r3:
                label_map[y, x] = 1   # superficial

    # Tambah noise realistis
    noise_mask = rng.random((img_size, img_size)) < 0.05
    label_map[noise_mask] = rng.integers(0, 4, size=noise_mask.sum())

    return label_map


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve index.html"""
    return send_file("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    """Cek status server dan ketersediaan GPU."""
    info = {
        "status": "ok",
        "ml_available": ML_AVAILABLE,
        "timestamp": datetime.now().isoformat(),
    }
    if ML_AVAILABLE:
        info["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
    return jsonify(info)


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Endpoint utama inferensi.

    Form data:
        image      : file gambar (JPG/PNG)
        model_type : 'unet' atau 'densenet169_unet'
        model_path : path ke file .pth
        img_size   : integer (default 256)

    Returns JSON:
        {
          "success": true,
          "seg_map_b64": "...",     # PNG base64 segmentation map
          "overlay_b64": "...",     # PNG base64 overlay
          "stats": { ... },
          "log": [ ... ],
          "device": "cuda/cpu"
        }
    """
    logs = []

    try:
        # ── Validasi input ────────────────────────────────────────────────
        if "image" not in request.files:
            return jsonify({"success": False, "error": "File gambar tidak ditemukan"}), 400

        file       = request.files["image"]
        model_type = request.form.get("model_type", "unet")
        model_path = request.form.get("model_path", "")
        img_size   = int(request.form.get("img_size", 256))

        if model_type not in ENCODERS:
            return jsonify({"success": False, "error": f"model_type tidak valid: {model_type}"}), 400

        image_bytes = file.read()
        filename    = file.filename or "image.jpg"

        logs.append({"type": "info",    "msg": f"$ python predict.py --image {filename} --model_type {model_type}"})
        logs.append({"type": "dim",     "msg": f"  Image size : {img_size}px"})
        logs.append({"type": "dim",     "msg": f"  File size  : {len(image_bytes)/1024:.1f} KB"})

        # ── Mode ML nyata ─────────────────────────────────────────────────
        if ML_AVAILABLE and model_path and os.path.exists(model_path):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logs.append({"type": "dim",     "msg": f"  Device     : {device}"})
            logs.append({"type": "dim",     "msg": f"  Encoder    : {ENCODERS[model_type]}"})
            logs.append({"type": "warning", "msg": "  Loading model weights..."})

            model = load_model_cached(model_type, model_path, device)
            logs.append({"type": "success", "msg": "  Model loaded ✓"})

            logs.append({"type": "warning", "msg": "  Preprocessing image..."})
            tensor, original = preprocess_image(image_bytes, img_size)

            logs.append({"type": "warning", "msg": "  Running forward pass..."})
            label_map = run_inference(model, tensor, device)
            logs.append({"type": "success", "msg": "  Inference complete ✓"})

        # ── Mode simulasi (fallback) ───────────────────────────────────────
        else:
            device = "cpu (simulation)"
            if not ML_AVAILABLE:
                logs.append({"type": "warning", "msg": "  ⚠ ML libraries tidak tersedia — mode simulasi"})
            elif not model_path:
                logs.append({"type": "warning", "msg": "  ⚠ Model path kosong — mode simulasi"})
            elif not os.path.exists(model_path):
                logs.append({"type": "warning", "msg": f"  ⚠ File tidak ditemukan: {model_path} — mode simulasi"})

            logs.append({"type": "dim", "msg": "  Generating simulated segmentation..."})

            # Decode gambar asli untuk overlay
            nparr    = np.frombuffer(image_bytes, np.uint8)
            original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB) if original is not None else \
                       np.zeros((img_size, img_size, 3), dtype=np.uint8)
            original = cv2.resize(original, (img_size, img_size))

            label_map = simulate_inference(img_size)
            logs.append({"type": "success", "msg": "  Simulation complete ✓"})

        # ── Buat output gambar ────────────────────────────────────────────
        orig_resized = cv2.resize(original, (img_size, img_size))
        color_mask   = label_to_color(label_map)
        overlay      = (orig_resized * 0.5 + color_mask * 0.5).astype(np.uint8)

        seg_b64     = numpy_to_b64(color_mask)
        overlay_b64 = numpy_to_b64(overlay)

        # ── Hitung statistik ──────────────────────────────────────────────
        stats = compute_stats(label_map)
        logs.append({"type": "dim",     "msg": f"  wound_pct_of_skin : {stats['wound_pct_of_skin']}%"})

        for cls in stats["per_class"][:4]:
            logs.append({"type": "dim", "msg": f"  {cls['name']:<30}: {cls['pct']}%"})

        # Simpan file hasil
        base_name   = Path(filename).stem
        result_path = OUTPUT_DIR / f"{base_name}_result_{datetime.now().strftime('%H%M%S')}.png"

        # Simpan overlay sebagai file
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(result_path), overlay_bgr)
        logs.append({"type": "success", "msg": f"  Results saved → {result_path}"})

        return jsonify({
            "success":      True,
            "seg_map_b64":  seg_b64,
            "overlay_b64":  overlay_b64,
            "stats":        stats,
            "log":          logs,
            "device":       device,
            "result_path":  str(result_path),
        })

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[BurnVision ERROR]\n{tb}")
        logs.append({"type": "error", "msg": f"  ERROR: {str(e)}"})
        return jsonify({
            "success": False,
            "error":   str(e),
            "log":     logs,
        }), 500


@app.route("/api/check_model", methods=["POST"])
def check_model():
    """Cek apakah file model .pth ada di path yang diberikan."""
    data       = request.get_json() or {}
    model_path = data.get("model_path", "")
    exists     = os.path.exists(model_path)
    size_mb    = round(os.path.getsize(model_path) / 1024 / 1024, 1) if exists else 0
    return jsonify({
        "exists":   exists,
        "path":     model_path,
        "size_mb":  size_mb,
    })


if __name__ == "__main__":
    print("=" * 55)
    print("  BurnVision Flask API")
    print("  http://localhost:5000")
    print("=" * 55)
    if not ML_AVAILABLE:
        print("  ⚠  Mode simulasi (install torch/smp untuk inferensi nyata)")
    app.run(debug=True, host="0.0.0.0", port=5000)
