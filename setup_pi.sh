#!/bin/bash
# ============================================================
# Raspberry Pi 4 — Yorgunluk Tespiti Kurulum Scripti
# Hedef: Miniforge / conda ortamı "proje_ortami", Python 3.10.20
# Kullanım: bash setup_pi.sh
# ============================================================

set -e

CONDA_ENV="proje_ortami"
PROJE_DIR="$(cd "$(dirname "$0")" && pwd)"

# conda komutlarını aktif et
CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniforge3")"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "=================================================="
echo "  Yorgunluk Tespiti — Pi 4 Kurulum"
echo "  Conda env : $CONDA_ENV"
echo "  Python    : $(python --version 2>&1)"
echo "  Proje     : $PROJE_DIR"
echo "=================================================="

# ----------------------------------------------------------
# 1. SİSTEM PAKETLERİ
#    libcamera → picamera2'nin bağımlılığı (pip sürümü için de gerekli)
#    lgpio     → gpiozero GPIO arka ucu
# ----------------------------------------------------------
echo ""
echo "[1/5] Sistem paketleri kuruluyor (sudo gerektirir)..."

sudo apt-get update -qq

sudo apt-get install -y \
    libcap-dev \
    python3-lgpio \
    libatlas-base-dev \
    libportaudio2 \
    v4l-utils

echo "  [OK] Sistem paketleri hazır."

# ----------------------------------------------------------
# 2. OPENCV ÇAKIŞMASI
#    opencv-python (4.8) + opencv-contrib-python (4.11) → hata
#    opencv-contrib-python zaten opencv-python'u kapsar → sadece contrib yeter
# ----------------------------------------------------------
echo ""
echo "[2/5] OpenCV çakışması temizleniyor..."

pip uninstall -y opencv-python 2>/dev/null && echo "  opencv-python (4.8) kaldırıldı." || echo "  opencv-python zaten yoktu."

echo "  [OK] Yalnızca opencv-contrib-python==4.11.0.86 bırakıldı."

# ----------------------------------------------------------
# 3. EKSİK PAKETLER — picamera2 ve gpiozero
# ----------------------------------------------------------
echo ""
echo "[3/5] Eksik paketler kuruluyor..."

pip install "gpiozero>=2.0" lgpio
echo "  [OK] gpiozero + lgpio kuruldu."

# Kamera V4L2 erişim kontrolü
echo ""
echo "  Kamera aygıtları:"
v4l2-ctl --list-devices 2>/dev/null || echo "  (v4l2-ctl çalıştırılamadı — kamera takılı mı?)"

# ----------------------------------------------------------
# 4. FLATBUFFERS GÜNCELLEMESİ
#    Mevcut sürüm (20181003210633) tflite-runtime 2.13 ile uyumsuz olabilir
# ----------------------------------------------------------
echo ""
echo "[4/5] Flatbuffers güncelleniyor..."

pip install "flatbuffers>=23.5.26"
echo "  [OK] Flatbuffers güncellendi: $(python -c 'import flatbuffers; print(flatbuffers.__version__)')"

# ----------------------------------------------------------
# 5. DOĞRULAMA
# ----------------------------------------------------------
echo ""
echo "[5/5] Kurulum doğrulanıyor..."

python - <<'PYCHECK'
import sys

def kontrol(isim, import_adi, versiyon_attr="__version__"):
    try:
        m = __import__(import_adi)
        ver = getattr(m, versiyon_attr, "?")
        print(f"  {'OK':<6} {isim:<22} {ver}")
        return True
    except Exception as e:
        print(f"  {'HATA':<6} {isim:<22} {e}")
        return False

print(f"\n  Python: {sys.version.split()[0]}\n")

kontrol("cv2 (kamera arayüzü)",    "cv2")
kontrol("gpiozero",                "gpiozero")
kontrol("opencv-contrib-python",   "cv2",        "__version__")
kontrol("mediapipe",               "mediapipe")
kontrol("numpy",                   "numpy")
kontrol("flatbuffers",             "flatbuffers")
kontrol("protobuf",                "google.protobuf", "DESCRIPTOR.syntax")

# tflite ayrı kontrol
try:
    from tflite_runtime.interpreter import Interpreter
    print(f"  {'OK':<6} tflite-runtime")
except ImportError:
    try:
        from tensorflow.lite.python.interpreter import Interpreter
        print(f"  {'OK':<6} tflite (tensorflow)")
    except ImportError as e:
        print(f"  {'HATA':<6} tflite-runtime         {e}")

# opencv-python çakışma kontrolü
try:
    import pkg_resources
    cv_pkgs = [p for p in pkg_resources.working_set if "opencv" in p.project_name]
    if len(cv_pkgs) > 1:
        print(f"\n  UYARI: Birden fazla OpenCV paketi var:")
        for p in cv_pkgs:
            print(f"    - {p.project_name}=={p.version}")
    else:
        print(f"\n  OpenCV çakışması yok.")
except Exception:
    pass
PYCHECK

echo ""
echo "=================================================="
echo "  Kurulum tamamlandı!"
echo ""
echo "  Çalıştırmak için:"
echo "    conda activate $CONDA_ENV"
echo "    python $PROJE_DIR/src/main.py"
echo "=================================================="
