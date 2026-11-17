#!/bin/bash
# =============================================================================
# scripts/run_notebook_wsl.sh
# =============================================================================
# Menjalankan notebook di WSL2 dengan GPU, dari PowerShell.
#
# CARA PAKAI:
#   wsl -d Ubuntu-26.04 -- bash scripts/run_notebook_wsl.sh 05_cnn
#   wsl -d Ubuntu-26.04 -- bash scripts/run_notebook_wsl.sh 06_mobilenetv2
#   wsl -d Ubuntu-26.04 -- bash scripts/run_notebook_wsl.sh 04_svm_rf
#
# Mengapa perlu script ini?
#   1. Mengaktifkan GPU (LD_LIBRARY_PATH ke library NVIDIA).
#   2. Menunjuk lokasi data ke disk NATIVE WSL ( jauh lebih cepat dari /mnt/).
#   3. Memastikan paket yang dibutuhkan sudah terpasang.
#
# CATATAN PENTING SOAL REPRODUKSIBILITAS:
#   Notebook ini juga bisa dijalankan di Windows (CPU) dengan perintah:
#       jupyter lab
#   Hasilnya identik secara metodologis (seed sama, data sama), hanya
#   lebih lambat. Jadi mahasiswa TIDAK wajib pakai WSL2 untuk bisa
#   mereproduksi, tapi jauh lebih cepat kalau memakainya.
#
# Variabel lingkungan yang dipakai:
#   SKIN_DATA_ROOT -> lokasi data mentah di disk WSL (dibuat oleh
#                     scripts/setup_wsl_gpu.sh atau salinan manual)
# =============================================================================
set -uo pipefail

REPO="/mnt/m/GrenPlums Roses/Program32bit/Forestrypansy/ML2026_Kelas_3_Kelompok_1_LesiKulit"
VENV=/opt/venvs/tf312
LOCAL_RAW="${SKIN_DATA_ROOT:-/mnt/f/ml_project_data/raw}"

if [ $# -lt 1 ]; then
  echo "Pakai: bash scripts/run_notebook_wsl.sh <nama_notebook>"
  echo "Contoh: bash scripts/run_notebook_wsl.sh 05_cnn"
  exit 1
fi

NAMA="$1"
NOTEBOOK="$REPO/notebooks/${NAMA}.ipynb"

if [ ! -f "$NOTEBOOK" ]; then
  echo "[X] Notebook tidak ditemukan: $NOTEBOOK"
  exit 1
fi

# --- 1. Aktifkan GPU ---
export LD_LIBRARY_PATH="/opt/tf-gpu-lib:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
export PATH="$VENV/bin:$PATH"
export SKIN_DATA_ROOT="$LOCAL_RAW"

# --- 2. Pastikan paket ---
if ! "$VENV/bin/python" -c "import pandas, sklearn, skimage" 2>/dev/null; then
  echo "[setup] Memasang paket yang kurang..."
  "$VENV/bin/pip" install --index-url https://pypi.org/simple \
      pandas scikit-learn scikit-image matplotlib seaborn joblib \
      pillow tqdm jupyterlab ipykernel nbconvert 2>&1 | tail -2
fi

cd "$REPO" || exit 1

echo "=========================================================="
echo " Menjalankan notebook: ${NAMA}.ipynb"
echo " Data   : $SKIN_DATA_ROOT"
echo " GPU    : $($VENV/bin/python -c "
import tensorflow as tf
g = tf.config.list_physical_devices('GPU')
print(g[0].name if g else 'TIDAK ADA')
" 2>/dev/null | tail -1)"
echo "=========================================================="

# --- 3. Jalankan notebook (dengan --inplace agar hasil tersimpan) ---
exec "$VENV/bin/python" -m jupyter nbconvert \
  --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=14400 \
  "$NOTEBOOK"
