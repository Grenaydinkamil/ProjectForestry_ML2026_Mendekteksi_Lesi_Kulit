#!/bin/bash
# =============================================================================
# scripts/setup_wsl_gpu.sh
# =============================================================================
# Menyiapkan WSL2 agar TensorFlow bisa memakai NVIDIA GPU.
#
# CARA PAKAI (dari PowerShell, di folder repo):
#   wsl --install -d Ubuntu-24.04                 # sekali saja, perlu admin
#   wsl -d Ubuntu-24.04 -u root -- bash scripts/setup_wsl_gpu.sh
#
# SETELAH ITU, training deep learning dijalankan lewat wrapper:
#   wsl -d Ubuntu-24.04 -- bash scripts/run_wsl_gpu.sh /opt/venvs/tf312/bin/jupyter lab
#
# -----------------------------------------------------------------------------
# CATATAN PENTING SOAL KETERBATASAN WINDOWS
# -----------------------------------------------------------------------------
# TensorFlow 2.11 ke atas TIDAK bisa memakai GPU di Windows native.
# Dokumentasi resmi (tensorflow.org/install/pip, bagian "Windows Native"):
#   "TensorFlow 2.10 was the last TensorFlow release that supported GPU on
#    native-Windows. Starting with TensorFlow 2.11, you will need to install
#    TensorFlow in WSL2..."
#
# Opsi yang realistis:
#   (a) WSL2 + GPU   -> dipakai skrip ini, cepat, gratis
#   (b) CPU Windows  -> selalu bisa, tapi lambat untuk MobileNetV2
#   (c) Google Colab -> GPU gratis, tapi butuh akun + upload data
#
# -----------------------------------------------------------------------------
# MENGAPA SKRIP INI PERLU ADA (hasil debugging nyata di mesin ini)
# -----------------------------------------------------------------------------
# Cara install CUDA untuk TensorFlow di Linux:
#   pip install "tensorflow[and-cuda]==2.21.0"
#
# Paket pip nvidia-* menaruh library .so (libcudart.so.12, libcublas.so.12,
# libcudnn.so.9, dst) di DALAM site-packages/nvidia/<paket>/lib.
#
# TensorFlow mencari library itu saat runtime dengan urutan:
#   RPATH objek -> LD_LIBRARY_PATH -> ld.so.cache -> /lib -> /usr/lib
#
# Kalau path itu tidak ada di sana, TensorFlow mencetak:
#   "Cannot dlopen some GPU libraries... Skipping registering GPU devices"
# dan SENGAJA diam-diam jatuh ke CPU (tidak error, cuma warning).
#
# Empat pendekatan yang sudah dicoba dan HASILNYA:
#   1. Symlink .so ke direktori TensorFlow    -> GAGAL
#      (direktori itu tidak ada di jalur loader)
#   2. /etc/ld.so.conf.d + ldconfig            -> GAGAL
#      (ldconfig di WSL hanya mengindeks path standar, bukan /opt)
#   3. /etc/profile.d/tf-gpu.sh                -> HANYA untuk shell LOGIN
#      (perintah `wsl -- bash -c` memakai shell non-login)
#   4. Seksi [env] di /etc/wsl.conf            -> TIDAK diteruskan ke proses
#      python yang dijalankan WSL
#
# YANG BEKERJA: shell LOGIN (`bash -lc`), karena /etc/profile.d/tf-gpu.sh
# di-source di sana. Wrapper scripts/run_wsl_gpu.sh menjamin hal ini.
# =============================================================================
set -uo pipefail

# --- Konfigurasi ---
DISTRO="${TF_WSL_DISTRO:-Ubuntu-26.04}"
VENV="${TF_VENV:-/opt/venvs/tf312}"
TF_VERSION="${TF_VERSION:-2.21.0}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
SHORT_LIB="${TF_GPU_LIB_DIR:-/opt/tf-gpu-lib}"

echo "======================================================================"
echo " SETUP WSL2 + TENSORFLOW GPU"
echo "   Distro : $DISTRO"
echo "   Venv   : $VENV"
echo "   TF     : $TF_VERSION (Python $PYTHON_VERSION)"
echo "======================================================================"

# ---------------------------------------------------------------------------
# 1. PASANG UV + PYTHON
# ---------------------------------------------------------------------------
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  echo ""
  echo "[1/5] Menginstal uv (manajer Python cepat)..."
  curl -LsSf https://astral.sh/uv/install.sh -o /tmp/uv.sh
  sh /tmp/uv.sh >/dev/null 2>&1
fi
echo "[1/5] uv: $(uv --version)"

# TensorFlow 2.21 hanya mendukung Python 3.10-3.13 (bukan 3.14)
uv python install "$PYTHON_VERSION"
if [ ! -d "$VENV" ]; then
  # --seed memastikan pip ikut terpasang (uv venv tanpa --seed tidak punya pip)
  uv venv --python "$PYTHON_VERSION" --seed "$VENV"
fi
echo "      Python: $($VENV/bin/python --version)"

# ---------------------------------------------------------------------------
# 2. PASANG TENSORFLOW DENGAN CUDA
# ---------------------------------------------------------------------------
echo "[2/5] Memasang tensorflow[and-cuda]==$TF_VERSION (bisa beberapa menit)..."
"$VENV/bin/python" -m pip install --upgrade pip --quiet
"$VENV/bin/pip" install "tensorflow[and-cuda]==$TF_VERSION" 2>&1 | tail -3

SP=$("$VENV/bin/python" -c "import site; print(site.getsitepackages()[0])")

# ---------------------------------------------------------------------------
# 3. BUAT DIREKTORI SYMLINK RINGKAS
# ---------------------------------------------------------------------------
echo "[3/5] Membuat direktori library ringkas $SHORT_LIB ..."
rm -rf "$SHORT_LIB"
mkdir -p "$SHORT_LIB"
count=0
while IFS= read -r d; do
  [ -z "$d" ] && continue
  [ -d "$d" ] || continue
  for so in "$d"/*.so*; do
    [ -e "$so" ] || continue
    ln -sf "$so" "$SHORT_LIB/$(basename "$so")" 2>/dev/null && count=$((count + 1))
  done
done < <(find "$SP/nvidia" -type d -name lib 2>/dev/null)
echo "      $count symlink dibuat"

# ptxas: compiler PTX, dipakai untuk JIT kernel pada GPU baru
Ptxas=$(find "$SP/nvidia" -name ptxas -type f 2>/dev/null | head -1)
if [ -n "$Ptxas" ]; then
  mkdir -p "$VENV/bin"
  ln -sf "$Ptxas" "$VENV/bin/ptxas"
  echo "      ptxas -> $Ptxas"
fi

# ---------------------------------------------------------------------------
# 4. PASANG PROFILE UNTUK SHELL LOGIN
# ---------------------------------------------------------------------------
echo "[4/5] Memasang /etc/profile.d/tf-gpu.sh ..."
cat > /etc/profile.d/tf-gpu.sh <<PROFILE
# Library NVIDIA untuk TensorFlow di WSL2.
# PENTING: hanya dibaca shell LOGIN. Untuk perintah non-login, pakai
# scripts/run_wsl_gpu.sh yang memaksa login shell.
export LD_LIBRARY_PATH="$SHORT_LIB:/usr/lib/wsl/lib:\${LD_LIBRARY_PATH:-}"
export PATH="$VENV/bin:\$PATH"
PROFILE
chmod +x /etc/profile.d/tf-gpu.sh
echo "      [OK] /etc/profile.d/tf-gpu.sh"

# ---------------------------------------------------------------------------
# 5. VERIFIKASI
# ---------------------------------------------------------------------------
echo "[5/5] Verifikasi GPU..."
export LD_LIBRARY_PATH="$SHORT_LIB:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"

"$VENV/bin/python" - <<'PY' 2>&1 | grep -v "oneDNN\|absl::\|cpu_feature\|To enable"
import tensorflow as tf

print("TensorFlow :", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
print("GPU devices:", gpus)

if not gpus:
    print("HASIL: GPU BELUM TERSEDETEK")
    raise SystemExit(1)

for g in gpus:
    print("  detail:", tf.config.experimental.get_device_details(g))

# Uji komputasi NYATA. Tanpa ini, TensorFlow bisa mendaftarkan GPU tapi
# tetap menjalankan operasi di CPU.
with tf.device("/GPU:0"):
    a = tf.random.normal((4096, 4096))
    b = tf.matmul(a, a)
print("Tes matmul GPU OK, mean:", float(tf.reduce_mean(b)))
print("HASIL: GPU SIAP DIPAKAI")
PY

echo ""
echo "======================================================================"
echo " SETUP SELESAI"
echo ""
echo " Cara menjalankan notebook training di WSL (dari PowerShell):"
echo "   wsl -d $DISTRO -- bash scripts/run_wsl_gpu.sh \\"
echo "       /opt/venvs/tf312/bin/jupyter lab"
echo ""
echo " CATATAN: RTX 50 series (Blackwell, sm_120) tidak punya binary kernel"
echo " di TensorFlow 2.21, jadi kernel di-JIT dari PTX. Run PERTAMA bisa"
echo " 10-30 menit lambat; setelah itu di-cache di ~/.nv/ComputeCache."
echo "======================================================================"
