#!/bin/bash
# =============================================================================
# scripts/run_wsl_gpu.sh
# =============================================================================
# Membungkus perintah agar TensorFlow di WSL2 bisa memakai NVIDIA GPU.
#
# -----------------------------------------------------------------------------
# MENGAPA ADA WRAPPER INI (hasil debugging nyata, bukan teori)
# -----------------------------------------------------------------------------
# TensorFlow mencari library CUDA dengan urutan:
#     RPATH objek  ->  LD_LIBRARY_PATH  ->  ld.so.cache  ->  /lib  ->  /usr/lib
#
# Paket pip "tensorflow[and-cuda]" menaruh library .so di dalam
# site-packages/nvidia/<paket>/lib. paths itu:
#   - TIDAK ada di RPATH TensorFlow
#   - TIDAK terindeks ldconfig di WSL (ldconfig hanya mengindeks path standar)
#   - TIDAK terbaca di shell NON-LOGIN (wsl -- bash -c "...")
# sehingga TensorFlow mencetak "Cannot dlopen some GPU libraries" lalu
# SENGAJA diam-diam jatuh ke CPU.
#
# Empat pendekatan yang sudah dicoba:
#   1. Symlink .so ke direktori TensorFlow    -> GAGAL (tidak di jalur loader)
#   2. /etc/ld.so.conf.d + ldconfig            -> GAGAL (tidak diindeks WSL)
#   3. /etc/profile.d/tf-gpu.sh                -> HANYA untuk shell login
#   4. Seksi [env] di /etc/wsl.conf            -> TIDAK diteruskan ke proses
#                                                  python yang dijalankan WSL
#
# Yang JADI BERHASIL: menjalankan lewat SHELL LOGIN (`bash -lc`), karena
# /etc/profile.d/tf-gpu.sh akan di-source di sana. Wrapper ini yang
# jaminan hal itu.
#
# -----------------------------------------------------------------------------
# CATATAN PENTING TENTANG GPU INI
# -----------------------------------------------------------------------------
# RTX 5070 (Blackwell) punya compute capability 12.0. TensorFlow 2.21
# dibangun dengan CUDA 12.5, jadi binary kernel-nya TIDAK menyertakan
# sm_120. TensorFlow akan JIT-compile dari PTX:
#
#   "TensorFlow was not built with CUDA kernel binaries compatible with
#    compute capability 12.0a. CUDA kernels will be jit-compiled from PTX,
#    which could take 30 minutes or longer."
#
# Artinya: run PERTAMA setelah venv dibuat bisa jauh lebih lambat
# (10-30 menit) karena semua kernel dikompilasi. Setelah itu, hasil
# kompilasi di-cache di ~/.nv/ComputeCache sehingga run berikutnya
# normal. Ini normal, bukan error.
# =============================================================================
set -uo pipefail

SHORT_LIB=/opt/tf-gpu-lib
VENV=/opt/venvs/tf312

# Pastikan symlink library ada (idempotent)
if [ ! -d "$SHORT_LIB" ]; then
  echo "[setup] Membuat $SHORT_LIB ..."
  SP="$VENV/lib/python3.12/site-packages"
  mkdir -p "$SHORT_LIB"
  while IFS= read -r d; do
    [ -z "$d" ] && continue
    [ -d "$d" ] || continue
    for so in "$d"/*.so*; do
      [ -e "$so" ] || continue
      ln -sf "$so" "$SHORT_LIB/$(basename "$so")" 2>/dev/null
    done
  done < <(find "$SP/nvidia" -type d -name lib 2>/dev/null)
fi

# Cek library wajib
MISSING=0
for lib in libcudart.so.12 libcublas.so.12 libcudnn.so.9 libcusparse.so.12; do
  if [ ! -e "$SHORT_LIB/$lib" ]; then
    echo "[PERINGATAN] Library tidak ditemukan: $lib"
    MISSING=1
  fi
done
if [ "$MISSING" = "1" ]; then
  echo "[ERROR] Setup GPU belum lengkap. Jalankan scripts/setup_wsl_gpu.sh dulu."
  exit 1
fi

export LD_LIBRARY_PATH="$SHORT_LIB:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
export PATH="$VENV/bin:$PATH"

# Peringatan PTX JIT hanya ditampilkan sekali, supaya tidak membingungkan
if [ -z "${TF_GPU_WARNED:-}" ]; then
  echo "[GPU] Menjalankan dengan NVIDIA GPU via $SHORT_LIB"
  echo "[GPU] Catatan: kernel Blackwell (sm_120) di-JIT dari PTX. Run pertama"
  echo "[GPU] bisa lambat; setelah itu di-cache. Lihat README bagian GPU."
  export TF_GPU_WARNED=1
fi

# Jalankan perintah yang diminta (semua argumen diteruskan)
exec "$@"
