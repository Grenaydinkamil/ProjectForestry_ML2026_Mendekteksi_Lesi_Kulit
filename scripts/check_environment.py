"""
scripts/check_environment.py
============================
Memeriksa apakah lingkungan (environment) sudah siap dipakai.

Fungsi-fungsi yang diperiksa:
  1. Versi Python dan library
  2. Apakah TensorFlow bisa import, dan apakah GPU terdeteksi
  3. Apakah dataset HAM10000 lengkap dan bisa dibaca
  4. Apakah modul src bisa diimpor dan fungsi preprocessing jalan

Jalankan:
    python scripts/check_environment.py

Keluar dengan kode 0 kalau semua inti OK, kode 1 kalau ada yang fatal.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Warna ANSI sederhana supaya output mudah dibaca di terminal
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> int:
    failures = 0

    # -----------------------------------------------------------------------
    section("1. VERSI PYTHON DAN LIBRARY")
    # -----------------------------------------------------------------------
    print(f"Python: {sys.version.split()[0]}")
    if sys.version_info < (3, 10):
        print(f"{RED}[X] Python >= 3.10 wajib (TensorFlow 2.21 tidak "
              f"mendukung 3.9){RESET}")
        failures += 1
    elif sys.version_info >= (3, 14):
        print(f"{RED}[X] Python >= 3.14 tidak didukung TensorFlow 2.21. "
              f"Gunakan Python 3.13.{RESET}")
        failures += 1
    else:
        print(f"{GREEN}[OK] Versi Python didukung TensorFlow 2.21{RESET}")

    packages = [
        ("tensorflow", "2.21.0"),
        ("keras", "3.15.1"),
        ("numpy", "2.5.4"),
        ("pandas", None),
        ("sklearn", None),
        ("skimage", None),
        ("matplotlib", None),
        ("seaborn", None),
        ("PIL", None),
        ("joblib", None),
        ("streamlit", None),
    ]
    for mod_name, expected in packages:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "tidak diketahui")
            flag = f"{GREEN}[OK]{RESET}"
            if expected and ver != expected:
                flag = f"{YELLOW}[!]{RESET} (diharapkan {expected})"
            print(f"  {flag} {mod_name:14s} {ver}")
        except ImportError:
            print(f"  {RED}[X] {mod_name:14s} TIDAK TERINSTALL{RESET}")
            failures += 1

    # -----------------------------------------------------------------------
    section("2. TENSORFLOW DAN GPU")
    # -----------------------------------------------------------------------
    try:
        import tensorflow as tf

        print(f"TensorFlow: {tf.__version__}")
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            print(f"{GREEN}[OK] GPU terdeteksi: {len(gpus)} device{RESET}")
            for g in gpus:
                print(f"     - {g}")
            # Uji komputasi nyata, bukan cuma deteksi
            with tf.device("/GPU:0"):
                a = tf.random.normal((2048, 2048))
                b = tf.matmul(a, a)
            print(f"{GREEN}[OK] Uji komputasi GPU berhasil (matmul 2048x2048){RESET}")
        else:
            print(f"{YELLOW}[!] GPU tidak terdeteksi -> training berjalan di CPU{RESET}")
            print("     Ini WAJAR dan BISA DITERIMA pada Windows native, karena")
            print("     TensorFlow >= 2.11 tidak mendukung GPU di Windows native.")
            print("     Opsi GPU: (a) WSL2, (b) Google Colab.")
    except Exception as exc:  # noqa: BLE001
        print(f"{RED}[X] TensorFlow gagal diimpor: {exc}{RESET}")
        failures += 1

    # -----------------------------------------------------------------------
    section("3. KELENGKAPAN DATASET")
    # -----------------------------------------------------------------------
    try:
        from src import config as C
        from src import data_loader

        df = data_loader.load_metadata()
        report = data_loader.verify_dataset_integrity(df, verbose=True)

        if report["n_missing_image_file"] > 0:
            print(f"{RED}[X] {report['n_missing_image_file']} file gambar "
                  f"hilang{RESET}")
            failures += 1
        else:
            print(f"{GREEN}[OK] Semua file gambar ada{RESET}")
    except Exception as exc:  # noqa: BLE001
        print(f"{RED}[X] Dataset belum siap: {exc}{RESET}")
        failures += 1

    # -----------------------------------------------------------------------
    section("4. UJI MODUL src")
    # -----------------------------------------------------------------------
    try:
        import numpy as np
        from src import features, preprocessing

        # Uji HOG + histogram pada gambar dummy
        from PIL import Image

        dummy = Image.new("RGB", (128, 128), (120, 90, 70))
        hog_feat = features.extract_hog_single(dummy)
        hist_feat = features.extract_color_histogram(dummy)
        print(f"  [OK] HOG         : {hog_feat.shape[0]} fitur")
        print(f"  [OK] Histogram   : {hist_feat.shape[0]} fitur "
              f"(total {features.hog_dim() + features.hist_dim()} fitur/gambar)")

        # Uji resize + normalisasi
        arr = np.zeros((300, 400, 3), dtype=np.float32)
        resized = preprocessing.resize_image(arr, 224)
        print(f"  [OK] Resize      : {resized.shape}")

        mob = preprocessing.normalize_for_mobilenet(resized)
        print(f"  [OK] Normalisasi MobileNetV2: rentang "
              f"[{float(tf.reduce_min(mob)):.2f}, {float(tf.reduce_max(mob)):.2f}]")
        if float(tf.reduce_min(mob)) < -1.5 or float(tf.reduce_max(mob)) > 1.5:
            print(f"{RED}[X] Rentang normalisasi MobileNetV2 salah "
                  f"(harus sekitar -1..1){RESET}")
            failures += 1

        # Uji layer augmentasi
        aug = preprocessing.build_train_augmentation(smartphone_style=True)
        out = aug(tf.zeros((1, 224, 224, 3)))
        print(f"  [OK] Augmentasi  : output {out.shape}")
    except Exception as exc:  # noqa: BLE001
        import traceback

        print(f"{RED}[X] Uji modul src gagal: {exc}{RESET}")
        traceback.print_exc()
        failures += 1

    # -----------------------------------------------------------------------
    section("5. RINGKASAN")
    # -----------------------------------------------------------------------
    if failures == 0:
        print(f"{GREEN}SEMUA PEMERIKSAAN LULUS. Lingkungan siap dipakai.{RESET}")
        return 0
    print(f"{RED}Ada {failures} masalah yang perlu diperbaiki.{RESET}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
