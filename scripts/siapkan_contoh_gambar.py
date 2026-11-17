"""
Siapkan contoh gambar untuk demo aplikasi.

Kenapa perlu?
  Demo video harus bisa menunjukkan aplikasi bekerja tanpa harus
  mencari-cari file foto di laptop. Tiga contoh gambar (satu jinak,
  satu ganas) membuat demo jauh lebih mulus.

Dasar pemilihan gambar:
  - Ambil dari HAM10000 (dataset yang dipakai training), lisensi
    CC BY-NC 4.0.
  - Pilih gambar dari data TEST (bukan train), supaya presenter bisa
    menunjukkan hasil yang sudah diketahui dari laporan.
  - Satulesi jinak (nevus) dan satu lesi ganas (melanoma), plus satu
    contoh dari PAD-UFES-20 (foto smartphone) untuk menunjukkan
    perbedaan domain.

Jalankan:
    python scripts/siapkan_contoh_gambar.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from PIL import Image

from src import config as C

KELUAR = REPO_ROOT / "app" / "contoh"

# Nama file keluaran -> (kelas diagnosis, keterangan)
TARGET = {
    "contoh_melanoma.jpg": ("mel", "Melanoma malignant (foto dermoskop)"),
    "contoh_nevus.jpg": ("nv", "Nevus jinak (foto dermoskop)"),
    "contoh_bcc.jpg": ("bcc", "Karsinoma sel basal (foto dermoskop)"),
}


def cari_test(path_ke_awal: str, kelas: str) -> str | None:
    """
    Cari satu gambar dari split test untuk kelas tertentu.

    `image_path` di test.csv disimpan relatif terhadap folder repo, jadi
    harus diselesaikan lebih dulu. Kalau tidak ketemu (mis. data ada
    di lokasi WSL), baru mencoba folder data mentah sebagai cadangan.
    """
    df = pd.read_csv(C.TEST_CSV)
    baris = df[df["dx"] == kelas]
    if baris.empty:
        return None

    for _, r in baris.iterrows():
        # 1. Relatif terhadap folder repo
        p = REPO_ROOT / r["image_path"]
        if p.exists():
            return str(p)
        # 2. Langsung di folder data mentah
        p2 = Path(path_ke_awal) / Path(r["image_path"]).name
        if p2.exists():
            return str(p2)
    return None


def main() -> int:
    KELUAR.mkdir(parents=True, exist_ok=True)

    # Lokasi data mentah (bisa berbeda antara Windows dan WSL)
    ham_dirs = sorted((C.DATA_RAW).glob("HAM10000_images_part_*"))
    pad_dirs = sorted((C.DATA_EXTERNAL / "images").glob("imgs_part_*"))

    dibuat = []

    for nama, (kelas, ket) in TARGET.items():
        for d in ham_dirs:
            p = cari_test(str(d), kelas)
            if p:
                shutil.copyfile(p, KELUAR / nama)
                dibuat.append((nama, ket, Path(p).name))
                break

    # Contoh foto smartphone dari PAD-UFES-20
    if pad_dirs:
        meta_pad = C.DATA_EXTERNAL / "metadata.csv"
        if meta_pad.exists():
            df_pad = pd.read_csv(meta_pad)
            baris = df_pad[df_pad["diagnostic"] == "MEL"]
            if not baris.empty:
                for _, r in baris.iterrows():
                    nama_file = Path(r["img_id"]).name
                    for d in pad_dirs:
                        p = d / nama_file
                        if p.exists():
                            # Ekstensi asli HARUS dipertahankan: file PNG
                            # yang diberi ekstensi .jpg akan membingungkan.
                            keluaran = "contoh_smartphone_mel" + p.suffix.lower()
                            shutil.copyfile(p, KELUAR / keluaran)
                            dibuat.append((
                                keluaran,
                                "Melanoma pada foto smartphone (PAD-UFES-20)",
                                nama_file,
                            ))
                            break
                    break

    print(f"{'file':<30}{'keterangan':<52}sumber")
    print("-" * 100)
    for nama, ket, asal in dibuat:
        ukuran = (KELUAR / nama).stat().st_size / 1024
        print(f"{nama:<30}{ket:<52}{asal} ({ukuran:.0f} KB)")
    print(f"\nTotal: {len(dibuat)} contoh -> {KELUAR}")
    return 0 if dibuat else 1


if __name__ == "__main__":
    sys.exit(main())
