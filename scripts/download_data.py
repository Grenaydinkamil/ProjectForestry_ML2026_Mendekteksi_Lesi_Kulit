"""
scripts/download_data.py
=========================
Skrip untuk mengunduh dan menyiapkan dataset yang dibutuhkan proyek.

PENTING untuk mahasiswa: skrip ini TIDAK melakukan login ke Kaggle maupun
menerima lisensi apa pun. Pengunduhan dataset dan persetujuan syarat
penggunaan harus dilakukan oleh mahasiswa secara manual (lihat README).
Skrip ini hanya membantu mengekstrak file yang sudah Anda unduh.

Cara pakai
----------
1. Unduh dataset secara manual:
   - HAM10000  : Kaggle "Skin Cancer MNIST: HAM10000"
                 atau Harvard Dataverse (cari "HAM10000 Tschandl")
   - PAD-UFES-20: Mendeley Data "A Large-Scale Dataset of Pigmented Skin
                 Lesions" (Pacheco dkk. 2020)
2. Simpan file .zip ke folder data/raw/
3. Jalankan:
       python scripts/download_data.py --extract-ham10000
       python scripts/download_data.py --extract-pad-ufes20
       python scripts/download_data.py --verify

Pilihan lain:
   --extract-ham10000 : ekstrak ZIP HAM10000 ke data/raw/
   --extract-pad-ufes20: ekstrak ZIP PAD-UFES-20 ke data/external/
   --verify           : periksa kelengkapan dataset yang sudah diekstrak
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

# Folder repo = parent dari folder scripts/
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_EXTERNAL = REPO_ROOT / "data" / "external"


# ---------------------------------------------------------------------------
# EKSTRAKSI
# ---------------------------------------------------------------------------
def extract_zip(zip_path: Path, dest: Path) -> int:
    """
    Mengekstrak ZIP ke folder tujuan.

    Kenapa pakai shutil.unpack_archive dan bukan loop manual?
      Library ini sudah menangani path separator Windows/Linux, encoding
      nama file, dan membuat folder tujuan bila belum ada.

    Mengembalikan jumlah file yang diekstrak.
    """
    if not zip_path.exists():
        raise FileNotFoundError(f"File ZIP tidak ditemukan: {zip_path}")

    dest.mkdir(parents=True, exist_ok=True)
    print(f"Mengekstrak: {zip_path.name}")
    print(f"          ke: {dest}")

    with zipfile.ZipFile(zip_path) as zf:
        # Cek dulu apakah ada file yang akan menimpa file yang sudah ada.
        existing = {p.resolve() for p in dest.rglob("*") if p.is_file()}
        collisions = [
            n for n in zf.namelist()
            if not n.endswith("/") and (dest / n).resolve() in existing
        ]
        if collisions:
            print(f"  Perhatian: {len(collisions)} file sudah ada dan akan "
                  "ditimpa (extracting akan gagal tanpa overwrite).")
            print("  Memakai mode overwrite.")

        # Ekstrak per entri supaya bisa menangani tabrakan nama.
        count = 0
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            target = dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
            count += 1

    print(f"  Selesai: {count} file diekstrak")
    return count


# ---------------------------------------------------------------------------
# VERIFIKASI
# ---------------------------------------------------------------------------
def verify() -> bool:
    """
    Memeriksa kelengkapan dataset TANPA mengasumsikan nama file.

    Prinsip: yang dicek adalah KONTEN (apakah metadata ada, apakah jumlah
    gambar cocok dengan metadata), bukan nama folder tertentu.
    """
    print("=" * 70)
    print("VERIFIKASI DATASET")
    print("=" * 70)

    ok = True

    # --- Cari metadata CSV di mana pun di data/raw ---
    metadata_files = list(DATA_RAW.rglob("*metadata*.csv")) if DATA_RAW.exists() else []
    if not metadata_files:
        print("[X] Metadata CSV tidak ditemukan di data/raw/")
        print("    Unduh dan ekstrak dataset HAM10000 terlebih dahulu.")
        return False

    metadata_path = metadata_files[0]
    print(f"[OK] Metadata: {metadata_path.relative_to(REPO_ROOT)}")

    import pandas as pd

    df = pd.read_csv(metadata_path)
    n_meta = len(df)
    print(f"     jumlah baris metadata: {n_meta:,}")

    # --- Hitung file gambar di seluruh subfolder data/raw ---
    jpg_files = list(DATA_RAW.rglob("*.jpg")) if DATA_RAW.exists() else []
    n_jpg = len(jpg_files)
    print(f"     jumlah file .jpg     : {n_jpg:,}")

    if n_jpg != n_meta:
        print(f"[!] Jumlah gambar ({n_jpg:,}) TIDAK sama dengan metadata "
              f"({n_meta:,}).")
        ok = False
    else:
        print("[OK] Jumlah gambar cocok dengan metadata.")

    # --- Cek folder gambar yang terdeteksi ---
    folders = sorted({p.parent for p in jpg_files})
    print(f"     folder gambar terdeteksi ({len(folders)}):")
    for f in folders:
        cnt = len(list(f.glob("*.jpg")))
        print(f"       - {f.relative_to(REPO_ROOT)}  ({cnt:,} gambar)")

    # --- Cek file kosong ---
    empty = [p for p in jpg_files if p.stat().st_size == 0]
    if empty:
        print(f"[X] {len(empty)} file gambar berukuran 0 byte (gagal ekstrak).")
        ok = False
    else:
        print("[OK] Tidak ada file gambar 0 byte.")

    # --- Cek PAD-UFES-20 (opsional, hanya untuk uji eksternal) ---
    pad_candidates = list(DATA_EXTERNAL.rglob("*.csv")) if DATA_EXTERNAL.exists() else []
    if pad_candidates:
        print(f"[OK] PAD-UFES-20 terdeteksi: {len(pad_candidates)} file CSV")
    else:
        print("[--] PAD-UFES-20 belum ada (hanya dibutuhkan untuk Fase 7).")

    print("=" * 70)
    print("HASIL:", "SEMUA LENGKAP" if ok else "ADA MASALAH")
    return ok


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Unduh/ekstrak & verifikasi dataset proyek ML lesi kulit."
    )
    parser.add_argument(
        "--extract-ham10000", action="store_true",
        help="Ekstrak file ZIP HAM10000 yang ada di data/raw/ ke data/raw/",
    )
    parser.add_argument(
        "--extract-pad-ufes20", action="store_true",
        help="Ekstrak file ZIP PAD-UFES-20 yang ada di data/raw/ ke data/external/",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Periksa kelengkapan dataset yang sudah diekstrak",
    )
    args = parser.parse_args()

    did_something = False

    if args.extract_ham10000:
        zips = sorted(DATA_RAW.glob("*.zip"))
        if not zips:
            print("[X] Tidak ada file .zip di data/raw/. Unduh manual dahulu.")
            return 1
        for z in zips:
            extract_zip(z, DATA_RAW)
        did_something = True

    if args.extract_pad_ufes20:
        zips = sorted(DATA_RAW.glob("*.zip"))
        if not zips:
            print("[X] Tidak ada file .zip di data/raw/. Unduh manual dahulu.")
            return 1
        for z in zips:
            extract_zip(z, DATA_EXTERNAL)
        did_something = True

    if args.verify or not did_something:
        ok = verify()
        return 0 if ok else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
