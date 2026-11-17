"""
data_loader.py
==============
Fungsi-fungsi untuk memuat dataset HAM10000 secara aman dan konsisten.

Kenapa modul ini ada?
---------------------
Tiga hal yang sering jadi sumber bug kalau ditulis ulang-ulang:
  1. Metadata DAN file gambar harus PASANGAN. Kalau metadata punya 10.015 baris
     tapi ada 5 gambar hilang, setiap indeks jadi geser dan label meleset.
     Modul ini memverifikasi kelengkapan dan memberi peringatan jelas.
  2. Pencarian file gambar harus tahu bahwa gambar dipecah di DUA folder.
  3.|image_id harus dipetakan ke path file secara eksplisit, bukan dengan
     asumsi "file ada di folder yang sama dengan urutan yang sama".

Modul ini dipakai bersama oleh notebook EDA, notebook training, dan aplikasi
web, sehingga tidak ada lagi penulisan ulang logika yang bisa melenceng.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import pandas as pd

from . import config as C


# ---------------------------------------------------------------------------
# 1. MEMUAT METADATA
# ---------------------------------------------------------------------------
def resolve_path(path_str: str) -> Path:
    """
    Mengubah path yang tersimpan di CSV menjadi Path yang bisa diakses.

    Kenapa ini perlu?
      Proyek ini bisa dijalankan di DUA tempat:
        (a) Windows native  -> repo di M:\\... dan python di C:\\Python314
        (b) WSL2 (untuk GPU) -> repo di /mnt/m/... dan python di /opt/venvs
      Kalau CSV menyimpan path absolut Windows ("M:\\...\\ISIC_x.jpg"), maka
      di WSL path itu tidak ada. Kalau menyimpan path relatif, keduanya
      aman karena path relatif selalu dihitung relatif ke REPO_ROOT.

    Fungsi ini menangani semua kasus:
      1. Path absolut yang langsung bisa diakses (Windows, path relatif WSL)
      2. Path relatif, dihitung dari REPO_ROOT
      3. Path dengan separator Windows (backslash) saat dijalankan di Linux
      4. Path absolut Windows, diterjemahkan ke /mnt/<drive>/... (dari WSL)

    Mengembalikan Path (paling mungkin belum ada; pemanggil yang mengecek).
    """
    p = Path(path_str)

    # 1. OVERRIDE DATA_RAW (kalau di-set, misal training di WSL2 dengan
    #    data di disk native WSL). Ini diprioritaskan karena tujuannya
    #    memang memaksa pembacaan dari lokasi cepat.
    raw_override = C.DATA_RAW
    default_raw = C.REPO_ROOT / "data" / "raw"
    if raw_override != default_raw:
        s_rel = str(path_str).replace("\\", "/")
        if s_rel.startswith("data/raw/"):
            candidate = raw_override / s_rel[len("data/raw/"):]
            if candidate.exists():
                return candidate

    # 2. Sudah absolut dan ada
    if p.is_absolute() and p.exists():
        return p

    # 3. Relatif terhadap repo root
    candidate = C.REPO_ROOT / p
    if candidate.exists():
        return candidate

    # 4. Path relatif dengan separator Windows (backslashes).
    #    Ini terjadi kalau CSV dibuat di Windows lalu dibaca di Linux.
    #    Di Linux, "data\\raw\\file.jpg" adalah SATU nama file, bukan path.
    s = str(path_str)
    if "\\" in s:
        unix_relative = Path(s.replace("\\", "/"))
        candidate = C.REPO_ROOT / unix_relative
        if candidate.exists():
            return candidate

    # 4. Terjemahkan path absolut Windows ke gaya WSL (/mnt/<drive>/...)
    #    Contoh: "M:\\GrenPlums\\x.jpg" -> "/mnt/m/GrenPlums/x.jpg"
    if len(s) > 2 and s[1] == ":" and s[2] in ("\\", "/"):
        wsl_path = Path(f"/mnt/{s[0].lower()}{s[2:].replace(chr(92), '/')}")
        if wsl_path.exists():
            return wsl_path
        return wsl_path

    return candidate


def load_metadata() -> pd.DataFrame:
    """
    Membaca HAM10000_metadata.csv dan menambahkan kolom pendukung.

    Kolom yang ditambahkan:
      - image_path : path absolut ke file gambar
      - has_image  : True kalau file gambarnya benar-benar ada di disk

    Kenapa tidak langsung pakai index? Karena ilang saat filter/split,
    dan index yang kacau adalah penyebab paling umum label meleset.
    """
    if not C.METADATA_CSV.exists():
        raise FileNotFoundError(
            f"Metadata tidak ditemukan di: {C.METADATA_CSV}\n"
            "Jalankan skrip unduh dataset terlebih dahulu (lihat README bagian "
            "'Menyiapkan Dataset')."
        )

    df = pd.read_csv(C.METADATA_CSV)

    # Pastikan kolom inti ada sebelum dipakai. Failing cepat lebih baik
    # daripada error aneh di tengah analisis.
    required_cols = {"lesion_id", "image_id", "dx"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Kolom wajib hilang dari metadata: {sorted(missing_cols)}. "
            f"Kolom yang tersedia: {list(df.columns)}"
        )

    # Bangun peta image_id -> path file dengan menioma seluruh folder gambar.
    # Peta ini menggantikan asumsi urutan file sama dengan urutan baris metadata.
    path_map: dict[str, Path] = {}
    for folder in C.IMAGE_DIRS:
        if not folder.exists():
            continue
        for p in folder.glob("*.jpg"):
            path_map[p.stem] = p

    df["image_path"] = df["image_id"].map(path_map)
    df["has_image"] = df["image_path"].notna()
    df["dx"] = df["dx"].str.strip().str.lower()

    # Ubah kolom flag agar jadi boolean sejati, bukan objek <NA> yang membingungkan
    df["has_image"] = df["has_image"].astype(bool)

    return df


def verify_dataset_integrity(df: pd.DataFrame, verbose: bool = True) -> dict:
    """
    Memeriksa apakah metadata dan file gambar benar-benar lengkap.

    Mengembalikan dict berisi ringkasan. Tidak melempar error, karena kita
    ingin tetap bisa melakukan EDA pada dataset yang sedikit bermasalah
    (dan mencatatnya di laporan).
    """
    report = {
        "n_rows_metadata": int(len(df)),
        "n_unique_lesion_id": int(df["lesion_id"].nunique()),
        "n_unique_image_id": int(df["image_id"].nunique()),
        "n_missing_image_file": int((~df["has_image"]).sum()),
        "n_zero_byte_image": 0,
        "n_duplicate_rows": int(df.duplicated().sum()),
        "n_missing_age": int(df["age"].isna().sum()),
        "classes": sorted(df["dx"].unique().tolist()),
        "n_images_per_lesion_max": int(df.groupby("lesion_id").size().max()),
    }

    # Cek file kosong (gagal unduh / korup). File 0 byte akan membuat
    # PIL raise error saat dibaca, jadi ini penting dicek.
    empty = 0
    for p in df.loc[df["has_image"], "image_path"].unique():
        try:
            if os.path.getsize(p) == 0:
                empty += 1
        except OSError:
            empty += 1
    report["n_zero_byte_image"] = empty

    if verbose:
        print("=== RINGKASAN INTEGRITAS DATASET ===")
        for k, v in report.items():
            print(f"  {k}: {v}")

    return report


# ---------------------------------------------------------------------------
# 2. MEMBUAT TABEL PATH LENGKAP
# ---------------------------------------------------------------------------
def build_image_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mengembalikan tabel berisi info setiap gambar yang siap dipakai model:
      image_id, image_path, dx, dx_idx, lesion_id, plus kolom metadata lain
    (age, sex, localization) yang berguna untuk EDA.
    """
    out = df[df["has_image"]].copy()
    out["dx_idx"] = out["dx"].map(C.CLASS_TO_IDX)

    if out["dx_idx"].isna().any():
        unknown = sorted(out.loc[out["dx_idx"].isna(), "dx"].unique())
        raise ValueError(
            f"Ada kelas yang tidak dikenal: {unknown}. "
            f"Kelas yang diharapkan: {C.CLASSES}"
        )

    out["dx_idx"] = out["dx_idx"].astype(int)
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. MEMBACA SATU GAMBAR
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _pil_available():
    from PIL import Image  # noqa: F401
    return True


def load_image_as_rgb(path: str | Path, size: int | None = None):
    """
    Membaca satu gambar dan mengubahnya menjadi array RGB.

    Parameter:
      path : path file gambar
      size : bila diberikan, gambar di-resize ke (size, size)

    Penting: dataset ini hampir semua gambar 600x450 (4:3), bukan persegi.
    Kita resize ke persegi (size x size) karena:
      - HOG dan CNN mengasumsikan input persegi
      - memakai padding (pad ke persegi) akan menambah background hitam yang
        bisa jadi artefak, jadi peregangan (stretch) lebih aman karena model
        juga dilatih dengan augmentasi yang meniru variasi framing
    """
    from PIL import Image

    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize((size, size), Image.BILINEAR)
    return img


# ---------------------------------------------------------------------------
# 4. MEMBUKA METADATA DENGAN PATH RELATIF (untuk aplikasi web)
# ---------------------------------------------------------------------------
def load_split(name: str):
    """
    Memuat salah satu file split (train/val/test) dari data/processed/.

    Fungsi ini mengembalikan DataFrame dengan kolom `image_path` yang SUDAH
    berupa path absolut yang valid untuk lingkungan saat ini (Windows atau
    WSL2). Dipakai oleh semua notebook tahap berikutnya supaya tidak ada
    lagi penulisan ulang logika path.

    Parameter:
      name : "train", "val", atau "test"

    Mengembalikan (dataframe, resolved_path_column_name)
    """
    csv_path = C.DATA_PROCESSED / f"{name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"File split '{name}.csv' tidak ditemukan di {csv_path}.\n"
            "Jalankan notebook 02_split.ipynb terlebih dahulu."
        )

    df = pd.read_csv(csv_path)

    # Resolusi path agar valid di Windows maupun WSL
    df["image_path"] = df["image_path"].map(lambda p: str(resolve_path(p)))
    df["dx_idx"] = df["dx_idx"].astype(int)

    # Verifikasi cepat: pastikan gambar benar-benar ada
    missing = df["image_path"].map(lambda p: not Path(p).exists()).sum()
    if missing > 0:
        raise FileNotFoundError(
            f"{missing} file gambar tidak ditemukan untuk split '{name}'.\n"
            "Contoh path yang gagal:\n" +
            "\n".join(df.loc[
                df["image_path"].map(lambda p: not Path(p).exists()),
                "image_path",
            ].head(3).tolist())
        )

    return df


def find_image_dir() -> Path | None:
    """Mencari folder gambar pertama yang tersedia (untuk aplikasi web)."""
    for folder in C.IMAGE_DIRS:
        if folder.exists() and any(folder.glob("*.jpg")):
            return folder
    return None
