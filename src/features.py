"""
features.py
===========
Ekstraksi fitur untuk model KLASIK (SVM dan Random Forest).

Analogi sederhana
-----------------
Bayangkan kita memberi tahu seseorang "sebutkan ciri-ciri yang membuat foto
iniMirip dengan foto lesi melanoma". Orang itu tidak akan menjawab "warnanya
sedikit cokelat di tengah"; dia akan menjawab "tepi-tepinya tidak rata, ada
pola seperti jaring di tengah, warnanya tidak merata".

Tugas modul ini persis seperti itu: mengubah gambar (yang tidak bisa dibaca
komputer secara langsung) menjadi daftar angka yang menggambarkan "ciri".

Dua ciri yang kita pakai
-------------------------
1. HOG (Histogram of Oriented Gradients)
   Mengukur ARAH DAN KEKUATAN GARIS di setiap bagian kecil gambar.
   Analoginya seperti menyinari gambar dari samping: bagian yang warnanya
   berubah tajam (tepi lesi) akan menghasilkan banyak garis yang mengarah ke
   arah tepi itu. Lesi dengan tepi tidak rata akan punya pola garis yang
   tidak teratur.

2. Histogram Warna
   Menghitung berapa banyak piksel tiap warna (merah/hijau/biru pada
   rentang tertentu) di gambar. Melanoma cenderung lebih gelap, lebih
   cokelat, dan warnanya tidak merata dibanding mola biasa.

Kenapa HOG + Histogram, bukan piksel mentah?
---------------------------------------------
Piksel mentah (128*128*3 = 49.152 fitur per gambar) membuat SVM dan
Random Forest:
  - sangat lambat (dimension tinggi =IFICATE curse of dimensionality)
  - mudah overfit, karena model bisa saja menghafal posisi piksel
HOG + histogram hanya menghasilkan 1.812 fitur yang sudah
merangkum "ciri penting", sehingga lebih cepat, lebih general, dan
lebih mudah dijelaskan saat tanya jawab.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import config as C

from PIL import Image
from skimage.feature import hog


# ---------------------------------------------------------------------------
# 1. EKSTRAKSI HOG DARI SATU GAMBAR
# ---------------------------------------------------------------------------
def extract_hog_single(
    img: Image.Image,
    orientations: int = C.HOG_ORIENTATIONS,
    pixels_per_cell: tuple = C.HOG_PIXELS_PER_CELL,
    cells_per_block: tuple = C.HOG_CELLS_PER_BLOCK,
) -> np.ndarray:
    """
    Menghitung vektor HOG dari satu gambar PIL.

    Parameter:
      img             : gambar PIL (harus sudah di-resize)
      orientations    : jumlah arah orientasi gradien yang dibedakan (9)
      pixels_per_cell : ukuran sel dalam piksel (16x16)
      cells_per_block : jumlah sel per blok (2x2)

    Kenapa `transform_sqrt=True`?
      HOG standar membandingkan ||gradien|| DAN ||gradien||^2. Yang kuadrat
      menekankan piksel dengan gradien sangat kuat (titik kontras tinggi),
      dan transformasi akar membuatnya lebih seimbang terhadap perbedaan
      pencahayaan. Ini parameter default skimage yang sudah tervalidasi.
    """
    # HOG butuh gambar grayscale: bentuk (tinggi, lebar)
    gray = np.asarray(img.convert("L"), dtype=np.float32)

    features = hog(
        gray,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True,
    )
    return features.astype(np.float32)


# ---------------------------------------------------------------------------
# 2. EKSTRAKSI HISTOGRAM WARNA DARI SATU GAMBAR
# ---------------------------------------------------------------------------
def extract_color_histogram(
    img: Image.Image,
    bins: int = C.HIST_BINS,
) -> np.ndarray:
    """
    Menghitung histogram warna per kanal RGB.

    Setiap kanal (R, G, B) dipecah menjadi `bins` rentang nilai 0-255.
    Hasilnya 3 * bins angka, masing-masing menyatakan "berapa persen piksel
    pada gambar ini punya intensitas merah di rentang tertentu".

    Normalisasi ke rentang [0, 1] dilakukan supaya:
      - tidak ada kanal yang mendominasi karena skalanya berbeda
      - semua fitur punya skala yang sama saat dipakai SVM (StandardScaler)
    """
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)  # (H, W, 3)
    hist = []
    for channel in range(3):  # 0=R, 1=G, 2=B
        counts, _ = np.histogram(
            arr[:, :, channel],
            bins=bins,
            range=(0.0, 256.0),
            density=False,
        )
        hist.append(counts.astype(np.float32))
    hist = np.concatenate(hist)
    # Normalisasi L1: jumlah semua bin = 1.0 (proporsi piksel)
    total = hist.sum()
    if total > 0:
        hist = hist / total
    return hist.astype(np.float32)


# ---------------------------------------------------------------------------
# 3. EKSTRAKSI GABUNGAN UNTUK SATU GAMBAR
# ---------------------------------------------------------------------------
def extract_features_single(
    img: Image.Image,
    img_size: int = C.IMG_SIZE_CLASSIC,
    hist_bins: int = C.HIST_BINS,
) -> np.ndarray:
    """
    Menggabungkan HOG + histogram warna menjadi SATU vektor fitur.

    Urutan bagian vektor: [HOG (bagian 1), Histogram RGB (bagian 2)].
    Panjang setiap bagian dicatat di konstanta modul supaya evaluation
    bisa memotong dengan benar jika perlu.
    """
    hog_feat = extract_hog_single(img)
    hist_feat = extract_color_histogram(img, bins=hist_bins)
    return np.concatenate([hog_feat, hist_feat])


def hog_dim(img_size: int = C.IMG_SIZE_CLASSIC) -> int:
    """Menghitung panjang vektor HOG tanpa perlu gambar (dipakai saat setup)."""
    dummy = Image.new("RGB", (img_size, img_size))
    return len(extract_hog_single(dummy))


def hist_dim(hist_bins: int = C.HIST_BINS) -> int:
    """Panjang vektor histogram warna."""
    return 3 * hist_bins


def feature_slice_boundaries(img_size: int = C.IMG_SIZE_CLASSIC,
                             hist_bins: int = C.HIST_BINS) -> tuple[int, int]:
    """
    Mengembalikan batas indeks fitur HOG dan fitur histogram.

    Vektor fitur disusun sebagai: [HOG | histogram]. Fungsi ini dipakai
    ColumnTransformer untuk memproses kedua bagian dengan cara berbeda:
      - HOG perlu StandardScaler + PCA (skalanya besar dan saling berkorelasi)
      - Histogram sudah ternormalisasi 0..1, jadi TIDAK boleh di-scale
        ulang (dan diberi bobot eksplisit)
    """
    return hog_dim(img_size), hist_dim(hist_bins)


# ---------------------------------------------------------------------------
# 4. EKSTRAKSI FITUR UNTUK SELURUH DATASET (dengan cache ke disk)
# ---------------------------------------------------------------------------
def extract_features_for_table(
    table: pd.DataFrame,
    img_size: int = C.IMG_SIZE_CLASSIC,
    hist_bins: int = C.HIST_BINS,
    cache_npy: bool = True,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Mengekstrak fitur untuk seluruh baris tabel.

    Parameter:
      table     : DataFrame hasil build_image_table()
      cache_npy : bila True, hasil disimpan ke .npy supaya tidak perlu
                  diekstrak ulang setiap kali notebook dijalankan.
                  Ekstraksi 10.015 gambar butuh beberapa menit, jadi
                  cache sangat menghemat waktu.

    Mengembalikan: (X, y, id_df)
      X     : matriks fitur (n_gambar x n_fitur)
      y     : label integer (n_gambar,)
      id_df : DataFrame berisi image_id, lesion_id, dx untuk pelacakan
    """
    cache_ok = False
    if cache_npy and C.HOG_FEATURES_NPY.exists() and C.HOG_LABELS_NPY.exists():
        try:
            X = np.load(C.HOG_FEATURES_NPY)
            y = np.load(C.HOG_LABELS_NPY)
            id_df = pd.read_csv(C.HOG_IDS_CSV)
            if len(X) == len(table) and len(y) == len(table):
                cache_ok = True
                if verbose:
                    print(f"Memuat fitur dari cache: {C.HOG_FEATURES_NPY.name}")
            else:
                if verbose:
                    print(
                        "Cache tidak cocok dengan tabel saat ini "
                        "(jumlah baris beda), mengekstrak ulang."
                    )
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"Cache tidak bisa dibaca ({exc}), mengekstrak ulang.")

    if not cache_ok:
        feats = []
        labels = []
        iterator = tqdm(
            table.itertuples(index=False),
            total=len(table),
            desc="Ekstraksi HOG + histogram",
            unit="img",
        )
        for row in iterator:
            img = Image.open(row.image_path).convert("RGB")
            img = img.resize((img_size, img_size), Image.BILINEAR)
            feats.append(extract_features_single(img, img_size, hist_bins))
            labels.append(row.dx_idx)

        X = np.vstack(feats).astype(np.float32)
        y = np.asarray(labels, dtype=np.int64)

        id_df = table[["image_id", "lesion_id", "dx", "dx_idx"]].reset_index(drop=True)

        if cache_npy:
            C.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
            np.save(C.HOG_FEATURES_NPY, X)
            np.save(C.HOG_LABELS_NPY, y)
            id_df.to_csv(C.HOG_IDS_CSV, index=False)
            if verbose:
                print(f"Fitur disimpan ke {C.HOG_FEATURES_NPY}")

    return X, y, id_df
