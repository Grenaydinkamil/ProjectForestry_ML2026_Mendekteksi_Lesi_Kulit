"""
scripts/diagnostik_hog_cell.py
=============================
Eksperimen untuk memilih ukuran sel HOG (pixels_per_cell).

Kenapa perlu eksperimen ini?
  Ukuran sel HOG menentukan berapa detail gradien yang tersimpan:
    - sel 8x8  -> 8.100 dimensi, banyak yang redundan
    - sel 16x16 -> 1.764 dimensi (dipakai proyek)
    - sel 32x32 -> 324 dimensi, detail bentuk halus hilang
  Mana yang terbaik harus diukur, bukan diasumsikan.

Script ini mengekstrak ulang fitur HOG pada beberapa ukuran sel, lalu
mengevaluasi model SVM yang sama persis untuk setiap ukuran sel.
Perbandingan ini FAIR karena hanya SATU variabel yang berubah.

Untuk sel 16x16 (pilihan proyek) fitur sudah ada di cache
(data/processed/hog_features.npy) jadi tidak diekstrak ulang.

Jalankan:
    python scripts/diagnostik_hog_cell.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src import config as C
from src import data_loader as DL
from src import features as F

# Klasifikasi ganas: mel, bcc, akiec
MALIGNANT_IDX = [C.CLASS_TO_IDX[c] for c in C.MALIGNANT_CLASSES]


def hog_dim_for_cell(cell: int) -> int:
    """Panjang vektor HOG untuk ukuran sel tertentu (tanpa gambar nyata)."""
    dummy = Image.new("RGB", (C.IMG_SIZE_CLASSIC, C.IMG_SIZE_CLASSIC))
    return len(F.extract_hog_single(dummy, pixels_per_cell=(cell, cell)))


def load_split():
    """Memuat split train/validasi."""
    return pd.read_csv(C.TRAIN_CSV), pd.read_csv(C.VAL_CSV)


def evaluate(y_true, y_pred) -> dict:
    """Metrik yang sama dipakai di notebook: F1 macro + recall ganas."""
    pred_mal = np.isin(y_pred, MALIGNANT_IDX).astype(int)
    true_mal = np.isin(y_true, MALIGNANT_IDX).astype(int)
    return {
        "f1_macro": f1_score(y_true, y_pred, average="macro",
                              zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "malignant_recall": recall_score(true_mal, pred_mal,
                                          zero_division=0),
    }


def extract_for_cell(df, cell):
    """Ekstrak HOG + histogram untuk satu dataframe pada ukuran sel tertentu."""
    hog = np.zeros((len(df), hog_dim_for_cell(cell)), dtype=np.float32)
    hist = np.zeros((len(df), F.hist_dim()), dtype=np.float32)

    for i, path_str in enumerate(df["image_path"]):
        # Path di CSV bisa relatif terhadap repo (dan berbeda antara Windows
        # dan WSL), jadi harus diselesaikan lewat helper yang sama dengan
        # dipakai notebook.
        path = DL.resolve_path(path_str)
        img = Image.open(path).convert("RGB").resize(
            (C.IMG_SIZE_CLASSIC, C.IMG_SIZE_CLASSIC), Image.BILINEAR)
        hog[i] = F.extract_hog_single(img, pixels_per_cell=(cell, cell))
        hist[i] = F.extract_color_histogram(img)

    return hog, hist


def pakai_cache(tr, va):
    """Ambil fitur sel 16x16 dari cache hasil ekstraksi notebook 03."""
    X = np.load(C.HOG_FEATURES_NPY)
    ids = pd.read_csv(C.HOG_IDS_CSV)
    posisi = {img_id: i for i, img_id in enumerate(ids["image_id"])}
    n_hog, n_hist = F.feature_slice_boundaries()

    def ambil(df):
        baris = [posisi[i] for i in df["image_id"]]
        return X[baris][:, :n_hog], X[baris][:, n_hog:]

    hog_tr, hist_tr = ambil(tr)
    hog_va, hist_va = ambil(va)
    return hog_tr, hist_tr, hog_va, hist_va


def main() -> int:
    tr, va = load_split()
    y_train = tr["dx_idx"].to_numpy()
    y_val = va["dx_idx"].to_numpy()

    print("=" * 78)
    print("DIAGNOSTIK UKURAN SEL HOG")
    print("=" * 78)
    print(f"Train {len(tr)} | Validasi {len(va)}")
    print(f"Model yang diuji: SVM (PCA {C.PCA_N_COMPONENTS} pada HOG + "
          f"histogram x{C.HIST_WEIGHT})\n")

    rows = []

    for cell in (8, 16, 32):
        t0 = time.time()
        n_hog_dim = hog_dim_for_cell(cell)

        if cell == C.HOG_PIXELS_PER_CELL[0] and C.HOG_FEATURES_NPY.exists():
            hog_tr, hist_tr, hog_va, hist_va = pakai_cache(tr, va)
            sumber = "cache"
        else:
            hog_tr, hist_tr = extract_for_cell(tr, cell)
            hog_va, hist_va = extract_for_cell(va, cell)
            sumber = "ekstraksi baru"

        # Rantai yang sama persis dengan proyek:
        #   PCA HANYA pada HOG, histogram dikali bobot, lalu digabung.
        n_comp = min(C.PCA_N_COMPONENTS, hog_tr.shape[1])
        scaler = StandardScaler().fit(hog_tr)
        pca = PCA(n_components=n_comp, random_state=C.SEED,
                  svd_solver="randomized")
        A = StandardScaler().fit_transform(
            pca.fit_transform(scaler.transform(hog_tr)))
        B = StandardScaler().fit_transform(
            pca.transform(scaler.transform(hog_va)))
        A = np.hstack([A, hist_tr * C.HIST_WEIGHT])
        B = np.hstack([B, hist_va * C.HIST_WEIGHT])

        clf = CalibratedClassifierCV(
            SVC(C=10, class_weight="balanced", random_state=C.SEED,
                cache_size=2000),
            method="sigmoid", cv=5, ensemble=False,
        )
        clf.fit(A, y_train)
        y_pred = clf.predict_proba(B).argmax(axis=1)

        m = evaluate(y_val, y_pred)
        el = time.time() - t0

        print(f"  sel {cell}x{cell}: HOG {n_hog_dim:>5} dim | "
              f"F1={m['f1_macro']:.4f} acc={m['accuracy']:.4f} "
              f"recG={m['malignant_recall']:.4f}  ({el:.0f}s, {sumber})")

        rows.append({
            "cell_size": f"{cell}x{cell}",
            "n_hog_dims": n_hog_dim,
            **m,
            "waktu_s": el,
            "sumber_fitur": sumber,
        })

    out = C.DOCS_LOGS / "diagnostik_hog_cell.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_json(out, orient="records", indent=2)
    df.to_csv(C.DOCS_ASSETS / "diagnostik_hog_cell.csv", index=False)

    print("\n" + "=" * 78)
    print("RINGKASAN")
    print("=" * 78)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nDisimpan: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
