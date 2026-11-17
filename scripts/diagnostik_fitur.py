"""
scripts/diagnostik_fitur.py
===========================
Eksperimen diagnostik untuk memahami kenapa fitur HOG + histogram perlu
dirancang ulang, dan untuk membuktikan pilihan-pilihan desain di
`src/config.py`.

Semua angka di `docs/CATATAN_TEMUAN_DIAGNOSTIK.md` berasal dari skrip ini.
Tidak ada angka yang diketik manual.

Eksperimen yang dijalankan:

  1. Baseline "selalu menebak nv" -- titik awal semua perbandingan.
  2. Random Forest pada SEMUA fitur (tanpa PCA).
  3. SVC dengan PCA yang diterapkan ke seluruh fitur (termasuk histogram).
  4. SVM / RF pada histogram warna saja (48 fitur).
  5. Rantai yang dipakai di proyek: PCA HANYA pada HOG + histogram asli.
  6. Sweep jumlah komponen PCA -- untuk menunjukkan "semakin banyak PCA,
     justru semakin buruk" (informasi warna hilang).
  7. Sweep bobot histogram (`HIST_WEIGHT`) -- untuk membenarkan nilai 6.
  8. Block-reduced HOG -- untuk menguji apakah dimensi HOG redundan.

Penting: skrip ini memakai batas fitur dari `src/features.py`, bukan
angka yang ditulis manual. Jadi kalau konfigurasi HOG berubah, skrip
tetap benar.

Jalankan:
    python scripts/diagnostik_fitur.py
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
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src import config as C
from src import features as F

# Klasifikasi ganas: mel, bcc, akiec
MALIGNANT_IDX = [C.CLASS_TO_IDX[c] for c in C.MALIGNANT_CLASSES]


def load_data():
    """Memuat fitur dan split yang sudah disiapkan."""
    X = np.load(C.HOG_FEATURES_NPY)
    ids = pd.read_csv(C.HOG_IDS_CSV)
    tr = pd.read_csv(C.TRAIN_CSV)
    va = pd.read_csv(C.VAL_CSV)

    # Peta image_id -> indeks baris fitur
    posisi = {img_id: i for i, img_id in enumerate(ids["image_id"])}
    X_train = X[[posisi[i] for i in tr["image_id"]]].astype(np.float32)
    y_train = tr["dx_idx"].to_numpy()
    X_val = X[[posisi[i] for i in va["image_id"]]].astype(np.float32)
    y_val = va["dx_idx"].to_numpy()
    return X_train, y_train, X_val, y_val


def evaluate(y_true, y_pred) -> dict:
    """Menghitung metrik utama."""
    pred_mal = np.isin(y_pred, MALIGNANT_IDX).astype(int)
    true_mal = np.isin(y_true, MALIGNANT_IDX).astype(int)
    return {
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "malignant_recall": recall_score(true_mal, pred_mal, zero_division=0),
    }


def svc_rbf():
    """SVC dengan setting yang sama seperti di src/classic_models.py."""
    return SVC(C=10, class_weight="balanced", random_state=C.SEED,
               cache_size=2000)


def rf_model(n_estimators=300):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced_subsample",
        random_state=C.SEED,
        n_jobs=-1,
    )


def chain_pca_hog_plus_hist(
    hog_train, hog_val, H_train, H_val, y_train, y_val,
    n_components: int, hist_weight: float, model="svc",
):
    """
    Rantai yang dipakai di proyek:

        HOG      -> StandardScaler -> PCA(n) -> StandardScaler
        histogram -> dikali bobot (tetap 0..1 * bobot, TIDAK di-scale ulang)
        digabung -> classifier

    Mengembalikan (metrik, selisih_informasi_histogram).
    """
    scaler = StandardScaler().fit(hog_train)
    n_comp = min(n_components, hog_train.shape[1])
    pca = PCA(n_components=n_comp, random_state=C.SEED,
              svd_solver="randomized")
    A = pca.fit_transform(scaler.transform(hog_train))
    B = pca.transform(scaler.transform(hog_val))
    A = StandardScaler().fit_transform(A)
    B = StandardScaler().fit_transform(B)

    A = np.hstack([A, H_train * hist_weight])
    B = np.hstack([B, H_val * hist_weight])

    clf = svc_rbf() if model == "svc" else rf_model()
    clf.fit(A, y_train)
    m = evaluate(y_val, clf.predict(B))

    # Seberapa besar kontribusi histogram terhadap informasi warna?
    # (Fisher-like: jarak antar rerata kelas ganas vs jinak)
    return m


def main() -> int:
    X_train, y_train, X_val, y_val = load_data()
    n_hog, n_hist = F.feature_slice_boundaries()
    n_total = X_train.shape[1]

    print("=" * 78)
    print("DIAGNOSTIK FITUR: kenapa HOG + histogram perlu dirancang ulang")
    print("=" * 78)
    print(f"Train {X_train.shape} | Val {X_val.shape}")
    print(f"HOG {n_hog} fitur | Histogram {n_hist} fitur | "
          f"Total {n_total} fitur")
    print(f"PCA default {C.PCA_N_COMPONENTS} | HIST_WEIGHT {C.HIST_WEIGHT}\n")

    H_train = X_train[:, n_hog:]
    H_val = X_val[:, n_hog:]
    hog_train = X_train[:, :n_hog]
    hog_val = X_val[:, :n_hog]

    rows = []

    def catat(label, m, el):
        print(f"  {label:48s} "
              f"F1={m['f1_macro']:.4f} acc={m['accuracy']:.4f} "
              f"recG={m['malignant_recall']:.4f} ({el:.0f}s)")
        rows.append({"konfigurasi": label, **m, "waktu_s": el})

    # --- 1. Baseline: selalu menebak kelas terbesar ---
    pred_baseline = np.full_like(y_val, C.CLASS_TO_IDX["nv"])
    catat("BASELINE: selalu nebak nv",
          evaluate(y_val, pred_baseline), 0.0)

    # --- 2. Random Forest pada SEMUA fitur (tanpa PCA) ---
    t0 = time.time()
    rf = rf_model()
    rf.fit(X_train, y_train)
    catat(f"RF pada {n_total} fitur penuh",
          evaluate(y_val, rf.predict(X_val)), time.time() - t0)

    # --- 3. SVC dengan PCA yang diterapkan ke SELURUH fitur ---
    # Ini konfigurasi "yang salah": PCA mengambil komponen yang didominasi
    # HOG, sehingga informasi warna hilang.
    print("\n  --- PCA diterapkan ke SEMUA fitur (termasuk histogram) ---")
    for n_comp in (30, 100, 400):
        n_comp = min(n_comp, n_total - 1)
        t0 = time.time()
        scaler = StandardScaler().fit(X_train)
        pca = PCA(n_components=n_comp, random_state=C.SEED,
                  svd_solver="randomized")
        A = pca.fit_transform(scaler.transform(X_train))
        B = pca.transform(scaler.transform(X_val))
        clf = svc_rbf()
        clf.fit(A, y_train)
        catat(f"SVC PCA({n_comp}) dari {n_total} fitur",
              evaluate(y_val, clf.predict(B)), time.time() - t0)

    # --- 4. Histogram warna saja ---
    print("\n  --- Histogram warna saja (48 fitur) ---")
    for label, model in [("SVC-RBF", "svc"), ("RF", "rf")]:
        t0 = time.time()
        clf = svc_rbf() if model == "svc" else rf_model()
        clf.fit(H_train, y_train)
        catat(f"{label} histogram saja ({n_hist})",
              evaluate(y_val, clf.predict(H_val)), time.time() - t0)

    # --- 5 + 6. Rantai yang benar, sweep jumlah komponen PCA ---
    print("\n  --- Rantai yang dipakai proyek: PCA(HOG) + histogram ---")
    for n_comp in (10, 30, 50, 100, 200, 400):
        t0 = time.time()
        m = chain_pca_hog_plus_hist(hog_train, hog_val, H_train, H_val,
                                    y_train, y_val, n_comp, C.HIST_WEIGHT)
        catat(f"SVC PCA({n_comp}) HOG + histogram", m, time.time() - t0)

    # --- 7. Sweep bobot histogram (pada PCA default) ---
    #
    # Sweep ini dijalankan untuk SVM DAN Random Forest, karena model
    # klasik yang dipakai proyek adalah Random Forest. Nilai bobot
    # yang sama belum tentu optimal untuk keduanya.
    print("\n  --- Sweep bobot histogram (PCA default) ---")
    for model in ("svc", "rf"):
        for w in (1.0, 3.0, 6.0, 12.0, 24.0, 48.0, 96.0):
            t0 = time.time()
            m = chain_pca_hog_plus_hist(hog_train, hog_val, H_train, H_val,
                                        y_train, y_val, C.PCA_N_COMPONENTS,
                                        w, model=model)
            nama_model = "SVC" if model == "svc" else "RF "
            catat(f"{nama_model} PCA({C.PCA_N_COMPONENTS}) HOG + hist x{w}",
                  m, time.time() - t0)

    # --- 8. Block-reduced HOG ---
    # Menggabungkan 4 sel HOG per blok menjadi rata-rata.
    n_block = (C.HOG_CELLS_PER_BLOCK[0] * C.HOG_CELLS_PER_BLOCK[1])
    cells = F.hog_dim() // C.HOG_ORIENTATIONS // n_block
    print(f"\n  --- Block-reduced HOG ({cells} blok x "
          f"{C.HOG_ORIENTATIONS} orientasi) ---")

    def block_reduce(hog_matrix: np.ndarray) -> np.ndarray:
        return (hog_matrix.reshape(len(hog_matrix), cells, n_block,
                                    C.HOG_ORIENTATIONS)
                .mean(axis=2)
                .reshape(len(hog_matrix), -1))

    Btr = block_reduce(hog_train)
    Bva = block_reduce(hog_val)
    for n_comp in (30, 100):
        t0 = time.time()
        scaler = StandardScaler().fit(Btr)
        pca = PCA(n_components=min(n_comp, Btr.shape[1]),
                  random_state=C.SEED, svd_solver="randomized")
        A = StandardScaler().fit_transform(
            pca.fit_transform(scaler.transform(Btr)))
        B = StandardScaler().fit_transform(pca.transform(scaler.transform(Bva)))
        A = np.hstack([A, H_train * C.HIST_WEIGHT])
        B = np.hstack([B, H_val * C.HIST_WEIGHT])
        clf = svc_rbf()
        clf.fit(A, y_train)
        catat(f"SVC blockPCA({n_comp}) + histogram",
              evaluate(y_val, clf.predict(B)), time.time() - t0)

    # --- Ringkasan ---
    out = C.DOCS_LOGS / "diagnostik_fitur.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(C.DOCS_ASSETS / "diagnostik_fitur.csv", index=False)
    df.to_json(out, orient="records", indent=2)

    print("\n" + "=" * 78)
    print("RINGKASAN (diurutkan F1 macro)")
    print("=" * 78)
    print(df.sort_values("f1_macro", ascending=False).to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nDisimpan: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
