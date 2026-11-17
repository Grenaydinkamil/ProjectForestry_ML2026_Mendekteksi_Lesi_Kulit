"""
Cek bobot histogram untuk model SVM yang DIPAKAI PROYEK.

Pertanyaan: nilai `HIST_WEIGHT` berapa yang terbaik untuk pipeline SVM
yang benar-benar dipakai proyek (`src/classic_models.build_svm`, yang
membungkus SVC dalam CalibratedClassifierCV)?

Kenapa perlu skrip ini?
  `scripts/diagnostik_fitur.py` memakai SVC polos untuk perbandingan
  cepat. Tapi model proyek memakai CalibratedClassifierCV (dibutuhkan
  supaya ada `predict_proba` untuk tuning threshold). Kalibrasi bisa
  mengubah prediksi, jadi hasil diagnostic TIDAK otomatis berlaku untuk
  model proyek. Skrip ini mengukurnya langsung.

Skrip ini juga menguji Random Forest, untuk membuktikan bahwa bobot
histogram TIDAK berpengaruh pada model pohon (pohon membelah
berdasarkan urutan nilai, jadi skala fitur tidak berpengaruh).

Jalankan:
    python scripts/cek_bobot_histogram.py
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
from sklearn.metrics import accuracy_score, f1_score, recall_score

from src import config as C
from src import classic_models as CM
from src import features as F

MALIGNANT_IDX = [C.CLASS_TO_IDX[c] for c in C.MALIGNANT_CLASSES]


def load():
    X = np.load(C.HOG_FEATURES_NPY)
    ids = pd.read_csv(C.HOG_IDS_CSV)
    tr = pd.read_csv(C.TRAIN_CSV)
    va = pd.read_csv(C.VAL_CSV)
    posisi = {img_id: i for i, img_id in enumerate(ids["image_id"])}
    Xtr = X[[posisi[i] for i in tr["image_id"]]].astype(np.float32)
    Xva = X[[posisi[i] for i in va["image_id"]]].astype(np.float32)
    return Xtr, tr["dx_idx"].to_numpy(), Xva, va["dx_idx"].to_numpy()


def evaluate(y_true, y_pred) -> dict:
    pred_mal = np.isin(y_pred, MALIGNANT_IDX).astype(int)
    true_mal = np.isin(y_true, MALIGNANT_IDX).astype(int)
    return {
        "f1_macro": f1_score(y_true, y_pred, average="macro",
                              zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "malignant_recall": recall_score(true_mal, pred_mal,
                                          zero_division=0),
    }


def main() -> int:
    Xtr, ytr, Xva, yva = load()
    n_hog, n_hist = F.feature_slice_boundaries()

    print("=" * 78)
    print("CEK BOBOT HISTOGRAM UNTUK MODEL PROYEK")
    print("=" * 78)
    print(f"Fitur: HOG {n_hog} + histogram {n_hist} = {Xtr.shape[1]}")
    print(f"Model: SVM (CalibratedClassifierCV) dan Random Forest\n")
    print(f"{'model':<10}{'bobot':>8}{'F1 macro':>11}{'accuracy':>11}"
          f"{'rec ganas':>11}{'waktu':>8}")
    print("-" * 78)

    baris = []

    for w in (1.0, 3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 384.0):
        # --- SVM proyek ---
        t0 = time.time()
        clf = CM.build_svm(hist_weight=w)
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xva)
        m = evaluate(yva, pred)
        el = time.time() - t0
        print(f"{'SVM':<10}{w:>8.0f}{m['f1_macro']:>11.4f}"
              f"{m['accuracy']:>11.4f}{m['malignant_recall']:>11.4f}"
              f"{el:>7.0f}s")
        baris.append({"model": "SVM", "hist_weight": w, **m,
                      "waktu_s": el})

    print("-" * 78)

    # --- Random Forest proyek (hanya beberapa bobot, sudah jelas insensitive) ---
    for w in (1.0, 6.0, 96.0):
        t0 = time.time()
        clf = CM.build_random_forest(hist_weight=w)
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xva)
        m = evaluate(yva, pred)
        el = time.time() - t0
        print(f"{'RF':<10}{w:>8.0f}{m['f1_macro']:>11.4f}"
              f"{m['accuracy']:>11.4f}{m['malignant_recall']:>11.4f}"
              f"{el:>7.0f}s")
        baris.append({"model": "RF", "hist_weight": w, **m,
                      "waktu_s": el})

    print("=" * 78)
    print("Catatan: Random Forest tidak terpengaruh bobot histogram,")
    print("karena pohon membelah berdasarkan URUTAN nilai, bukan jarak.")
    print("Bobot hanya berpengaruh pada SVM (RBF) yang menghitung jarak.")

    out = C.DOCS_LOGS / "cek_bobot_histogram.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(baris)
    df.to_json(out, orient="records", indent=2)
    df.to_csv(C.DOCS_ASSETS / "cek_bobot_histogram.csv", index=False)
    print(f"\nDisimpan: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
