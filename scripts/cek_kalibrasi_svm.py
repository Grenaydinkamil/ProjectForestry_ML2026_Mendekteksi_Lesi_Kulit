"""
Bandingkan cara mendapatkan probabilitas dari SVM.

Masalah yang ditemukan:
  `src/classic_models.build_svm()` membungkus pipeline SVC dengan
  `CalibratedClassifierCV(..., ensemble=False)`. Pengukuran pada data
  validasi menunjukkan F1 macro-nya hanya 0,2153 dengan recall ganas
  0,1859 -- jauh lebih buruk daripada SVC polos yang menghasilkan
  F1 macro 0,3423 dan recall ganas 0,4776 (lihat
  scripts/diagnostik_fitur.py).

  Kenapa bisa begitu? Dengan `ensemble=False`, sklearn menghitung
  kalibrator dari prediksi out-of-fold, lalu MELATIH ULANG estimator
  dasar pada SELURUH data latih. Decision function model yang dilatih
  ulang itu bisa punya skala berbeda dari model lipatan yang dipakai
  untuk fitting kalibrator, sehingga sigmoid-nya tidak lagi "seimbang"
  dan argmax pun berubah.

  Tapi SVM wajib punya `predict_proba`, karena ambang batas "ganas/jinak"
  dihitung dari probabilitas.

Skrip ini menguji empat cara dan mengukur semuanya di data validasi:

  1. SVC polos (tanpa probabilitas) -- acuan akurasi
  2. SVC(probability=True) -- kalibrasi internal Platt (deprecated di
     scikit-learn 1.9, tapi masih berfungsi sampai 1.11)
  3. CalibratedClassifierCV(ensemble=False) -- yang dipakai sekarang
  4. CalibratedClassifierCV(ensemble=True)  -- rata-rata model lipatan

Jalankan:
    python scripts/cek_kalibrasi_svm.py
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.svm import SVC

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
    prep = CM.build_preprocessor()

    print("=" * 78)
    print("CARA MENDAPATKAN PROBABILITAS DARI SVM (di data validasi)")
    print("=" * 78)

    baris = []

    # --- 1. SVC polos: acuan akurasi (tanpa probabilitas) ---
    t0 = time.time()
    pipe = CM.build_preprocessor()
    Ztr = pipe.fit_transform(Xtr, ytr)
    Zva = pipe.transform(Xva)
    svc = SVC(C=10, class_weight="balanced", random_state=C.SEED,
              cache_size=2000)
    svc.fit(Ztr, ytr)
    pred_plain = svc.predict(Zva)
    m = evaluate(yva, pred_plain)
    print(f"{'SVC polos (acuan)':<40}{m['f1_macro']:>10.4f}"
          f"{m['accuracy']:>10.4f}{m['malignant_recall']:>12.4f}"
          f"{time.time() - t0:>8.0f}s")
    baris.append({"cara": "SVC polos (acuan)", **m})

    # --- 2. SVC(probability=True) ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.time()
        pipe2 = CM.build_preprocessor()
        Ztr2 = pipe2.fit_transform(Xtr, ytr)
        Zva2 = pipe2.transform(Xva)
        svc2 = SVC(C=10, class_weight="balanced", random_state=C.SEED,
                   cache_size=2000, probability=True)
        svc2.fit(Ztr2, ytr)
        proba2 = svc2.predict_proba(Zva2)
        m = evaluate(yva, proba2.argmax(axis=1))
        print(f"{'SVC(probability=True)':<40}{m['f1_macro']:>10.4f}"
              f"{m['accuracy']:>10.4f}{m['malignant_recall']:>12.4f}"
              f"{time.time() - t0:>8.0f}s")
        baris.append({"cara": "SVC(probability=True)", **m})
        baris[-1]["waktu_s"] = time.time() - t0

    # --- 3 & 4. CalibratedClassifierCV dengan ensemble False dan True ---
    for ensemble in (False, True):
        t0 = time.time()
        pipe3 = CM.build_preprocessor()
        Ztr3 = pipe3.fit_transform(Xtr, ytr)
        Zva3 = pipe3.transform(Xva)
        base = SVC(C=10, class_weight="balanced", random_state=C.SEED,
                   cache_size=2000)
        clf = CalibratedClassifierCV(base, method="sigmoid", cv=5,
                                      ensemble=ensemble)
        clf.fit(Ztr3, ytr)
        proba3 = clf.predict_proba(Zva3)
        m = evaluate(yva, proba3.argmax(axis=1))
        nama = (f"CalibratedClassifierCV(ensemble={ensemble})")
        print(f"{nama:<40}{m['f1_macro']:>10.4f}"
              f"{m['accuracy']:>10.4f}{m['malignant_recall']:>12.4f}"
              f"{time.time() - t0:>8.0f}s")
        baris.append({"cara": nama, **m, "waktu_s": time.time() - t0})

    print("=" * 78)
    print(f"{'cara':<40}{'F1 macro':>10}{'accuracy':>10}{'rec ganas':>12}")
    print("-" * 78)

    out = C.DOCS_LOGS / "cek_kalibrasi_svm.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(baris)
    df.to_json(out, orient="records", indent=2)
    df.to_csv(C.DOCS_ASSETS / "cek_kalibrasi_svm.csv", index=False)
    print(f"\nDisimpan: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
