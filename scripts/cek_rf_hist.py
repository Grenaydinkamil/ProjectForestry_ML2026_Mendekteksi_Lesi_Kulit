"""
Cek apakah Random Forest benar-benar tidak terpengaruh bobot histogram.

Di `scripts/diagnostik_fitur.py`, hasil sweep menunjukkan RF memberi angka
yang PERSIS SAMA untuk bobot 3, 6, 12, 24, 48, dan 96. Itu mencurigakan:
bobot 96 seharusnya membuat fitur histogram mendominasi.

Skrip ini memverifikasi langsung:
  1. Apakah nilai histogram di feature cache memang bervariasi?
  2. Apakah prediksi RF benar-benar identik untuk bobot berbeda?
  3. Seberapa besar importance fitur histogram di model RF?
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src import config as C
from src import features as F

X = np.load(C.HOG_FEATURES_NPY)
ids = pd.read_csv(C.HOG_IDS_CSV)
tr = pd.read_csv(C.TRAIN_CSV)
va = pd.read_csv(C.VAL_CSV)

posisi = {img_id: i for i, img_id in enumerate(ids["image_id"])}
n_hog, n_hist = F.feature_slice_boundaries()

Xtr = X[[posisi[i] for i in tr["image_id"]]].astype(np.float32)
Xva = X[[posisi[i] for i in va["image_id"]]].astype(np.float32)
ytr = tr["dx_idx"].to_numpy()
yva = va["dx_idx"].to_numpy()

Htr, Hva = Xtr[:, n_hog:], Xva[:, n_hog:]
hogtr, hogva = Xtr[:, :n_hog], Xva[:, :n_hog]

print(f"batas fitur: HOG={n_hog} hist={n_hist} total={Xtr.shape[1]}")
print(f"histogram: min={Htr.min():.4f} max={Htr.max():.4f} "
      f"mean={Htr.mean():.4f} std={Htr.std():.4f}")
print(f"jumlah per baris (harus ~1.0): "
      f"{Htr.sum(axis=1).min():.4f} .. {Htr.sum(axis=1).max():.4f}")

scaler = StandardScaler().fit(hogtr)
pca = PCA(n_components=C.PCA_N_COMPONENTS, random_state=C.SEED,
          svd_solver="randomized")
A = StandardScaler().fit_transform(pca.fit_transform(scaler.transform(hogtr)))
B = StandardScaler().fit_transform(pca.transform(scaler.transform(hogva)))

# PENTING: matriks yang dilatih RF hanya berisi komponen PCA (30 kolom)
# + histogram (48 kolom) = 78 kolom, BUKAN 1.812 kolom fitur asli.
# Jadi batas potong untuk feature importance adalah n_comp, bukan n_hog.
n_comp = A.shape[1]
print(f"matriks RF: {n_comp} komponen PCA + {n_hist} histogram "
      f"= {n_comp + n_hist} kolom\n")

prediksi = {}
for w in (1.0, 3.0, 6.0, 24.0, 96.0, 384.0):
    Aw = np.hstack([A, Htr * w])
    Bw = np.hstack([B, Hva * w])
    rf = RandomForestClassifier(n_estimators=300,
                                class_weight="balanced_subsample",
                                random_state=C.SEED, n_jobs=-1)
    rf.fit(Aw, ytr)
    pred = rf.predict(Bw)
    prediksi[w] = pred

    imp = rf.feature_importances_
    imp_hist = imp[n_comp:].sum()
    imp_pca = imp[:n_comp].sum()

    print(f"w={w:>6}: pred unik={len(np.unique(pred))} "
          f"importance PCA={imp_pca:.4f} hist={imp_hist:.4f} "
          f"| hist mean={Htr.mean() * w:.3f} vs PCA std={A.std():.3f}")

print("\nBerapa banyak prediksi yang BERBEDA antar bobot?")
daftar = sorted(prediksi)
for i, w1 in enumerate(daftar):
    for w2 in daftar[i + 1:]:
        beda = int((prediksi[w1] != prediksi[w2]).sum())
        if beda:
            print(f"  w={w1} vs w={w2}: {beda} prediksi berbeda "
                  f"({beda / len(yva):.1%})")
