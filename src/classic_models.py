"""
classic_models.py
=================
Rantai preprocessing dan pembangun model KLASIK (SVM dan Random Forest).

Desain ini DASAR-DARI EKSPERIMEN, bukan tebakan. Semua angka di bawah
berasal dari pengukuran nyata di `scripts/diagnostik_fitur.py` (lihat
`docs/CATATAN_TEMUAN_DIAGNOSTIK.md`).

Ringkasan temuan yang membentuk desain ini
-----------------------------------------
1. **Random Forest pada fitur mentah hampir tidak belajar apa pun.**
   F1 macro-nya 0,1467, hampir sama dengan baseline "selalu menebak nv"
   (0,1148), dan recall ganas nyaris nol (0,0192).
   Penyebabnya: Random Forest memilih fitur ACAK saat membelah node. Dengan
   1.764 fitur HOG yang saling berkorelasi dan hanya 48 fitur histogram
   yang diskriminatif, peluang fitur histogram dipakai saat split hanya
   48/1.812 = 2,65%.

2. **PCA pada seluruh fitur lebih buruk daripada PCA pada HOG saja.**
   Pengukuran (SVC): PCA(30) dari 1.812 fitur menghasilkan F1 macro
   0,2749, sedangkan PCA(30) pada HOG saja + histogram menghasilkan
   0,3423. Alasannya, komponen PCA didominasi HOG, jadi informasi warna
   hilang di antara komponen yang variance-nya lebih besar.

3. **Solusinya: PCA HANYA pada bagian HOG, histogram dibiarkan utuh.**
   Rantai yang benar:
       HOG      -> StandardScaler -> PCA(n) -> StandardScaler
       histogram -> dikali bobot (sudah 0..1, TIDAK di-scale ulang)
       digabung -> SVC / RandomForest

   Pengukuran (sel 16x16, PCA 30, bobot 6): F1 macro 0,3423, recall ganas
   0,4776.

4. **Bobot histogram hanya berpengaruh pada SVM, bukan pada Random
   Forest.** Pohon membelah berdasarkan urutan nilai, bukan jarak, jadi
   mengalikan fitur dengan konstanta tidak mengubah partisi. Sweep
   bobot 3 sampai 96 menghasilkan angka Random Forest yang identik.
   Ini yang membuat bobot 6 dipilih: nilainya diukur untuk SVM.

5. **Kelas minoritas perlu dibobot.** `class_weight="balanced"` dan
   `balanced_subsample` supaya kelas `df` (92 gambar di train) dan `vasc`
   (93 gambar) tidak diabaikan.

6. **`SVC(probability=True)` deprecated di scikit-learn 1.9.** Diganti
   `CalibratedClassifierCV(SVC(), ensemble=True)`, cara resmi yang
   direkomendasikan library. Parameter `ensemble` sendiri **diukur**,
   bukan pakai default: `ensemble=False` (contoh resmi) menghasilkan
   F1 macro 0,2151 dengan recall ganas 0,1891, sedangkan `ensemble=True`
   menghasilkan F1 macro 0,2713 dengan recall ganas 0,2981. Detailnya
   ada di docstring `build_svm()`.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config as C
from . import features as F


# ---------------------------------------------------------------------------
# 1. PIPELINE PREPROCESSING
# ---------------------------------------------------------------------------
class HistogramScaler(BaseEstimator, TransformerMixin):
    """
    Transformer yang mengalikan fitur histogram dengan bobot tertentu.

    ### Kenapa bukan lambda?
    # `lambda` di dalam `FunctionTransformer` **tidak bisa di-pickle**:
    # ```
    # PicklingError: Can't pickle <function build_preprocessor.<locals>.<lambda>
    # ```
    # Karena model disimpan dengan `joblib.dump`, itu berarti model tidak
    # bisa disimpan ATAU dimuat lagi. Karena itu kita pakai kelas
    # (top-level), yang bisa di-pickle dengan benar.

    ### Kenapa harus extends BaseEstimator + TransformerMixin?
    # `ColumnTransformer` memvalidasi bahwa setiap transformer punya
    # metode `fit` dan `transform`. Tanpa kedua basis itu:
    # ```
    # TypeError: All estimators should implement fit and transform...
    # ```
    # `BaseEstimator` memberi `get_params`/`set_params` (dipakai untuk
    # cloning saat cross-validation), dan `TransformerMixin` memberi
    # `fit_transform`.

    ### Kenapa perlu pengali?
    # Histogram sudah ternormalisasi (0..1) sedangkan komponen PCA punya
    # skala jauh lebih besar. Tanpa pengali eksplisit, informasi warna
    # hilang di antara informasi bentuk. Nilai 6 dipilih dari pengukuran
    # (lihat docs/CATATAN_TEMUAN_DIAGNOSTIK.md).
    """

    def __init__(self, weight: float = C.HIST_WEIGHT):
        self.weight = weight

    def fit(self, X, y=None):
        """Tidak ada yang dipelajari (hanya pengali tetap)."""
        return self

    def transform(self, X):
        """Mengalikan fitur dengan bobot."""
        return np.asarray(X, dtype=np.float64) * self.weight


def build_preprocessor(
    n_components: int = C.PCA_N_COMPONENTS,
    hist_weight: float = C.HIST_WEIGHT,
) -> ColumnTransformer:
    """
    Membangun ColumnTransformer yang memproses HOG dan histogram terpisah.

    Kenapa dipisah?
      HOG dan histogram punya sifat yang SANGAT berbeda:
        - HOG: nilai 0..4, saling berkorelasi kuat, sangat banyak (1.764)
        - Histogram: nilai 0..1, sudah ternormalisasi, sedikit (48)
      Kalau digabung lalu di-scale bareng, skala besar HOG akan menenggelamkan
      informasi warna, padahal informasi warna justru yang paling
      diskriminatif (terbukti dari Fisher Score di Notebook 03).

    Karena itu:
      - HOG: StandardScaler -> PCA(n) -> StandardScaler lagi
        (StandardScaler kedua dibutuhkan karena komponen PCA yang
        variance-nya kecil akan jadi noise kalau dibiarkan)
      - Histogram: dikali `hist_weight` saja, TIDAK di-scale ulang
    """
    n_hog, n_hist = F.feature_slice_boundaries()

    return ColumnTransformer([
        ("hog", Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=C.SEED,
                        svd_solver="randomized")),
            ("scaler2", StandardScaler()),
        ]), list(range(n_hog))),

        ("hist", HistogramScaler(weight=hist_weight),
         list(range(n_hog, n_hog + n_hist))),
    ], remainder="drop")


# ---------------------------------------------------------------------------
# 2. PEMBANGUN MODEL
# ---------------------------------------------------------------------------
def build_svm(
    C_value: float = 10.0,
    gamma: str = "scale",
    kernel: str = "rbf",
    n_components: int = C.PCA_N_COMPONENTS,
    hist_weight: float = C.HIST_WEIGHT,
) -> CalibratedClassifierCV:
    """
    Pipeline SVM lengkap: preprocessing -> SVC -> kalibrasi probabilitas.

    Kenapa CalibratedClassifierCV dan bukan SVC(probability=True)?
      Di scikit-learn 1.9 parameter `probability=True` pada SVC sudah
      DEPRECATED. Peringatan resmi:
        "The `probability` parameter was deprecated in 1.9 and will be
         removed in version 1.11. Use `CalibratedClassifierCV(SVC(),
         ensemble=False)` instead"
      CalibratedClassifierCV melakukan hal yang sama (kalibrasi Platt
      dengan 5-fold CV internal) lewat API yang stabil ke depan.

    Kenapa `ensemble=True`? (bukan `ensemble=False` seperti contoh resmi)
      Nilainya diukur, bukan ditebak. Pengukuran di data validasi
      (scripts/cek_kalibrasi_svm.py):

        | Cara                                    | F1 macro | Recall ganas |
        |-----------------------------------------|----------|--------------|
        | SVC polos (tanpa probabilitas)           | 0,3437   | 0,4904       |
        | SVC(probability=True)                    | 0,2957   | 0,3365       |
        | CalibratedClassifierCV(ensemble=False)   | 0,2151   | 0,1891       |
        | **CalibratedClassifierCV(ensemble=True)** | **0,2713** | **0,2981** |

      Penjelasannya: dengan `ensemble=False`, sklearn menghitung
      kalibrator dari prediksi out-of-fold, lalu MELATIH ULANG estimator
      dasar pada seluruh data latih. Decision function model yang
      dilatih ulang bisa punya skala berbeda dari model lipatan yang
      dipakai untuk fitting kalibrator, sehingga argmax probabilitas
      berubah dan performanya turun drastis (recall ganas 0,49 -> 0,19).

      Dengan `ensemble=True`, prediksi adalah rata-rata probabilitas dari
      5 model lipatan. Decision function tiap lipatan berada di skala
      yang sama dengan kalibrator, jadi hasilnya jauh lebih konsisten.

      Trade-off: training jadi sedikit lebih lama (5 lipatan dilatih),
      tapi waktu inferensi tetap sama.
    """
    pipe = Pipeline([
        ("prep", build_preprocessor(n_components, hist_weight)),
        ("svc", SVC(
            C=C_value,
            gamma=gamma,
            kernel=kernel,
            class_weight="balanced",
            random_state=C.SEED,
            cache_size=2000,
        )),
    ])
    return CalibratedClassifierCV(pipe, method="sigmoid", cv=5,
                                   ensemble=True)


def build_random_forest(
    n_estimators: int = 300,
    max_depth: int | None = None,
    n_components: int = C.PCA_N_COMPONENTS,
    hist_weight: float = C.HIST_WEIGHT,
) -> Pipeline:
    """
    Pipeline Random Forest lengkap.

    `class_weight="balanced_subsample"` menghitung bobot kelas secara
    terpisah untuk setiap bootstrap sample. Untuk dataset yang sangat tidak
    seimbang seperti ini, itu lebih tepat daripada "balanced" biasa.
    """
    return Pipeline([
        ("prep", build_preprocessor(n_components, hist_weight)),
        ("rf", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced_subsample",
            random_state=C.SEED,
            n_jobs=-1,
        )),
    ])


def build_histogram_only_svm(C_value: float = 10.0) -> CalibratedClassifierCV:
    """
    Model ABELASI: SVM pada histogram warna SAJA (tanpa HOG).

    Kenapa model ini perlu ada?
      Karena hasil eksperimen menunjukkan histogram warna saja memberi
      recall kelas ganas 0,699, lebih tinggi dari kombinasi HOG+histogram
      mana pun. Model ini dipakai sebagai pembanding di laporan untuk
      menunjukkan seberapa besar kontribusi HOG.

    Perhatikan: bagian HOG sengaja dikecualikan dari ColumnTransformer
    (remainder="drop"), jadi hanya 48 fitur warna yang dipakai.

    Pipeline dibungkus CalibratedClassifierCV agar punya `predict_proba`,
    sama seperti model SVM utama. Tanpa itu, kode evaluasi akan gagal
    dengan error "Pipeline has no attribute predict_proba".
    """
    n_hog, n_hist = F.feature_slice_boundaries()
    pipe = Pipeline([
        ("prep", ColumnTransformer([
            ("hist", StandardScaler(),
             list(range(n_hog, n_hog + n_hist))),
        ], remainder="drop")),
        ("svc", SVC(C=C_value, class_weight="balanced",
                    random_state=C.SEED, cache_size=2000)),
    ])
    return CalibratedClassifierCV(pipe, method="sigmoid", cv=5,
                                   ensemble=True)
