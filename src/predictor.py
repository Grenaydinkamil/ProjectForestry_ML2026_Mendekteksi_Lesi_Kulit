"""
predictor.py
============
Pembungkus satu antarmuka (interface) untuk menjalankan prediksi, dipakai
oleh aplikasi Streamlit.

Kenapa modul ini ada?
----------------------
Aplikasi web harus menjalankan preprocessing yang **persis sama** dengan
saat training. Kalau ada perbedaan (misal ukuran gambar 224 vs 128, atau
normalisasi berbeda), hasil prediksi di aplikasi bisa jauh melenceng dari
angka yang ada di laporan.

Modul ini mengunci aturan tersebut di satu tempat:

  1. Ukuran gambar diambil dari `models/preprocess_config.json`
     (ditulis otomatis oleh notebook training, bukan diketik manual).
  2. Untuk model klasik: resize -> HOG + histogram -> predict_proba.
     Normalisasi (StandardScaler) ada DI DALAM pipeline sklearn, jadi
     tidak boleh diulang.
  3. Untuk model deep learning: resize -> float32 0..255 -> predict.
     Normalisasi ada DI DALAM model (layer Rescaling), jadi TIDAK boleh
     diulang di sini.

Aturan "jangan normalisasi dua kali" ini yang paling sering dilanggar
dan penyebab paling umum hasil aplikasi tidak cocok dengan laporan.
"""

from __future__ import annotations

import json

import numpy as np
from PIL import Image

from . import config as C
from . import features as F

# Nilai resample PIL. Dipisah agar mudah dituning di satu tempat.
Image_RESAMPLE = Image.BILINEAR


# ---------------------------------------------------------------------------
# 1. MEMUAT KONFIGURASI DAN MODEL
# ---------------------------------------------------------------------------
def load_preprocess_config() -> dict:
    """Membaca konfigurasi preprocessing dari folder models/."""
    path = C.MODELS_DIR / C.PREPROCESS_CONFIG_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"Konfigurasi preprocessing tidak ditemukan: {path}\n"
            "Jalankan notebook training terlebih dahulu."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_label_map() -> dict:
    """Membaca pemetaan label (indeks -> nama kelas) dari folder models/."""
    path = C.MODELS_DIR / C.LABEL_MAP_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"Label map tidak ditemukan: {path}\n"
            "Jalankan notebook training terlebih dahulu."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_threshold_config() -> dict:
    """Membaca threshold ganas (dihitung dari validasi di Notebook 07)."""
    path = C.MODELS_DIR / "threshold_config.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Threshold config tidak ditemukan: {path}\n"
            "Jalankan Notebook 07_evaluasi_perbandingan.ipynb terlebih dahulu."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _model_filename(model_name: str) -> str:
    """Menentukan nama file model berdasarkan nama model yang dipilih."""
    mapping = {
        "SVM (HOG)": "svm_hog.joblib",
        "Random Forest (HOG)": "rf_hog.joblib",
        "CNN (from scratch)": "cnn_best.keras",
        "MobileNetV2": "mobilenetv2_best.keras",
    }
    return mapping.get(model_name, "")


def load_predictor(model_name: str | None = None):
    """
    Memuat model beserta semua yang dibutuhkannya.

    Mengembalikan dict berisi:
      model         : objek model
      model_type    : "classic" atau "deep"
      img_size      : ukuran gambar yang harus dipakai
      label_map     : pemetaan label
      threshold     : threshold ganas (dari validasi)
      model_name    : nama model yang dipakai

    Model disimpan dengan `@st.cache_resource` di level aplikasi, jadi
    fungsi ini dipanggil sekali per sesi browser (bukan tiap interaksi).
    """
    if model_name is None:
        thr = load_threshold_config()
        model_name = thr["selected_model"]

    label_map = load_label_map()
    cfg = load_preprocess_config()
    threshold = load_threshold_config().get("threshold", 0.5)

    filename = _model_filename(model_name)
    if not filename:
        raise ValueError(
            f"Model '{model_name}' tidak dikenal. "
            f"Pilihan yang tersedia: {list(_model_filename.__defaults__ or [])}"
        )

    path = C.MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"File model tidak ditemukan: {path}\n"
            f"Jalankan notebook training untuk model '{model_name}'."
        )

    if filename.endswith(".joblib"):
        import joblib

        return {
            "model": joblib.load(path),
            "model_type": "classic",
            "img_size": int(cfg.get("img_size", C.IMG_SIZE_CLASSIC)),
            "label_map": label_map,
            "threshold": threshold,
            "model_name": model_name,
        }

    import tensorflow as tf

    model = tf.keras.models.load_model(path)
    return {
        "model": model,
        "model_type": "deep",
        # Ukuran gambar diambil dari model itu sendiri, sumber paling otoritatif
        "img_size": int(model.input_shape[1]),
        "label_map": label_map,
        "threshold": threshold,
        "model_name": model_name,
    }


# ---------------------------------------------------------------------------
# 2. PREPROCESSING UNTUK INFERENSI
# ---------------------------------------------------------------------------
def preprocess_for_inference(img, img_size: int, model_type: str):
    """
    Menyiapkan satu gambar untuk inferensi.

    Mengembalikan array float32 dengan bentuk (1, H, W, 3).

    PENTING: nilai dikembalikan dalam rentang 0..255 (bukan 0..1).
    Normalisasi ke rentang target dilakukan oleh layer Rescaling DI DALAM
    model, sehingga:
      - tidak ada risiko normalisasi dilakukan dua kali
      - preprocessing di aplikasi identik dengan training
    """
    arr = np.asarray(img.convert("RGB").resize(
        (img_size, img_size), Image_RESAMPLE), dtype=np.float32)
    return arr[None, ...]


# ---------------------------------------------------------------------------
# 3. PREDIKSI
# ---------------------------------------------------------------------------
def predict(predictor: dict, image) -> dict:
    """
    Menjalankan prediksi pada satu gambar.

    Mengembalikan dict:
      class_names   : daftar nama kelas sesuai urutan indeks
      probabilities : array probabilitas (7,)
      predicted_idx : indeks kelas terpilih
      predicted_cls : kode kelas (misal "mel")
      risk_malignant: jumlah probabilitas kelas ganas
      threshold     : threshold yang dipakai
      is_uncertain  : True bila prediksi berada di area abu-abu
    """
    model = predictor["model"]
    img_size = predictor["img_size"]
    label_map = predictor["label_map"]
    classes = label_map["classes"]
    malignant = label_map["malignant_classes"]

    if predictor["model_type"] == "classic":
        small = image.convert("RGB").resize((img_size, img_size),
                                            Image_RESAMPLE)
        feat = F.extract_features_single(small, img_size)
        proba = model.predict_proba(feat)[0]
    else:
        arr = preprocess_for_inference(image, img_size, predictor["model_type"])
        proba = model.predict(arr, verbose=0)[0]

    pred_idx = int(np.argmax(proba))
    mal_idx = [classes.index(c) for c in malignant]
    risk = float(proba[mal_idx].sum())

    return {
        "class_names": classes,
        "dx_indo": label_map.get("dx_indo", {}),
        "dx_full_name": label_map.get("dx_full_name", {}),
        "probabilities": proba,
        "predicted_idx": pred_idx,
        "predicted_cls": classes[pred_idx],
        "risk_malignant": risk,
        "threshold": predictor["threshold"],
        "malignant_classes": malignant,
        "benign_classes": label_map.get("benign_classes", []),
    }


def is_uncertain(result: dict, band: float | None = None) -> bool:
    """
    Menentukan apakah prediksi berada di area abu-abu (tidak yakin).

    Logika: model dianggap "tidak yakin" kalau salah satu dari dua
    kondisi ini terpenuhi:
      1. **Risiko ganas sangat dekat dengan ambang.** Kalau angka risiko
         hampir sama dengan ambang, keputusan "ganas atau jinak" bisa
         berubah hanya karena satu piksel berbeda.
      2. **Modelnya sendiri tidak yakin.** Probabilitas kelas teratas
         terlalu rendah, artinya model tidak benar-benar memilih satu
         diagnosis.

    Kenapa lebar area abu-abu (band) TIDAK dibuat tetap?
      Nilai ambang berubah depending model (mis. 0,11 untuk MobileNetV2
      di proyek ini, tapi 0,25 untuk model lain). Kalau band-nya tetap
      0,15, maka seluruh rentang risiko jinak (0 sampai 0,11) akan
      masuk ke dalam area abu-abu, dan hampir semua foto akan diberi
      label "belum meyakinkan" -- kotak hijau "Tidak ditemukan ciri
      ganas" nyaris tidak pernah muncul. Jadi bandnya **mengikuti
      skala ambang**: 30% dari ambang, dengan minimal 0,02.

    Parameter:
      band : lebar area abu-abu di sekitar ambang. None = dihitung
             otomatis dari ambang.
    """
    risk = result["risk_malignant"]
    threshold = result["threshold"]
    top_prob = float(result["probabilities"].max())

    if band is None:
        band = max(0.02, 0.30 * threshold)

    dekat_threshold = abs(risk - threshold) < band
    model_ragu = top_prob < 0.45

    return dekat_threshold or model_ragu
