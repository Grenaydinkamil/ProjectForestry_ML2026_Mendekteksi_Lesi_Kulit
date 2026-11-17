"""
src/__init__.py
==============
Paket src berisi seluruh fungsi reusable proyek:

  config.py       -> konstanta terpusat (ukuran gambar, kelas, path, seed)
  data_loader.py  -> pemuatan & verifikasi dataset
  features.py     -> ekstraksi fitur HOG + histogram warna (model klasik)
  preprocessing.py-> preprocessing citra untuk CNN/MobileNetV2
  modeling.py     -> pembuatan model deep learning + training loop
  evaluation.py   -> metrik, confusion matrix, laporan per kelas

Modul-modul ini diimpor dengan `from src import config as C` dari notebook,
dan `from src import ...` dari aplikasi Streamlit. Konsekuensinya, preprocessing
yang dipakai notebook dan aplikasi dijamin identik.
"""

__all__ = [
    "config",
    "data_loader",
    "features",
    "preprocessing",
    "modeling",
    "evaluation",
]
