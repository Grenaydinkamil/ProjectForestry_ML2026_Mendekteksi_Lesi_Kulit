"""
config.py
=========
Konfigurasi terpusat untuk seluruh proyek.

Kenapa file ini ada?
-------------------
Angka-angka seperti "ukuran gambar 128x128" atau "nama folder gambar" dipakai
di banyak tempat: notebook EDA, notebook training, notebook evaluasi, sampai
aplikasi Streamlit. Kalau angka itu ditulis ulang di tiap file, cepat atau
lambat nilainya akan berbeda antar bagian, dan hasil prediksi di aplikasi bisa
tidak cocok dengan hasil training. Dengan memusatkannya di sini, satu kali
ubah berlaku untuk semua.

Semua nilai yang dipakai di sini dapat direproduksi karena dataset dan seed
sudah ditetapkan (lihat SEED di bawah).
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. SEED (benih acak)
# ---------------------------------------------------------------------------
# Kenapa perlu seed?
#   Training machine learning punya dua sumber ketidakpastian:
#   (a) inisialisasi bobot model (neuron/pohon acak)
#   (b) urutan data latih (shuffle) dan augmentasi acak
#   Dengan SEED yang sama, dua kali menjalankan kode menghasilkan angka yang
#   sama persis. Ini syarat agar "hasil evaluasi benar-benar dijalankan"
#   bisa dipertanggungjawabkan saat tanya jawab.
SEED = 42

# ---------------------------------------------------------------------------
# 2. LOKASI FOLDER
# ---------------------------------------------------------------------------
# Repo root = folder yang berisi folder "notebooks", "src", "models", dst.
# resolved() dipakai supaya path menjadi absolut dan tidak berubah karena
# perbedaan working directory.
REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_PROCESSED = REPO_ROOT / "data" / "processed"  # hasil split, fitur .npy
DATA_EXTERNAL = REPO_ROOT / "data" / "external"    # PAD-UFES-20 (uji eksternal)
MODELS_DIR = REPO_ROOT / "models"                  # model terlatih
DOCS_ASSETS = REPO_ROOT / "docs" / "assets"        # gambar untuk PPT dan laporan
DOCS_LOGS = REPO_ROOT / "docs" / "logs"            # metrik JSON hasil evaluasi


# ---------------------------------------------------------------------------
# 1b. OVERRIDE LOKASI DATA (untuk training di WSL2)
# ---------------------------------------------------------------------------
# Kenapa perlu?
#   Training model deep learning di WSL2 (untuk memakai GPU) jauh lebih
#   cepat kalau file gambarnya ada di disk NATIVE WSL, bukan di drive
#   Windows yang di-mount ke /mnt/. Pembacaan file lewat /mnt/ (protokol
#   9p/drvfs) punya latensi tinggi, dan di.sinilah bottleneck_tf.data.
#
#   Variable lingkungan SKIN_DATA_ROOT dipakai untuk menunjuk ke salinan
#   data di disk WSL. Kalau tidak di-set, dipakai lokasi default di repo.
#
#   Cara pakai (dari PowerShell):
#     wsl -d Ubuntu-26.04 -- bash -c \
#       "export SKIN_DATA_ROOT=/mnt/f/ml_project_data/raw && \
#        /opt/venvs/tf312/bin/jupyter lab"
#
#   PENTING: hanya untuk speeding training. Hasil dan model tetap disimpan
#   di repo (models/, docs/logs/), jadi tidak ada yang hilang.
DATA_ROOT_OVERRIDE = os.environ.get("SKIN_DATA_ROOT")

if DATA_ROOT_OVERRIDE:
    DATA_RAW = Path(DATA_ROOT_OVERRIDE)
else:
    DATA_RAW = REPO_ROOT / "data" / "raw"


def to_relative_path(absolute_path) -> str:
    """
    Mengubah path absolut menjadi path RELATIF terhadap root repo.

    Kenapa relatif, bukan absolut?
      Proyek ini bisa dijalankan di dua lingkungan:
        (a) Windows native -> repo di "M:\\GrenPlums Roses\\..."
        (b) WSL2 untuk GPU -> repo di "/mnt/m/GrenPlums Roses/..."
      Kalau CSV menyimpan path absolut Windows, path itu tidak ada di WSL,
      dan notebook akan gagal membaca gambar dengan error yang membingungkan
      ("file not found" padahal filenya ada).

      Path relatif aman di keduanya, karena selalu dihitung relatif ke
      REPO_ROOT yang dihitung dari lokasi file src/config.py.

    Kenapa separator-nya forward slash ("/")?
      Forward slash dipakai di Windows DAN Linux, jadi satu file CSV bisa
      dibaca di kedua sistem tanpa modifikasi. Backslash di Linux
      diperlakukan sebagai karakter biasa dalam nama file, bukan pemisah
      path, dan itu penyebab paling umum "file not found" saat dataset
      dipindah dari Windows ke WSL.
    """
    try:
        relatif = Path(absolute_path).resolve().relative_to(REPO_ROOT)
        return str(relatif).replace("\\", "/")
    except ValueError:
        # Di luar repo root (misal dataset eksternal) -> pakai absolut
        return str(absolute_path).replace("\\", "/")

# ---------------------------------------------------------------------------
# 3. NAMA FILE DATASET
# ---------------------------------------------------------------------------
# CATATAN PENTING: nama file di bawah ini diverifikasi langsung dari isi ZIP
# yang diunduh, bukan diasumsikan. Isi ZIP "Datasheet Skin Cancer MNIST
# HAM10000.zip" memuat folder gambar dipecah menjadi DUA bagian (part_1 dan
# part_2), bukan satu folder "HAM10000_images" seperti versi di Kaggle.
# Kalau kod mengasumsikan satu folder, semua gambar akan "tidak ditemukan".
METADATA_CSV = DATA_RAW / "HAM10000_metadata.csv"

# Dua folder gambar. Keduanya dipakai; urutan tidak berpengaruh.
IMAGE_DIRS = [
    DATA_RAW / "HAM10000_images_part_1",
    DATA_RAW / "HAM10000_images_part_2",
]

# ---------------------------------------------------------------------------
# 4. KELAS DIAGNOSIS (dx) DAN ARTINYA
# ---------------------------------------------------------------------------
# Kode dx di dataset HAM10000 memakai singkatan dermatologi standar. Kita
# expand ke nama panjang (Inggris) dan nama Indonesia supaya mahasiswa bisa
# menjelaskan tiap kelas saat presentasi.
DX_FULL_NAME = {
    "akiec": "Atypical Intraepithelial Keratosis",
    "bcc":   "Basal Cell Carcinoma",
    "bkl":   "Benign Keratosis-like Lesions",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma",
    "nv":    "Melanocytic Nevi (moles)",
    "vasc":  "Vascular Lesions",
}

# Terjemahan ke Bahasa Indonesia untuk ditampilkan di aplikasi.
DX_INDO = {
    "akiec": "Keratosis Intraepitel Atipis",
    "bcc":   "Karsinoma Sel Basal",
    "bkl":   "Lesi Keratosis Jinak",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma",
    "nv":    "Mola (Nevus Melanositik)",
    "vasc":  "Lesi Vaskular",
}

# Urutan kelas yang dipakai konsisten di seluruh proyek (model, label_map,
# confusion matrix, aplikasi). Angka ini adalah indeks label (0..6).
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# ---------------------------------------------------------------------------
# 5. PENGELOMPOKAN BINER: GANAS vs TIDAK GANAS
# ---------------------------------------------------------------------------
# Kenapa perlu pengelompokan biner?
#   Aplikasi dipakai sebagai alat SKRINING. Informasi yang paling berguna
#   bukan "lesi ini kelas apa persisnya", tapi "apakah ini perlu segera
#   diperiksa dokter atau boleh ditunda".
#
# ganas  = mel, bcc, akiec
#   Ketiganya punya potensi metastasis bila terlambat ditangani, sehingga
#   nilai dari deteksi dini sangat tinggi.
# tidak ganas = nv, bkl, df, vasc
#   Umumnya jinak dan tidak berubah menjadi ganas tanpa faktor lain.
#
# CATATAN PENTING: "tidak ganas" BUKAN berarti pasti aman. Istilah yang benar
# dipakai di aplikasi adalah "tidak menunjukkan ciri ganas", bukan "aman".
MALIGNANT_CLASSES = ["mel", "bcc", "akiec"]
BENIGN_CLASSES = ["nv", "bkl", "df", "vasc"]

# ---------------------------------------------------------------------------
# 6. KONFIGURASI PREPROCESSING (WAJIB IDENTIK untuk train dan inference)
# ---------------------------------------------------------------------------
# Ukuran gambar untuk model KLASIK (HOG). Dipilih 128x128 karena:
#   - HOG klasik dirancang untuk ukuran moderate (sekitar 100-200 piksel),
#     bukan 224
#   - 128x128 = 16.384 piksel per gambar; 10.015 gambar masih muat di RAM
#     (10.015 x 16.384 x 3 byte RGB = sekitar 492 MB, aman di laptop)
#   - Ukuran lebih kecil membuat ekstraksi HOG jauh lebih cepat
IMG_SIZE_CLASSIC = 128

# Ukuran gambar untuk MobileNetV2. WAJIB 224x224 karena pretrained weight
# ImageNet dilatih pada 224x224. Mengubahnya berarti mengubah makna feature.
IMG_SIZE_MOBILENET = 224

# Ukuran gambar untuk CNN sederhana. Kita pakai 128x128 supaya:
#   - waktu training di CPU masih masuk akal
#   - lesi masih terlihat jelas (lesi occupy bagian besar dari frame)
IMG_SIZE_CNN = 128

# Jumlah bin histogram warna per kanal (R, G, B). 16 bin per kanal memberi
# 16^3 = 4096 kombinasi warna, cukup granular untuk membedakan lesi
# kemerahan dan kecoklatan tanpa terlalu sparse.
HIST_BINS = 16

# Parameter HOG. Dipilih standar (default skimage) yang umum dipakai di
# paper skin lesion: ukuran sel 8x8 piksel, block 2x2 sel.
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)

# ---------------------------------------------------------------------------
# 6b. KONFIGURASI REDUKSI DIMENSI (hasil eksperimen, bukan tebakan)
# ---------------------------------------------------------------------------
# Ukuran sel HOG 16x16 dipilih berdasarkan EKSPERIMEN nyata
# (scripts/diagnostik_hog_cell.py, docs/logs/diagnostik_hog_cell.json):
#
#   pixels_per_cell  8 ->  8.100 fitur  -> F1 macro 0,2163 (SVM proyek)
#   pixels_per_cell 16 ->  1.764 fitur  -> F1 macro 0,2118 (SVM proyek)  <- DIPILIH
#   pixels_per_cell 32 ->    324 fitur  -> F1 macro 0,2175 (SVM proyek)
#
# PENTING: selisih performa antar ukuran sel cuma 0,0057 -- jauh
# lebih kecil daripada selisih antar model. Jadi alasan memilih 16x16
# BUKAN skornya paling tinggi, tapi karena HEMAT:
#   1. 1.764 dimensi adalah 4,6x lebih kecil dari 8.100 dimensi, jadi
#      ekstraksi fitur, PCA, dan training jauh lebih cepat.
#   2. Sel 8x8 menghasilkan banyak dimensi yang saling berkorelasi
#      (noise), sel 32x32 menghilangkan bentuk dan tepi lesi yang
#      justru penting membedakan melanoma dari mola.
#   3. Performanya setara, jadi kita ambil yang paling murah.
#
# Jumlah komponen PCA untuk meringkas bagian HOG: 30. Dari 1.764 fitur HOG
# menjadi 30 komponen utama, sehingga bagian HOG tidak lagi "membanjiri"
# informasi warna yang terbukti paling diskriminatif.
PCA_N_COMPONENTS = 30

# Bobot pengali fitur histogram warna saat digabung dengan hasil PCA.
# Histogram sudah ternormalisasi (0..1) sedangkan komponen PCA punya skala
# jauh lebih besar, jadi tanpa pengali ini informasi warna hilang di antara
# informasi bentuk. Nilai 6 dipilih dari hasil pengukuran.
HIST_WEIGHT = 6.0

# ---------------------------------------------------------------------------
# 7. PEMBAGIAN DATA
# ---------------------------------------------------------------------------
# Dibagi per lesion_id (bukan per gambar) karena beberapa lesi difoto lebih
# dari sekali. Kalau di-split per gambar, foto lesi yang sama bisa muncul di
# train DAN test, sehingga model menghafal lesi itu, bukan belajar pola
# umum. Ini bernama data leakage dan membuat metrik jadi terlalu optimistis.
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# ---------------------------------------------------------------------------
# 8. NAMA FILE HASIL ANTARA
# ---------------------------------------------------------------------------
TRAIN_CSV = DATA_PROCESSED / "train.csv"
VAL_CSV = DATA_PROCESSED / "val.csv"
TEST_CSV = DATA_PROCESSED / "test.csv"

SPLIT_SUMMARY_JSON = DATA_PROCESSED / "split_summary.json"
HOG_FEATURES_NPY = DATA_PROCESSED / "hog_features.npy"
HOG_LABELS_NPY = DATA_PROCESSED / "hog_labels.npy"
HOG_IDS_CSV = DATA_PROCESSED / "hog_ids.csv"

# ---------------------------------------------------------------------------
# 9. KONFIGURASI PREPROCESSING UNTUK APLIKASI (disimpan di models/)
# ---------------------------------------------------------------------------
# Aplikasi web membaca file JSON ini, bukan menulis angka sendiri.
# Kalau preprocessing berubah, cukup simpan ulang file ini.
PREPROCESS_CONFIG_NAME = "preprocess_config.json"
LABEL_MAP_NAME = "label_map.json"

# ---------------------------------------------------------------------------
# 10. PARAMETER MODEL
# ---------------------------------------------------------------------------
# Ukuran batch. 32 adalah titik tengah: cukup besar agar CPU (dan GPU)
# efisien, cukup kecil agar RAM tidak meledak saat augmentasi.
BATCH_SIZE = 32

# Jumlah epoch maksimum. EarlyStopping menghentikan lebih awal kalau validasi
# sudah tidak membaik, jadi ini batas ATAS, bukan target.
MAX_EPOCHS_CNN = 30
MAX_EPOCHS_MOBILENET_STAGE1 = 15   # classifier saja, backbone dibekukan
MAX_EPOCHS_MOBILENET_STAGE2 = 15   # fine-tuning sebagian lapisan atas
EARLY_STOP_PATIENCE = 5

# Threshold default untuk keputusan "ganas". Angka ini akan di-TUNE di Fase 6
# memakai data validasi; nilai di sini hanya titik awal yang wajar.
DEFAULT_MALIGNANT_THRESHOLD = 0.50
