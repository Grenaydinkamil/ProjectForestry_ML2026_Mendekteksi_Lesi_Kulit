# Klasifikasi Lesi Kulit Berbasis Citra

**Skrining Awal Risiko Keganasan dengan HOG-SVM, HOG-Random Forest, CNN, dan MobileNetV2, Termasuk Evaluasi Domain Shift pada Foto Smartphone**

Machine Learning Project - Mata Kuliah Pembelajaran Mesin (ML2026)

---

## RINGKASAN PROYEK

Proyek ini membangun **empat model machine learning** untuk mengklasifikasikan
lesi kulit ke dalam 7 kelas diagnosis dari foto dermatoskop, lalu mengevaluasi
seberapa baik model tersebut bekerja pada **foto smartphone** (dataset
PAD-UFES-20) sebagai uji eksternal.

| Komponen | Detail |
|---|---|
| **Masalah** | Klasifikasi citra multi-kelas (7 kelas) + ringkasan biner ganas/tidak ganas |
| **Dataset utama** | HAM10000 (10.015 foto dermatoskop, 7.470 lesi) |
| **Dataset uji eksternal** | PAD-UFES-20 (2.298 foto smartphone) |
| **Model** | HOG-SVM, HOG-Random Forest, CNN dari nol, MobileNetV2 (transfer learning) |
| **Metrik utama** | F1 macro, recall per kelas, confusion matrix, **recall kelas ganas** |
| **Aplikasi** | Streamlit (web app) |

> **Peringatan.** Proyek ini adalah **alat bantu belajar dan skrining edukatif**,
> **BUKAN alat diagnosis medis**. Prediksi "tidak ganas" tidak menjamin lesi aman.
> Lihat bagian [Batasan Etis](#batasan-etis-dan-medis).

---

## DOKUMEN UTAMA

| Dokumen | Isi | Cocok untuk |
|---|---|---|
| `docs/LAPORAN_PROJECT_FINAL.md` | **Laporan formal proyek**: bab 1–6, abstrak, tinjauan pustaka, metodologi, hasil, pembahasan, kesimpulan, daftar pustaka, dan lampiran status proyek | Dosen, laporan tertulis, tanya jawab mendalam |
| `docs/PANDUAN_PENGGUNAAN.md` | **Cara pakai produk**: instalasi, menjalankan aplikasi, membaca hasil, contoh alur demo, pemecahan masalah, deploy | Pengguna aplikasi, pilot demo |
| `docs/AUDIT_STREAMLIT_ANTISLOP.md` | Hasil audit skill Streamlit resmi + antislop, delivery gate | Reviewer kode, tanya jawab kualitas UI |
| `docs/REKAP_LAPORAN.md` | Ringkasan angka hasil (dibangun otomatis dari log) | Cek cepat angka |
| `docs/CATATAN_TEMUAN_DIAGNOSTIK.md` | 13 bug dan eksperimen yang membentuk keputusan desain | Pembahasan teknis |
| `docs/PERSIAPAN_TANYA_JAWAB.md` | 40+ pertanyaan beserta bukti file | Latihan presentasi |
| `docs/CHECKLIST_VIDEO_DEMO.md` | Checklist rekaman video maksimal 7 menit | Pembuatan video demo |
| `docs/PPT_Laporan_Project.pptx` | PPT versi 1 — 14 slide, detail lengkap | Presentasi panjang |
| `docs/PPT_Laporan_Project_v2.pptx` | **PPT versi 2 — 12 slide**, ringkas, palet klinis baru | Presentasi 5–7 menit + video |

Kedua file PPT dibangun dari `docs/logs/*.json`, sehingga tidak ada angka
yang diketik manual. Jalankan `node scripts/build_ppt.js` dan
`node scripts/build_ppt_v2.js` untuk membangun ulang.

---

## STRUKTUR REPO

```
ML2026_Kelas_3_Kelompok_1_LesiKulit/
│
├── data/
│   ├── raw/              # Dataset mentah (tidak di-commit, ~2,6 GB)
│   ├── processed/        # train/val/test.csv, fitur .npy (tidak di-commit)
│   └── external/         # PAD-UFES-20 (tidak di-commit)
│
├── notebooks/
│   ├── 01_eda.ipynb              # Eksplorasi data
│   ├── 02_split.ipynb            # GroupShuffleSplit + anti-leakage assert
│   ├── 03_fitur_klasik.ipynb     # Ekstraksi HOG + histogram warna
│   ├── 04_svm_rf.ipynb           # Training SVM & Random Forest
│   ├── 05_cnn.ipynb              # CNN dari nol
│   ├── 06_mobilenetv2.ipynb      # Transfer learning MobileNetV2
│   ├── 07_evaluasi_perbandingan.ipynb
│   └── 08_uji_eksternal_pad_ufes.ipynb
│
├── src/                       # Modul reusable (dipakai notebook DAN aplikasi)
│   ├── config.py              # Konfigurasi terpusat
│   ├── data_loader.py         # Pemuatan + verifikasi dataset
│   ├── features.py            # Ekstraksi HOG + histogram warna
│   ├── preprocessing.py       # Preprocessing citra + augmentasi
│   ├── modeling.py            # Builder model + training loop
│   └── evaluation.py          # Metrik + visualisasi
│
├── models/                    # Model terlatih (.keras) + label_map.json
│                              # + preprocess_config.json + threshold_config.json
│
├── app/
│   ├── app.py                 # Aplikasi Streamlit
│   ├── contoh/                # Contoh gambar untuk demo (4 file)
│   └── requirements.txt       # Requirement khusus deploy (tensorflow-cpu)
│
├── scripts/
│   ├── check_environment.py       # Cek kesiapan environment
│   ├── download_data.py           # Ekstrak & verifikasi dataset
│   ├── make_notebooks.py          # Generator notebook dari sumber .py
│   ├── setup_wsl_gpu.sh           # Setup WSL2 + TensorFlow GPU
│   ├── run_notebook_wsl.sh        # Runner notebook di WSL (GPU)
│   │
│   │   # ---- Pemeriksaan otomatis (wajib lulus sebelum submit) ----
│   ├── check_api_calls.py         # Cek argumen fungsi yang tidak ada di API
│   ├── check_undefined_names.py   # Cek variabel dipakai sebelum didefinisikan
│   ├── check_indonesian_text.py   # Cek kalimat campuran Inggris / teks rusak
│   ├── sanity_check_text.py       # Cek karakter CJK + BOM
│   │
│   │   # ---- Eksperimen diagnostik (dasar angka di laporan) ----
│   ├── diagnostik_fitur.py        # Perbandingan desain fitur HOG + histogram
│   ├── diagnostik_hog_cell.py     # Perbandingan ukuran sel HOG
│   ├── cek_bobot_histogram.py     # Sweep bobot histogram untuk SVM & RF
│   │
│   │   # ---- Utilitas ----
│   ├── siapkan_contoh_gambar.py   # Siapkan contoh gambar untuk demo
│   ├── build_ppt.js               # Generator PPT (angka dari docs/logs)
│   ├── literature.py              # Daftar literatur terverifikasi
│   ├── verify_literature.py       # Verifikasi sitasi ke PubMed
│   ├── test_app_content.py        # Uji isi aplikasi (AppTest)
│   └── notebook_src/              # Sumber notebook (bentuk .py)
│
├── docs/
│   ├── assets/                # Gambar untuk PPT dan laporan
│   ├── logs/                  # Metrik JSON hasil evaluasi
│   ├── CATATAN_TEMUAN_DIAGNOSTIK.md   # Temuan eksperimen + bug yang ditemukan
│   ├── PERSIAPAN_TANYA_JAWAB.md       # 40+ soal potensi
│   ├── PANDUAN_DEPLOY.md               # Panduan deploy aplikasi
│   ├── CHECKLIST_VIDEO_DEMO.md        # Checklist rekaman video demo
│   └── REKAP_LAPORAN.md               # Rekap angka hasil
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## CARA MENJALANKAN DARI NOL

### Langkah 1: Siapkan Python 3.13 dan Virtual Environment

TensorFlow 2.21 **tidak mendukung Python 3.14** dan **tidak mendukung GPU
di Windows native**. Jadi kita butuh Python 3.13.

```powershell
# Cek versi python yang tersedia
py -0p

# Buat virtual environment
py -3.13 -m venv .venv

# Aktifkan (PowerShell)
.\.venv\Scripts\Activate.ps1

# Pasang semua dependency
pip install --upgrade pip
pip install -r requirements.txt
```

### Langkah 2: Unduh Dataset (MANUAL, oleh mahasiswa)

**Dataset TIDAK di-commit ke repository** karena ukurannya besar dan lisensinya
non-komersial. Aggara yang mengunduh:

1. **HAM10000** (dataset utama)
   - Kaggle: <https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000>
   - Alternatif (asli): Harvard Dataverse, cari "HAM10000 Tschandl"
2. **PAD-UFES-20** (uji eksternal, dibutuhkan di Fase 7)
   - Mendeley Data: <https://data.mendeley.com/datasets/36r8h2ej4q/1>

Simpan file `.zip` ke `data/raw/`, lalu ekstrak:

```powershell
python scripts/download_data.py --extract-ham10000
python scripts/download_data.py --verify
```

### Langkah 3: Verifikasi Environment

```powershell
python scripts/check_environment.py
```

Script ini memeriksa versi library, deteksi GPU, kelengkapan dataset, dan
menguji fungsi-fungsi inti di `src/`.

### Langkah 4: Jalankan Notebook Secara Berurutan

```powershell
jupyter lab
```

Buka dan jalankan `notebooks/01_eda.ipynb` sampai `08_uji_eksternal_pad_ufes.ipynb`
secara berurutan. **Jangan loncat**, karena notebook berikutnya bergantung
pada file yang dihasilkan notebook sebelumnya.

### Langkah 5: Jalankan Aplikasi

```powershell
streamlit run app/app.py
```

---

## STRATEGI GPU (PENTING)

Masalah klasik: **TensorFlow >= 2.11 tidak bisa memakai GPU di Windows native.**
Ini dikonfirmasi oleh dokumentasi resmi TensorFlow
(<https://www.tensorflow.org/install/pip>, bagian "Windows Native"):

> Caution: TensorFlow 2.10 was the last TensorFlow release that supported GPU on
> native-Windows. Starting with TensorFlow 2.11, you will need to install
> TensorFlow in WSL2, or install tensorflow or tensorflow-cpu and, optionally,
> try the TensorFlow-DirectML-Plugin.

Pilihan yang tersedia dan yang kita pakai:

| Opsi | Cara Kerja | Kelebihan | Kekurangan |
|---|---|---|---|
| **A. CPU di Windows** | `pip install tensorflow` | Sederhana, setup satu tempat | Lambat untuk MobileNetV2 |
| **B. WSL2 + GPU** (dipakai) | `pip install "tensorflow[and-cuda]"` | GPU RTX 5070 aktif, training cepat | Perlu setup sekali, path antar OS |
| **C. Google Colab** | Upload notebook ke Colab | Gratis GPU T4, tanpa setup | Butuh akun, upload data lama, tidak cocok untuk demo lokal |
| **D. TensorFlow-DirectML** | Plugin Microsoft | Pakai GPU di Windows native | Plugin tidak resmi, dukungan terbatas, sering bermasalah |

**Keputusan: Opsi B (WSL2 + GPU) untuk training, Opsi A (CPU) untuk inference
aplikasi.**

Setup WSL2 (sekali saja, sudah dilakukan di mesin development):

```powershell
# 1. Aktifkan WSL2 (perlu admin, bisa lewat Windows Features atau PowerShell)
wsl --install -d Ubuntu-24.04

# 2. Pasang Python 3.12 + TensorFlow dengan CUDA di dalam WSL
wsl -d Ubuntu-24.04 -u root -- bash scripts/setup_wsl_gpu.sh

# 3. Verifikasi GPU terdeteksi
wsl -d Ubuntu-24.04 -u root -- bash scripts/run_wsl_gpu.sh \
  /opt/venvs/tf312/bin/python -c \
  "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Menjalankan notebook training di WSL (dengan GPU)

```powershell
wsl -d Ubuntu-26.04 -- bash scripts/run_notebook_wsl.sh 05_cnn
wsl -d Ubuntu-26.04 -- bash scripts/run_notebook_wsl.sh 06_mobilenetv2
```

### Tips kecepatan: data di disk NATIVE WSL

Pembacaan file lewat `/mnt/m/` (drive Windows yang di-mount) punya latensi
tinggi dan membuat `tf.data` I/O-bound. Pengukuran di mesin ini:

| Lokasi data | Waktu per epoch (218 step) |
|---|---|
| `/mnt/m/...` (drive Windows) | ~1.626 detik/step |
| `/mnt/f/...` (disk native WSL) | **0,08 detik/step** |

Salin dataset ke disk WSL sekali saja:
```bash
mkdir -p /mnt/f/ml_project_data
cp -r "/mnt/m/.../data/raw" /mnt/f/ml_project_data/raw
```

> **Catatan menjalankan RTX 50 series (Blackwell, sm_120):** TensorFlow 2.21
> dibangun dengan CUDA 12.5 yang tidak punya binary kernel untuk sm_120, jadi
> kernel di-JIT dari PTX. **Run pertama** lambat (bisa 10-30 menit), setelah
> itu di-cache di `~/.nv/ComputeCache` dan kecepatannya normal (30 epoch CNN
> = 9 menit). Ini normal, bukan error.

> **Menjalankan di Windows (CPU) juga bisa** dan hasilnya identik secara
> metodologis (seed sama, data sama), hanya lebih lambat. Jadi WSL2 adalah
> opsional yang nyaman, bukan kewajiban.

---

## ATURAN METODOLOGI YANG DIPEGANG PROYEK INI

Bagian ini penting untuk tanya jawab. Setiap aturan di sini bisa ditunjukkan
ke kodenya.

| # | Aturan | Kenapa | Di mana di kode |
|---|---|---|---|
| 1 | Split data **per `lesion_id`** | 2.545 foto adalah lesi yang sama; split per foto = data leakage | `02_split.ipynb` |
| 2 | **Assert** tidak ada lesi/image yang overlap | Membuktikan klaim, bukan sekadar menyatakannya | `02_split.ipynb` |
| 3 | Scaler di-fit **hanya di train** | Fitting di data lain = "bocor" statistik | `src/classic_models.py` |
| 4 | Augmentasi **hanya di train** | Augmentasi pada val/test merusak pengukuran | `src/preprocessing.py` |
| 5 | Test set dipakai **sekali di akhir** | Test bukan alat tuning | `07_evaluasi_perbandingan.ipynb` |
| 6 | Normalisasi **tidak dua kali** | MobileNetV2 pakai rentang -1..1, bukan 0..1 | `src/modeling.py` |
| 7 | **Accuracy bukan** dasar pemilihan model | Imbalance 58:1, model malas bisa dapat 67% | `07_evaluasi_perbandingan.ipynb` |
| 8 | Preprocessing app **identik** dengan training | Kalau beda, akurasi di app bisa anjlok | `src/config.py` + `models/preprocess_config.json` |
| 9 | PAD-UFES-20 **tidak** untuk training | Uji eksternal harus benar-benar "asing" | `08_uji_eksternal_pad_ufes.ipynb` |
| 10 | Seed **42** di semua tempat | Supaya hasil bisa direproduksi | `src/config.py` |
| 11 | **Class weight** dipakai di semua model | Tanpa itu model hanya bisa menebak `nv` | `src/modeling.py`, `src/classic_models.py` |
| 12 | **Tidak ada lambda** di dalam pipeline model | Lambda tidak bisa di-pickle, model jadi tak bisa disimpan | `src/classic_models.py` |
| 13 | **Ambang batas per-model** di uji eksternal | Tiap model punya titik operasi sendiri dari validasi; satu ambang untuk semua membuat perbandingan tidak adil | `08_uji_eksternal_pad_ufes.ipynb` |
| 14 | **Hyperparameter diukur, bukan diambil dari nilai bawaan** | `ensemble=True`, jumlah komponen PCA, dan bobot histogram semuanya dipilih dari eksperimen, bukan tebakan | `scripts/cek_kalibrasi_svm.py`, `scripts/diagnostik_fitur.py` |
| 15 | **Empat pemeriksa otomatis** harus lulus sebelum submit | Kesalahan seperti variabel salah nama atau argumen API yang tidak ada hanya terlihat saat dieksekusi | `scripts/check_*.py` |

---

## LISENSI DAN ATRIBUSI DATASET

### HAM10000

- **Sumber:** <https://doi.org/10.7910/DVN/DBX6T5> (Harvard Dataverse)
- **Artikel:** Tschandl, C., et al. (2018). "The HAM10000 dataset: A large collection
  of diverse 7-class dermatoscopic images for algorithms in dermatology."
  *Scientific Data*, 5, 180214. <https://doi.org/10.1038/sdata.2018.214>
- **Lisensi:** **CC BY-NC 4.0** (Kreatif Commons Atribusi-NonKomersial 4.0)
- **Ketentuan:** penggunaan **hanya untuk keperluan akademik/non-komersial**.
  Dilarang penggunaan komersial. Atribusi wajib dicantumkan.
- **Keterbatasan demografis:** seluruh gambar berasal dari populasi pigmentasi
  Fitzpatrick I-II (kulit putih). **Tidak ada representasi kulit gelap.**
  Ini keterbatasan serius untuk konteks Indonesia.

### PAD-UFES-20

- **Sumber:** Mendeley Data, <https://data.mendeley.com/datasets/36r8h2ej4q/1>
- **Artikel:** Pacheco, A. G., et al. (2020). "PAD-UFES-20: A Large-Scale Dataset
  of Pigmented Skin Lesions." *Data in Brief*, 39, 107601.
  <https://doi.org/10.1016/j.dib.2020.107601>
- **Lisensi:** **CC BY-NC 4.0**
- **Kegunaan:** **hanya untuk uji eksternal**, tidak untuk training.

> **Catatan kepatuhan:** repository ini tidak memuat data maupun bobot model
> yang berasal dari dataset berlisensi. Pengguna yang memakai proyek ini untuk
> keperluan akademik **wajib mencantumkan kedua sitasi di atas** di laporan dan
> di aplikasi.

---

## BATASAN ETIS DAN MEDIS

Aplikasi ini adalah **alat bantu belajar**, bukan alat diagnosis.

- Hasil **"tidak ganas" TIDAK berarti lesi aman.** Model bisa salah, terutama
  untuk melanoma yang tampil atypikal.
- **Segera periksakan ke dokter kulit** bila lesi berubah bentuk, warna, atau
  ukuran, berdarah, Persistent_questionable, atau terasañas.
- Model dilatih pada data dermoskopi dengan **populasi terbatas** (kulit putih,
  Dominan Derived Europe), **tanpa validasi klinis**, dan **tidakmencakup** variasi
  kulit gelap Fitzpatrick III-V yang lebih umum di Indonesia.
- Efektivitas alat skrining **tidak boleh** dianggap menggantikan pemeriksaan
  klinis oleh dokter.

---

## PEMBAGIAN TUGAS

| Anggota | Tugas |
|---|---|
| Mahasiswa 1 | Dataset, EDA, preprocessing, split data (Notebook 01-02) |
| Mahasiswa 2 | Feature engineering, model klasik + deep learning, evaluasi (Notebook 03-07) |
| Mahasiswa 3 | Implementasi aplikasi Streamlit, uji eksternal, PPT, video demo (Notebook 08, app/) |

---

## LISENSI KODE

Kode proyek ini released di bawah lisensi MIT. Dataset **tetap** mengikuti
lisensi CC BY-NC masing-masing dan **tidak** ikut di-released dalam lisensi ini.
