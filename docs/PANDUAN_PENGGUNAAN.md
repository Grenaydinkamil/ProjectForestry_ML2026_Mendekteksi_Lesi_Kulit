# PANDUAN PENGGUNAAN APLIKASI

## Skrining Awal Risiko Lesi Kulit

Aplikasi ini dibangun untuk keperluan **pembelajaran dan demonstrasi**.
Ia **bukan alat diagnosis medis** dan belum divalidasi secara klinis.

---

## 1. Siapa yang boleh memakai?

| Pengguna | Tujuan yang tepat |
|---|---|
| Mahasiswa / dosen | Mendemonstrasikan pipeline machine learning dari data hingga aplikasi |
| Peneliti pemula | Memahami cara model klasik dan deep learning dibandingkan |
| Pengguna umum | **Hanya** sebagai edukasi, bukan pengganti dokter |

Yang **tidak** boleh dilakukan:
- Menjadikan hasil aplikasi sebagai dasar keputusan pengobatan.
- Mengabaikan keluhan atau perubahan lesi hanya karena aplikasi berkata "aman".
- Memakai aplikasi tanpa pengawasan tenaga kesehatan untuk kasus nyata.

---

## 2. Persyaratan Sistem

| Kebutuhan | Keterangan |
|---|---|
| Python | 3.10–3.13 (proyek dikembangkan dengan 3.13.9) |
| Sistem operasi | Windows, Linux, atau macOS |
| Browser | Chrome, Edge, Firefox, atau Safari versi terkini |
| RAM | Minimal 4 GB (disarankan 8 GB untuk training ulang) |
| GPU | **Tidak wajib** untuk menjalankan aplikasi; training memakai CPU atau WSL2 GPU |
| Disk | ±300 MB untuk kode + model (dataset tidak ikut repo, ±6 GB bila diunduh) |

---

## 3. Instalasi

### 3.1 Unduh atau clone proyek

```bash
git clone <URL-repository> ML2026_Kelas_3_Kelompok_1_LesiKulit
cd ML2026_Kelas_3_Kelompok_1_LesiKulit
```

### 3.2 Buat virtual environment

**Windows (PowerShell):**
```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-training.txt
```

**Linux / macOS / WSL:**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-training.txt
```

### 3.3 Verifikasi instalasi

```bash
python scripts/check_environment.py
```

Skrip ini memeriksa versi Python, pustaka yang dibutuhkan, keberadaan
model, dan struktur folder. Jika ada yang kurang, ia akan memberitahu
secara spesifik.

### 3.4 Dataset (hanya jika ingin training ulang)

Dataset **tidak disertakan** di repo karena ukurannya besar dan lisensinya.
Untuk menjalankan aplikasi, dataset **tidak diperlukan**--model sudah
tersedia di folder `models/`.

Bila ingin training ulang:

1. Unduh HAM10000 dari <https://doi.org/10.7910/DVN/DBX6T5> dan terima
   lisensinya (perlu akun Kaggle).
2. Unduh PAD-UFES-20 dari <https://doi.org/10.17632/zr7vgbcyr2>.
3. Letakkan di `data/raw/` dan `data/external/` sesuai struktur yang
   dijelaskan di `README.md`.
4. Jalankan notebook secara berurutan dari 01 sampai 08.

---

## 4. Menjalankan Aplikasi

Dari folder utama proyek:

```bash
streamlit run app/app.py
```

Powershell akan menampilkan alamat, biasanya:

```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

Buka alamat tersebut di browser. Halaman pertama memuat model sekali
(beberapa detik); setelah itu interaksi menjadi cepat.

### Cara menghentikan

Tekan `Ctrl + C` di terminal, atau tutup jendela terminal.

---

## 5. Tampilan dan Fungsi

### 5.1 Header dan disclaimer

Di bagian atas halaman:

- Judul: **Skrining Risiko Lesi Kulit dari Foto**
- Penjelasan singkat empat model dan jumlah data
- Kotak merah berisi disclaimer: aplikasi adalah alat bantu belajar,
  bukan diagnosis; hasil "tidak ganas" **tidak** berarti lesi aman.

Disclaimer ini sengaja tidak bisa disembunyikan.

### 5.2 Sidebar kiri

| Elemen | Fungsi |
|---|---|
| Panduan memotret | Lima tips agar foto lebih berguna: dekat & fokus, cahaya merata, lesi di tengah, tanpa filter, kulit terlihat jelas |
| Privasi | Menjelaskan bahwa foto diproses di memori dan tidak disimpan atau dikirim ke pihak ketiga |

### 5.3 Panel "Unggah foto"

Ada **dua cara** memberi gambar:

**Cara A — Unggah sendiri**
1. Klik area unggahan atau tarik file ke area tersebut.
2. Pilih file JPG, JPEG, atau PNG, maksimal 10 MB.
3. Foto akan langsung tampil di panel dengan nama dan ukurannya.

**Cara B — Pakai contoh yang sudah disediakan**
1. Di kotak "Atau pakai contoh", pilih salah satu:
   - **Melanoma (dermoskop)** — lesi ganas, foto dermoskop
   - **Karsinoma sel basal (dermoskop)** — lesi ganas
   - **Nevus jinak (dermoskop)** — lesi jinak
   - **Melanoma (foto smartphone)** — contoh domain shift
2. Centang **"Pakai contoh ini"**.
3. Foto langsung tampil tanpa perlu mencari file.

Cara B paling cocok untuk demo dan untuk mencoba aplikasi pertama kali.

### 5.4 Panel "Hasil analisis"

Setelah gambar dipilih, panel kanan menampilkan:

**a. Kotak status risiko** (tiga kemungkinan)

| Warna | Label | Arti |
|---|---|---|
| Hijau | **Tidak ditemukan ciri ganas** | Risiko ganas di bawah ambang dan model cukup yakin |
| Merah | **Ciri ganas terdeteksi** | Risiko ganas di atas ambang--segera periksakan ke dokter |
| Kuning | **Hasil belum meyakinkan** | Risiko terlalu dekat dengan ambang atau model ragu--tetap periksakan ke dokter |

**b. Diagnosis paling mungkin**
Nama kelas dalam Bahasa Indonesia, nama lengkap, dan tingkat keyakinan
model dalam persen.

**c. Probabilitas per kelas**
Tujuh bar mendatar, diurutkan dari yang tertinggi. **Bar merah** menandai
kelas ganas (melanoma, karsinoma sel basal, keratosis atipis); bar biru
menandai kelas jinak. Angka di sebelah kanan adalah persentase.

**d. Informasi tambahan**
Penjelasan warna bar, nilai ambang, dan asal ambang tersebut (data
validasi, bukan data uji).

### 5.5 Panel bawah "Model yang dipakai"

Menampilkan nama model (MobileNetV2), jenis, ukuran input (224×224),
ambang risiko ganas, dan sumber dataset lengkap dengan lisensinya.

---

## 6. Contoh Alur Penggunaan (Demo 3 Menit)

1. Buka aplikasi, tunggu model selesai dimuat.
2. Baca disclaimer di bagian atas.
3. Pilih contoh **Nevus jinak** → centang "Pakai contoh ini".
   Amati kotak hijau dan bar biru yang dominan.
4. Ganti ke contoh **Melanoma (dermoskop)**.
   Amati kotak merah dan bar merah yang naik.
5. Ganti ke contoh **Melanoma (foto smartphone)**.
   Amati bahwa model bisa salah dengan keyakinan tinggi--ini adalah
   contoh nyata domain shift yang dibahas di laporan.
6. Tutup dengan membaca disclaimer sekali lagi.

---

## 7. Memahami Angka dengan Benar

| Angka | Arti | Cara membaca |
|---|---|---|
| Risiko ganas | Jumlah probabilitas ketiga kelas ganas | Dibandingkan dengan ambang (lihat panel bawah) |
| Ambang | Batas keputusan dari data validasi | Bukan angka universal; berbeda untuk tiap model |
| Probabilitas kelas | Keyakinan model untuk tiap diagnosis | Tujuh angka berjumlah 100% |
| "Belum meyakinkan" | Peringatan, bukan hasil | Tetap harus periksa ke dokter |

**Yang sering disalahpahami:**
- Accuracy tinggi tidak berarti model bagus untuk skrining. Random Forest
  punya accuracy 0,7132 tetapi hanya mendeteksi 26% lesi ganas.
- Recall tinggi bisa berarti model menandai semuanya sebagai ganas.
  Selalu lihat specificity bersama recall.
- Aplikasi tidak melihat "jenis kulit" secara eksplisit; seluruh data
  latih berasal dari Fitzpatrick I-II.

---

## 8. Pemecahan Masalah

| Gejala | Penyebab | Solusi |
|---|---|---|
| Halaman kosong / loading lama | Model sedang dimuat | Tunggu 5–15 detik; ini normal di awal |
| "Model belum siap dipakai..." | File model/config hilang | Pastikan `models/` berisi `mobilenetv2_best.keras`, `label_map.json`, `preprocess_config.json`, `threshold_config.json` |
| "File bukan gambar..." | File rusak atau ekstensi salah | Pakai JPG/PNG asli, bukan file teks yang diganti ekstensinya |
| "Ukuran file melebihi batas 10 MB" | Foto terlalu besar | Kecilkan resolusi atau pakai contoh gambar |
| Prediksi aneh pada foto smartphone | Domain shift | Pakai foto dermoskop dengan lesi di tengah dan cahaya merata |
| `ModuleNotFoundError` | Virtual environment belum aktif | Aktifkan `.venv` lalu `pip install -r requirements-training.txt` |
| Port 8501 sudah dipakai | Ada Streamlit lain berjalan | `streamlit run app/app.py --server.port 8502` |

---

## 9. Deploy ke Internet (Opsional)

Panduan lengkap ada di `docs/PANDUAN_DEPLOY.md`. Ringkasannya:

1. Unggah kode ke GitHub (**jangan** unggah `data/`).
2. Buka Streamlit Community Cloud, hubungkan repository.
3. Pilih `app/app.py` sebagai entry point, branch `main`.
4. Di **Advanced settings**, pilih **Python 3.13**. Platform menemukan
   `app/requirements.txt` sendiri karena letaknya sefolder dengan entry
   point; tidak perlu menambah file requirements lain.
5. Perhatikan batas RAM 1 GB pada tier gratis; alternatifnya Hugging Face
   Spaces dengan RAM lebih besar.

---

## 10. Verifikasi dan Reproduksi

```bash
# Cek seluruh proyek (13 pemeriksaan + bangun ulang PPT & laporan)
python scripts/verify_final.py

# Uji isi aplikasi saja
python scripts/test_app_content.py

# Uji perilaku status "tidak yakin"
python scripts/cek_uncertain.py

# Bangun ulang laporan dan PPT dari log
python scripts/build_rekap.py
node scripts/build_ppt.js
```

Semua angka di laporan dan slide diambil dari `docs/logs/*.json`. Bila
notebook dijalankan ulang, jalankan perintah di atas untuk menyelaraskan
dokumen.

---

## 11. Batasan Etis dan Medis

- Model **belum divalidasi klinis**.
- Dataset HAM10000 berlisensi **CC BY-NC 4.0** (hanya non-komersial).
  PAD-UFES-20 berlisensi **CC BY 4.0**.
- Foto pengguna diproses di memori, tidak disimpan, tidak dikirim ke
  pihak ketiga.
- Hasil negatif **tidak** menyingkirkan kemungkinan penyakit. Bila lesi
  berubah bentuk, warna, atau ukuran, berdarah, atau terasa tidak nyaman,
  **tetap periksakan ke dokter kulit**.

---

## 12. Ringkasan Perintah

```bash
# Instalasi
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements-training.txt

# Cek environment
python scripts/check_environment.py

# Jalankan aplikasi
streamlit run app/app.py

# Verifikasi akhir
python scripts/verify_final.py
```

Dokumen terkait: `README.md`, `docs/LAPORAN_PROJECT_FINAL.md`,
`docs/REKAP_LAPORAN.md`, `docs/CATATAN_TEMUAN_DIAGNOSTIK.md`,
`docs/CHECKLIST_VIDEO_DEMO.md`.
