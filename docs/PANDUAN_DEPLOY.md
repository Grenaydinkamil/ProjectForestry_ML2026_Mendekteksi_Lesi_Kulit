# Panduan Deployment Aplikasi Streamlit

Dokumen ini berisi langkah demi langkah untuk men-deploy aplikasi skrining
lesi kulit ke internet, plus penjelasan trade-off di balik setiap keputusan.

---

## 1. PILIHAN PLATFORM

Ada tiga platform yang masuk akal. Perbandingan berdasarkan kondisi nyata
proyek ini (bukan tebakan):

| Platform | Biaya | Batas RAM | Batas ukuran file | Repository publik | Kesulitan |
|---|---|---|---|---|---|
| **Streamlit Community Cloud** | Gratis | 1 GB (free tier) | 200 MB | Tidak perlu | Mudah |
| **Hugging Face Spaces (CPU)** | Gratis | 16 GB (CPU basic) | 50 GB per file | Tidak perlu | Mudah |
| **Render / Railway / VPS** | $5-25/bulan | 512 MB - 8 GB | Tergantung | Tidak perlu | Sedang |

### Rekomendasi: Streamlit Community Cloud (opsi A), dengan HF Spaces (opsi B) sebagai cadangan

**Alasan memilih Streamlit Community Cloud:**
1. **Gratis** dan tidak minta kartu kredit.
2. **Deploy dari GitHub** dalam 5 klik, tanpa upload manual.
3. **Dukungan Streamlit bawaan**, jadi tidak ada konfigurasi reverse proxy.
4. Ter-integrasi dengan GitHub Actions (CI/CD otomatis).

**Keterbatasan yang harus diketahui:**
- **RAM 1 GB.** Model MobileNetV2 + TensorFlow + Streamlit akan memakai
  sebagian besar RAM itu. Ini **risiko nyata** dan harus diuji.
- **Cold start bisa 30-60 detik** karena paket (~400 MB) diunduh tiap kali
  server dinyalakan.
- **Ukuran repo dibatasi 1 GB.** Model `.keras` masuk repo, tapi ukurannya
  kecil (9,3 MB untuk MobileNetV2), jadi aman.
- **Dataset TIDAK boleh di-commit.** Repo kita 2,6 GB dataset, jauh melebihi
  batas. `.gitignore` sudah mengexclude `data/`.

**Kapan sebaiknya pilih Hugging Face Spaces:**
- Kalau RAM 1 GB ternyata tidak cukup (error `OutOfMemory` saat app start).
- Kalau butuh inferensi lebih cepat (HF Spaces CPU basic punya 2 vCPU, dan
 inferensi tidak dibatasi GPU).
- Kalau mau menyimpan model lewat LFS (Git LFS) tanpa batas ukuran.

**Kapan jangan pilih platform berbayar:**
- Untuk tugas kuliah dengan demo singkat, ini buang-buangan.
- Platform gratis sudah cukup, dan keunggulannya ada di "semua orang bisa akses
  URL-nya".

---

## 2. PRA-PERSIAPAN (SEBELUM DEPLOY)

### 2.1 Pastikan semua file yang dibutuhkan ada

```powershell
# Cek model sudah dilatih dan tersimpan
Get-ChildItem models/
# Harus ada:
#   mobilenetv2_best.keras     (atau cnn_best.keras)
#   label_map.json
#   preprocess_config.json
#   threshold_config.json
```

### 2.2 Pastikan aplikasi jalan lokal

```powershell
streamlit run app/app.py
```

Buka `http://localhost:8501`, unggah beberapa gambar uji, dan pastikan:
- [ ] Foto muncul
- [ ] Probabilitas 7 kelas tampil
- [ ] Status risiko muncul dengan warna yang sesuai
- [ ] Tidak ada error di terminal

### 2.3 Pastikan requirement deploy sudah minimal

`app/requirements.txt` memakai `tensorflow-cpu` (bukan `tensorflow`).
Alasannya: package `tensorflow` menarik CUDA toolkit ~3 GB yang tidak akan
dipakai di server, dan bisa membuat deploy gagal karena ukuran.

File itu juga memuat `streamlit-shadcn-ui==1.5.0` dan `plotly==7.1.0`
karena antarmuka memakai komponen kartu/lencana dan grafik probabilitas.
Keduanya ringan (total di bawah 15 MB) dan punya wheel untuk Linux.

### 2.4 Cek ukuran repo

```powershell
git count-objects -vH
# atau, lebih cepat:
python scripts/simulasi_deploy.py
```

Repo ini berisi **147 file, sekitar 97,5 MB**. Batas Streamlit Community
Cloud adalah 1 GB per repo dan 200 MB per berkas, jadi masih jauh di bawah
batas. Yang paling besar adalah `models/cnn_best.keras` (25,3 MB) dan
`models/rf_hog.joblib` (54,1 MB).

Kalau nanti ukuran repo melewati 1 GB, hampir pasti `data/` ikut ter-commit.
Cek dengan:

```powershell
git check-ignore -v data/raw/HAM10000_metadata.csv
```

---

## 3. LANGKAH DEPLOY KE STREAMLIT COMMUNITY CLOUD

### Langkah 1: Push ke GitHub

Repo proyek ini sudah ada di GitHub:

```text
https://github.com/Grenaydinkamil/ProjectForestry_ML2026_Mendekteksi_Lesi_Kulit
```

Kalau kamu memakai repo baru, ganti remote-nya:

```powershell
cd <path-ke-repo>
git remote set-url origin https://github.com/<USERNAME>/<NAMA-REPO>.git
git push -u origin main
```

**Penting:** pastikan `data/` TIDAK ikut ter-push. Cek dulu:
```powershell
git status --short
# Harusnya tidak ada baris untuk data/raw/ atau data/external/
```

### Langkah 2: Buat akun Streamlit

1. Buka <https://share.streamlit.io>
2. Daftar dengan akun GitHub
3. Verifikasi email

### Langkah 3: Deploy

1. Klik **"New app"**
2. Pilih repository `Grenaydinkamil/ProjectForestry_ML2026_Mendekteksi_Lesi_Kulit`
3. Branch: `main`
4. **Main file path**: `app/app.py`
5. Buka **"Advanced settings"** lalu pilih **Python 3.13**
6. Klik **"Deploy"**

### Langkah 4: Dependensi dan versi Python

Streamlit Community Cloud mencari file dependensi **mulai dari folder
entrypoint, lalu root**, dan **hanya membaca file pertama yang ditemukan**.
Karena entrypoint adalah `app/app.py`, platform membaca
`app/requirements.txt`. Repo ini sengaja hanya punya **satu**
`requirements.txt`; file untuk training di root bernama
`requirements-training.txt` supaya tidak mungkin salah pilih.

Verifikasi sendiri:

```powershell
git ls-files | Select-String "requirements.txt$"
# harus menghasilkan satu baris: app/requirements.txt
```

**Versi Python** dipilih lewat dropdown **"Python version"** di modal
**"Advanced settings"**, bukan lewat `runtime.txt` atau `.python-version`.
Pilih **3.13** karena itu versi yang dipakai saat training dan semua paket
sudah di-pin ke versi yang punya wheel Linux untuk 3.13 (dicek oleh
`scripts/cek_wheel_linux.py`).

Tidak perlu `packages.txt` karena aplikasi tidak butuh dependensi apt-get.

### Langkah 5: Verifikasi

1. Tunggu build selesai (2-5 menit)
2. Buka URL yang diberikan
3. **Uji dengan foto nyata** (bukan placeholder)
4. Cek log di **Manage app → Logs** kalau ada error

---

## 4. MENGATUR UKURAN MODEL (BILANGAN PERLU)

Model yang dipakai aplikasi, `models/mobilenetv2_best.keras`, berukuran
**9,3 MB**. Ini **jauh di bawah batas 100 MB per file** di GitHub, jadi
**tidak perlu Git LFS**.

Berkas model lain di repo ini:

| Berkas | Ukuran | Dipakai aplikasi? |
|---|---|---|
| `mobilenetv2_best.keras` | 9,3 MB | **Ya** |
| `cnn_best.keras` | 25,3 MB | Tidak (pembanding) |
| `rf_hog.joblib` | 54,1 MB | Tidak (pembanding) |
| `svm_hog.joblib` | 15,2 MB | Tidak (pembanding) |

Semua tetap di-commit karena laporan membandingkan empat model. Total repo
tetap di bawah 100 MB.

Kalau nanti model jadi jauh lebih besar (misal ensemble beberapa model),
ada tiga opsi:

| Opsi | Ukuran | Trade-off |
|---|---|---|
| **.keras apa adanya** | ~9 MB | Paling mudah, disarankan |
| **Git LFS** | ~9 MB | Butuh akun LFS (gratis 1 GB), repo jadi lebih rumit |
| **TFLite** | ~3 MB (kuantisasi) | Kecepatan inferensi naik 2-3x, tapi **akurasi bisa turun 1-2%** dan pre-processing harus pakai operator TFLite |

**Rekomendasi: jangan pakai TFLite** untuk project ini. Alasannya:
- Kecepatan inferensi di Streamlit Community Cloud sudah cukup (prediksi
  1 gambar di bawah 1 detik).
- Kuantisasi bisa mengubah probabilitas, yang berimplikasi ke threshold
  yang sudah kita tuning dari validasi. **Threshold jadi tidak konsisten.**
- TFLite hanya perlu kalau targetnya aplikasi mobile, bukan web.

---

## 5. TROUBLESHOOTING

### Error: `ModuleNotFoundError: No module named 'src'`

**Penyebab:** Streamlit menjalankan `app/app.py`, dan `src/` ada satu level
di atas.

**Solusi:** sudah ditangani di `app/app.py` lewat:
```python
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
```

Kalau masih error, pastikan `src/` ikut ter-push ke GitHub.

### Error: `FileNotFoundError: model tidak ditemukan`

**Penyebab:** file model tidak ikut ter-commit.

**Solusi:**
```powershell
git add models/
git commit -m "Tambahkan model terlatih"
git push
```

### Error: `OutOfMemory` saat app start

**Penyebab:** RAM 1 GB di free tier tidak cukup untuk TensorFlow + model.

**Solusi:**
1. Pindah ke Hugging Face Spaces (RAM 16 GB)
2. Atau pakai model klasik (SVM/RF) yang hanya butuh sklearn

### Error: `No matching distribution found for tensorflow-cpu`

**Penyebab:** versi Python di platform tidak cocok dengan versi paket yang
di-pin.

**Solusi:** buka **"Advanced settings"** pada halaman deploy, pilih
**Python 3.13**, lalu deploy ulang. `runtime.txt` dan `.python-version`
tidak dibaca untuk pemilihan versi Python di Streamlit Community Cloud.

### App lambat banget (cold start)

**Penyebab:** server baru saja dinyalakan dan mengunduh paket ~400 MB.

**Solusi:** normal, akan cepat setelah build pertama selesai. Untuk demo,
buka app **sebelum** presentasi dimulai.

---

## 6. RENCANA CADANGAN (JIKA DEPLOY GAGAL)

**Selalu siapkan plan B untuk demo.** Aplikasi harus bisa dijalankan lokal
dengan satu perintah:

```powershell
streamlit run app/app.py
```

Kalau laptop presentasi punya Python 3.13 dan TensorFlow terpasang, ini
selalu berhasil. Pastikan dengan mencoba **sebelum hari presentasi**.

---

## 7. YANG TIDAK BOLEH DI-COMMIT

| File | Alasan | Sudah di-ignore? |
|---|---|---|
| `data/raw/` | Lisensi non-komersial + 2,6 GB | Ya |
| `data/external/` | Lisensi + 3,4 GB | Ya |
| `data/processed/*.npy` | 311 MB, bisa di-generate ulang | Ya |
| `.venv/` | Berisi ribuan file, tidak portable | Ya |
| `models/*.npy`, `*.pkl` | besar, bisa di-generate ulang | Ya |
| `models/*.keras` | **BOLEH di-commit** (9-26 MB per berkas) | Tidak (sengaja) |
| `models/*.joblib` | **BOLEH di-commit** (SVM 15,2 MB; RF 54,1 MB) | Tidak (sengaja) |

> **Catatan lisensi:** dataset tetap milik pemegangnya dan lisensinya
> (CC BY-NC untuk HAM10000, CC BY untuk PAD-UFES-20) tidak berubah karena
> kita tidak meng-commit-nya. Kita tetap **wajib menyertakan sitasinya**
> di README, di aplikasi, dan di laporan.
