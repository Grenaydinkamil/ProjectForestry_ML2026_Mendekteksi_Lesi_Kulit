"""
Siapkan versi terbaru docs/PANDUAN_DEPLOY.md dan
docs/PANDUAN_PENGGUNAAN.md di folder temp.

Perubahan PANDUAN_DEPLOY.md (berdasarkan dokumentasi resmi Streamlit
Community Cloud, docs.streamlit.io/deploy/streamlit-community-cloud):

  1. Lokasi file dependensi yang BENAR:
     Platform mencari di folder entrypoint DULU, lalu root, dan hanya
     membaca file PERTAMA yang ditemukan. Jadi bukan "membaca dari root".
  2. Saran "buat runtime.txt / .python-version" DIHAPUS: versi Python
     dipilih lewat dropdown "Advanced settings" di UI Streamlit, bukan
     lewat berkas di repo.
  3. Ukuran model dikoreksi (9,3 MB, bukan 14 MB).
  4. Tabel "yang tidak boleh di-commit" dikoreksi: .joblib TIDAK ikut
     (tidak dibutuhkan aplikasi, berukuran 15 MB + 54 MB).
  5. Ditambah langkah verifikasi: simulasi deploy dari checkout bersih.

Perubahan PANDUAN_PENGGUNAAN.md: rujukan pip install.
"""

import pathlib

REPO = pathlib.Path(
    r"M:\GrenPlums Roses\Program32bit\Forestrypansy"
    r"\ML2026_Kelas_3_Kelompok_1_LesiKulit"
)
TMP = pathlib.Path(r"C:\Users\GrenWisesLeaf\AppData\Local\Temp\opencode")

# =====================================================================
# PANDUAN_DEPLOY.md
# =====================================================================
p = REPO / "docs" / "PANDUAN_DEPLOY.md"
t = p.read_text(encoding="utf-8")

ganti = []

ganti.append((
    """### 2.3 Pastikan requirement deploy sudah minimal

`app/requirements.txt` memakai `tensorflow-cpu` (bukan `tensorflow`).
Alasannya: package `tensorflow` menarik CUDA toolkit ~3 GB yang tidak akan
dipakai di server, dan bisa membuat deploy gagal karena ukuran.""",
    """### 2.3 Pastikan requirement deploy sudah minimal

`app/requirements.txt` memakai `tensorflow-cpu` (bukan `tensorflow`).
Alasannya: package `tensorflow` menarik CUDA toolkit ~3 GB yang tidak akan
dipakai di server, dan bisa membuat deploy gagal karena ukuran.

**Repo ini hanya punya SATU file `requirements.txt`**, yaitu
`app/requirements.txt`. File untuk training diberi nama
`requirements-training.txt` (di root) supaya tidak mungkin salah pilih.
Alasannya dijelaskan di Bagian 3 Langkah 4."""))

ganti.append((
    """### 2.4 Cek ukuran repo

```powershell
git count-objects -vH
```

Kalau repo > 1 GB, pastikan `data/` tidak ikut ter-commit:

```powershell
git check-ignore -v data/raw/HAM10000_metadata.csv
```""",
    """### 2.4 Cek ukuran repo

```powershell
git count-objects -vH
```

Kalau repo > 1 GB, pastikan `data/` tidak ikut ter-commit:

```powershell
git check-ignore -v data/raw/HAM10000_metadata.csv
```

Repo proyek inisekitar **97,5 MB** dengan 142 file. Batas Streamlit
Community Cloud adalah 1 GB per repo dan 200 MB per berkas, jadi aman.

### 2.5 Simulasikan deploy SEBELUM push (sangat disarankan)

Skrip `scripts/simulasi_deploy.py` replicating apa yang terjadi di server
cloud: ia exporting isi git ke folder bersih (tanpa `data/`, tanpa
`.venv/`), lalu menjalankan aplikasi dari sana. Kalau langkah ini lulus,
k Chances besar deploy-nya juga lulus.

```powershell
python scripts/simulasi_deploy.py
```"""))

ganti.append((
    """### Langkah 4: Atur dependensi (jika perlu)

Streamlit membaca `requirements.txt` dari **root repository**. Karena
project kita punya dua file requirements, tambahkan file ini di root:

```txt
# requirements.txt (root) - sudah ada, tapi tambahkan tensorflow-cpu
```

Sebaiknya buat file `runtime.txt` atau `.python-version` di root:

```
3.13
```""",
    """### Langkah 4: Atur dependensi dan versi Python

**Dokumentasi resmi Streamlit Community Cloud** menyatakan dua hal penting:

1. Platform mencari file dependensi **mulai dari folder entrypoint, lalu
   root**, dan **hanya membaca file PERTAMA yang ditemukan**. Kalau ada
   dua file dependensi, hanya yang pertama yang dipakai.
2. PlatformQEJP menampilkan peringatan: *"You should only use one
   dependency file for your app."*

Karena entrypoint kita adalah `app/app.py`, platform akan membaca
`app/requirements.txt` (tensorflow-cpu). Repo ini sengaja hanya punya
satu `requirements.txt`--file di root bernama `requirements-training.txt`
(tensorflow penuh + jupyterlab) sehingga tidak mungkin salah pilih.

**Verifikasi sendiri:** hanya boleh ada satu berkas `requirements.txt`:

```powershell
git ls-files | Select-String "requirements.txt$"
# harus menghasilkan: app/requirements.txt
```

**Versi Python** dipilih lewat **dropdown "Python version"** di modal
**"Advanced settings"** pada halaman deploy, BUKAN lewat berkas
`runtime.txt` atau `.python-version` di repo. Pilih **3.13** karena itu
versi yang dipakai saat training, dan semua paket sudah di-pin ke versi
yang punya wheel Linux untuk 3.13 (dicek oleh
`scripts/cek_wheel_linux.py`).

Tidak perlu `packages.txt` karena aplikasi tidak butuh dependensi apt-get."""))

ganti.append((
    """## 4. MENGATUR UKURAN MODEL (BILANGAN PERLU)

Model MobileNetV2 dalam format `.keras` berukuran sekitar 14 MB. Ini **jauh
di bawah batas 100 MB per file** di GitHub, jadi **tidak perlu Git LFS**.""",
    """## 4. MENGATUR UKURAN MODEL (BILANGAN PERLU)

Model MobileNetV2 dalam format `.keras` berukuran **9,3 MB**. Ini **jauh
di bawah batas 100 MB per file** di GitHub, jadi **tidak perlu Git LFS**.
Berkas terbesar di repo adalah `models/cnn_best.keras` (25,3 MB), juga
jauh di bawah batas 200 MB per berkas di Streamlit Community Cloud."""))

ganti.append((
    """### Error: `No matching distribution found for tensorflow`

**Penyebab:** versi Python di platform tidak cocok.

**Solusi:** tambah `runtime.txt` atau `.python-version` di root repo berisi
`3.13`.""",
    """### Error: `No matching distribution found for tensorflow`

**Penyebab:** versi Python di platform tidak cocok dengan versi paket yang
di-pin.

**Solusi:** di halaman deploy, buka **"Advanced settings"** dan pilih
Python **3.13** pada dropdown "Python version". Platform Community Cloud
memakai Python 3.12 sebagai bawaan, jadi pilihan ini wajib.

Untuk memastikan semua versi paket memang punya wheel Linux 3.13, jalankan:

```powershell
python scripts/cek_wheel_linux.py
```

Skrip itu membaca `app/requirements.txt` dan memeriksa daftar berkas PyPI
setiap paket. Hasil terakhir: **10 dari 10 paket punya wheel manylinux
cp313** (atau merupakan paket pure-Python)."""))

ganti.append((
    """| `models/*.keras` | **BOLEH di-commit** (14 MB) | Tidak (sengaja) |
| `models/*.joblib` | **BOLEH di-commit** ( SVM ~10 MB) | Tidak (sengaja) |""",
    """| `models/*.keras` | **BOLEH di-commit** (9,3 MB) | Tidak (sengaja) |
| `models/*.joblib` | Tidak perlu di-commit (15 MB + 54 MB, dan aplikasi tidak memakainya) | Ya |
| `models/cnn_*_best.keras` (varian per-latihan) | Tidak perlu; bisa dibuat ulang dengan Notebook 05 | Ya |"""))

for lama, baru in ganti:
    if lama not in t:
        print(f"[!] TIDAK DITEMUKAN: {lama.splitlines()[0][:60]}")
    else:
        t = t.replace(lama, baru)
        print(f"[OK] {lama.splitlines()[0][:60]}")

(TMP / "PANDUAN_DEPLOY.md").write_text(t, encoding="utf-8")
print(f"\ndisimpan -> {TMP / 'PANDUAN_DEPLOY.md'}")

# =====================================================================
# PANDUAN_PENGGUNAAN.md
# =====================================================================
p2 = REPO / "docs" / "PANDUAN_PENGGUNAAN.md"
t2 = p2.read_text(encoding="utf-8")
n = t2.count("pip install -r requirements.txt")
t2 = t2.replace("pip install -r requirements.txt",
                "pip install -r requirements-training.txt")
t2 = t2.replace(
    "4. Tambahkan `requirements.txt` dari folder `app/`.",
    "4. Tidak perlu menambahkan apa pun. Platform membaca "
    "`app/requirements.txt` secara otomatis karena entrypoint berada di "
    "folder `app/`.")
(TMP / "PANDUAN_PENGGUNAAN.md").write_text(t2, encoding="utf-8")
print(f"PANDUAN_PENGGUNAAN.md: {n} rujukan diperbarui -> "
      f"{TMP / 'PANDUAN_PENGGUNAAN.md'}")
