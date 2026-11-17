# Catatan Temuan Diagnostik

Dokumen ini merekam temuan eksperimen diagnostik selama pengembangan
proyek. **Semua angka berasal dari kode yang benar-benar dijalankan**, dan
setiap tabel punya skrip yang bisa dijalankan ulang:

| Temuan | Skrip sumber angka | File keluaran |
|---|---|---|
| Desain fitur HOG + histogram | `scripts/diagnostik_fitur.py` | `docs/logs/diagnostik_fitur.json` |
| Ukuran sel HOG | `scripts/diagnostik_hog_cell.py` | `docs/logs/diagnostik_hog_cell.json` |
| Bobot histogram | `scripts/cek_bobot_histogram.py` | `docs/logs/cek_bobot_histogram.json` |

Split dan seed sama dengan notebook utama: `GroupShuffleSplit` per
`lesion_id`, seed 42, 6.959 gambar train / 1.529 validasi.

---

## 1. Temuan Utama: Random Forest pada Fitur Mentah Hampir Tidak Belajar Apa-Apa

### Pengukuran awal

| Model | F1 macro | Accuracy | Recall ganas |
|---|---|---|---|
| Baseline "selalu menebak nv" | 0,1148 | 0,6717 | 0,0000 |
| Random Forest, 1.812 fitur penuh | **0,1467** | 0,6815 | **0,0192** |

Random Forest pada seluruh fitur hanya sedikit di atas baseline, dan
recall ganas nyaris nol (0,019). Artinya model itu praktis selalu
memprediksi `nv` (kelas terbesar).

### Penyebabnya

Random Forest memilih fitur **secara acak** saat membelah node. Dengan
1.764 fitur HOG yang saling berkorelasi dan hanya 48 fitur histogram
warna yang diskriminatif, peluang RF memilih fitur histogram saat satu
split hanya:

```
48 / 1.812 = 2,65%
```

(feature importance yang diukur: **61,7%** berasal dari 48 fitur
histogram, sedangkan 30 komponen PCA menyumbang 38,3%. Jadi
informasi warna memang yang paling diskriminatif, tapi RF tidak
memilihnya secara konsisten karena mekanismenya memang acak.)

### Bukti tambahan: PCA pada SELURUH fitur lebih buruk

| Konfigurasi SVM | F1 macro | Recall ganas |
|---|---|---|
| PCA(30) dari 1.812 fitur | 0,2749 | 0,5160 |
| PCA(100) dari 1.812 fitur | 0,3189 | 0,4455 |
| PCA(400) dari 1.812 fitur | 0,3069 | 0,3429 |
| **PCA(30) HOG saja + histogram** | **0,3423** | 0,4776 |

Kalau PCA diterapkan ke seluruh fitur, komponennya didominasi HOG, dan
informasi warna hilang di antara komponen yang variance-nya lebih besar.

### Solusi yang berhasil

```
HOG (1.764)  ->  StandardScaler  ->  PCA(n)  ->  StandardScaler
histogram (48) -------------------------> dikali bobot -> GABUNGKAN
```

Kunci: **tidak ada StandardScaler global setelah penggabungan.** Scale
sebelum PCA (untuk HOG), scale lagi setelah PCA (untuk komponen), tapi
histogram dibiarkan apa adanya supaya bobot relatifnya bisa diatur.

---

## 2. Temuan Kedua: Semakin Banyak Komponen PCA, Semakin Buruk

Pengukuran dengan rantai di atas (SVC, bobot histogram 6):

| Komponen PCA | F1 macro | Recall ganas |
|---|---|---|
| 10 | 0,3351 | 0,5962 |
| **30** | **0,3423** | 0,4776 |
| 50 | 0,3218 | 0,4295 |
| 100 | 0,2708 | 0,2853 |
| 200 | 0,2371 | 0,1667 |
| 400 | 0,1910 | 0,1058 |

**Pola yang jelas: performance turun monotonik setelah 30 komponen.**
Alasannya: 48 fitur histogram hanya menyumbang sebagian kecil dari total
kontribusi komponen utama, jadi ketika PCA mengambil banyak komponen,
**hampir semua informasi warna hilang**.

Karena itu jumlah komponen dikunci di **30** (`C.PCA_N_COMPONENTS`).

---

## 3. Temuan Ketiga: Warna adalah Sinyal Terkuat, Bentuk HOG Saja Tidak Cukup

| Model (48 fitur histogram saja) | F1 macro | Accuracy | Recall ganas |
|---|---|---|---|
| SVC-RBF histogram saja | 0,3728 | 0,5716 | **0,7115** |
| Random Forest histogram saja | 0,3604 | **0,7011** | 0,2564 |

Ini mengonfirmasi temuan EDA di Notebook 03: **Fisher Score histogram
4,6x lebih tinggi** dari HOG (rata-rata 0,0207 vs 0,0045).

**Pola yang konsisten di seluruh proyek:**
- **SVC** menang di **recall ganas**.
- **Random Forest** menang di **accuracy**.

**Implikasi:** untuk aplikasi skrining, **recall ganas lebih penting**
daripada accuracy. Model dengan accuracy tinggi tapi recall ganas rendah
justru model yang paling berbahaya untuk skrining kanker. Ini alasan
kenapa accuracy tidak boleh jadi satu-satunya dasar pemilihan model.

---

## 4. Temuan Keempat: Bobot Histogram Hanya Berpengaruh pada SVM

Histogram dinormalisasi L1 (jumlah semua bin = 1,0 per gambar), sedangkan
komponen PCA di-standardisasi (rata-rata 0, simpangan 1). Tanpa bobot
eksplisit, informasi warna tenggelam di antara informasi bentuk.

Sweep bobot histogram. **Kolom kiri** memakai SVC polos (bandingkan
konfigurasi dengan cepat). **Kolom kanan** memakai model SVM yang
benar-benar dipakai proyek, yaitu SVC di dalam
`CalibratedClassifierCV` (`scripts/cek_bobot_histogram.py`):

| Bobot | SVC polos F1 | SVC polos rec ganas | SVM proyek F1 | SVM proyek rec ganas | RF F1 | RF rec ganas |
|---|---|---|---|---|---|---|
| 1 | 0,3070 | 0,4808 | 0,2113 | 0,1699 | 0,3361 | 0,1763 |
| 3 | 0,3252 | 0,4872 | 0,2062 | 0,1827 | - | - |
| 6 | 0,3423 | 0,4776 | 0,2153 | 0,1859 | 0,3363 | 0,1763 |
| 12 | 0,3472 | 0,4904 | 0,2156 | 0,1923 | - | - |
| 24 | 0,3712 | 0,5288 | **0,2247** | **0,2212** | - | - |
| 48 | 0,3868 | 0,6474 | 0,2201 | 0,2340 | - | - |
| 96 | **0,4096** | **0,6987** | 0,1892 | 0,1731 | 0,3363 | 0,1763 |
| 192 | - | - | 0,1291 | 0,0321 | - | - |
| 384 | - | - | 0,1419 | 0,0417 | - | - |

**Perhatikan selisihnya.** SVC polos naik terus sampai bobot 96, tapi SVM
proyek (yang dikalibrasi) **tidak** monotonic: puncaknya di bobot 24
(F1 0,2247) lalu jatuh drastis di bobot 96 ke atas (0,1892). Jadi
"bobot terbaik" tidak bisa ditentukan dari SVC polos saja -- harus
diukur pada model yang benar-benar dipakai. Detail soal ini ada di
Temuan Kelima.

Dengan bobot 6 (nilai yang dipakai proyek, dipilih dari pengukuran
sebelumnya), SVM proyek menghasilkan F1 macro 0,2153 dengan recall ganas
0,1859. Angka inilah yang dipakai di laporan.

Bobot 6 dipertahankan karena: (1) nilainya diukur, bukan asumsi;
(2) ketika bobot dinaikkan, model's performa justru turun (lihat
kolom SVM proyek); (3) Random Forest sama sekali tidak terpengaruh,
jadi tidak ada alasan mengubahnya.

### Temuan yang tidak terduga: Random Forest TIDAK terpengaruh sama sekali

Angka Random Forest **identik** dari bobot 3 sampai 96 (selisih hanya 1
prediksi antara bobot 1 dan bobot di atasnya). Ini bukan bug, tapi
konsekuensi langsung dari cara kerja pohon:

> Pohon membelah berdasarkan **urutan nilai** (mengapa nilai ini lebih
> besar dari itu), bukan berdasarkan jarak. Mengalikan seluruh fitur
> histogram dengan konstanta positif hanya menggeser titik batas split,
> tanpa mengubah partisi mana yang mungkin.

Jadi:

- **Bobot histogram hanya berpengaruh pada SVM** (RBF menghitung jarak
  Euclidean, sehingga skala fitur mengubah geometri data).
- **Bobot histogram tidak berpengaruh pada Random Forest** (pohon
  scale-invariant).

Konsekuensi untuk laporan: kalau hanya ada Random Forest, bobot histogram
tidak perlu disetel sama sekali. Bobot hanya relevan karena proyek ini
juga memakai SVM.

---

## 4b. Temuan Fifth: Cara Mendapatkan Probabilitas dari SVM Mengubah
## Performa secara Drastis

SVM wajib punya `predict_proba`, karena ambang batas "ganas/jinak"
dihitung dari probabilitas. Tapi ada beberapa cara mendapatkannya, dan
hasilnya **sangat berbeda** (semua diukur di data validasi, script
`scripts/cek_kalibrasi_svm.py`):

| Cara mendapatkan probabilitas | F1 macro | Accuracy | Recall ganas | Waktu training |
|---|---|---|---|---|
| SVC polos (tanpa probabilitas) -- **acuan** | **0,3437** | 0,6769 | **0,4904** | 13 s |
| `SVC(probability=True)` | 0,2957 | 0,7057 | 0,3365 | 54 s |
| `CalibratedClassifierCV(ensemble=False)` | 0,2151 | 0,6893 | 0,1891 | 26 s |
| **`CalibratedClassifierCV(ensemble=True)`** | 0,2713 | 0,7063 | 0,2981 | 14 s |

### Kenapa `ensemble=False` (contoh resmi sklearn) sangat buruk?

Dengan `ensemble=False`, sklearn:
1. Melatih 5 model pada 5 lipatan, dan **menghitung kalibrator** dari
   prediksi out-of-fold.
2. Lalu **melatih ulang** estimator dasar pada **seluruh** data latih.

Masalahnya: decision function model yang dilatih ulang itu bisa punya
skala berbeda dari model lipatan yang dipakai saat fitting kalibrator.
Akibatnya sigmoid kalibrator tidak lagi "seimbang" antar kelas, dan
argmax probabilitas berubah. Efeknya besar: recall ganas turun dari
**0,4904** menjadi **0,1891** -- model jadi hampir tidak bisa
mendeteksi lesi ganas.

Dengan `ensemble=True`, prediksi adalah **rata-rata probabilitas dari 5
model lipatan**. Semua decision function berada di skala yang sama
dengan kalibrator, jadi hasilnya jauh lebih konsisten.

### Keputusan proyek

Dipakai **`CalibratedClassifierCV(..., ensemble=True)`**, dengan alasan:
- API ini yang **resmi direkomendasikan** scikit-learn, bukan API
  deprecated (`SVC(probability=True)` akan dihapus di versi 1.11).
- Performanya **jauh lebih baik** dari `ensemble=False` (F1 macro 0,2151
  -> 0,2713; recall ganas 0,1891 -> 0,2981).
- Harganya kecil: training 14 detik, dan waktu inferensi tetap sama.

Pelajaran yang lebih luas: **parameter yang punya nilai bawaan pun
perlu diukur**, terutama kalau mengubah cara model mengambil
keputusan. `ensemble=False` adalah contoh resmi dari library, tapi
untuk dataset ini ia jelas lebih buruk.

---

## 5. Temuan Kelima: Block-Reduced HOG Tidak Lebih Baik di Konfigurasi Ini

Menggabungkan 4 sel HOG per blok menjadi rata-rata menghasilkan 441
dimensi (49 blok x 9 orientasi) alih-alih 1.764:

| Konfigurasi | F1 macro | Recall ganas |
|---|---|---|
| PCA(30) HOG penuh (1.764 dim) | **0,3423** | 0,4776 |
| Block-reduced PCA(30) (441 dim) | 0,3077 | 0,5160 |
| Block-reduced PCA(100) (441 dim) | 0,2913 | 0,3365 |

Block-reducing **tidak** unggul: vektor yang lebih ringkas justru
kehilangan informasi yang dibutuhkan. Jadi **HOG penuh + PCA(30)** yang
dipakai proyek.

---

## 5b. Temuan Sixth: Ukuran Sel HOG Hampir Tidak Berpengaruh pada Performa

Eksperimen dengan model yang sama persis untuk ketiga ukuran sel
(`scripts/diagnostik_hog_cell.py`, hasilnya
`docs/logs/diagnostik_hog_cell.json`):

| Ukuran sel | Dimensi HOG | F1 macro | Accuracy | Recall ganas | Waktu |
|---|---|---|---|---|---|
| 8x8 | 8.100 | 0,2163 | 0,6900 | 0,2179 | 181 s |
| **16x16** | **1.764** | 0,2118 | 0,6880 | 0,1859 | 27 s |
| 32x32 | 324 | 0,2175 | 0,6913 | 0,1667 | 92 s |

**Selisih F1 macro antar ketiganya cuma 0,0057** -- jauh lebih
kecil daripada selisih antar model di proyek ini (0,21 sampai 0,42).

### Konsekuensi untuk cara kita menjelaskan pilihan

Temuan ini **memperbaiki** penjelasan yang sebelumnya dipakai. coletiva
sebelumnya menulis "16x16 dipilih karena performanya paling tinggi",
dan itu **tidak didukung** oleh eksperimen dengan konfigurasi final.

Alasan yang benar (dan lebih kuat, karena bisa dipertanggungjawabkan):

1. **Hemat tanpa mengorbankan performa.** 1.764 dimensi adalah **4,6x**
   lebih sedikit dari 8.100 dimensi, sehingga ekstraksi fitur, PCA, dan
   training jauh lebih cepat. Saat performanya setara, pilih yang murah.
2. Sel 8x8 menghasilkan banyak dimensi yang saling berkorelasi (noise),
   sedangkan sel 32x32 menghilangkan bentuk dan tepi lesi yang justru
   penting untuk membedakan melanoma dari mola.
3. Vektor yang lebih kecil juga membuat file model dan joblib lebih
   ringan untuk dideploy.

Pelajaran metodologisnya: **sebelum mengklaim sebuah hyperparameter
"dipilih karena terbaik", pastikan eksperimennya mengukur hal itu
dengan konfigurasi final.** Klaim lama berasal dari eksperimen dengan
sebelum konfigurasi HOG dan SVM-nya berubah, jadi tidak bisa dipakai
lagi.

---

## 6. Temuan Keenam: Class Weight WAJIB untuk CNN, Tanpa Itu Model Hanya Bisa Menebak "nv"

Pengukuran pada CNN kecil (Rescaling -> Conv32 -> Pool -> Conv64 -> Pool ->
GAP -> Dense(7)), 10 epoch, data validasi 1.529 gambar:

| Konfigurasi | F1 macro | Accuracy | Recall ganas | Catatan |
|---|---|---|---|---|
| lr 1e-3 **+** class weight | 0,180 | 0,442 | **0,930** | Pelajar yang belajar |
| lr 1e-3 **tanpa** class weight | **0,115** | 0,672 | **0,000** | **Persis baseline** |
| lr 3e-4 **+** class weight | 0,177 | 0,402 | 0,853 | Pelajar yang belajar |
| lr 3e-4 **tanpa** class weight | **0,115** | 0,672 | **0,000** | **Persis baseline** |

**Temuan yang sangat penting:** tanpa `class_weight`, akurasi validasi
mengunci tepat di **0,672** (= proporsi kelas `nv`) dan recall ganas
**nol**. Train loss hanya turun dari 1,11 ke 0,96, yang untuk 7 kelas
(ln 7 = 1,95 = acak) berarti model **belum belajar apa pun**.

**Penjelasan teknisnya:** tanpa bobot kelas, fungsi loss didominasi
oleh kelas `nv` yang jumlahnya 67% dari data. Gradien yang diterima model
hampir seluruhnya berasal dari benar prediksi `nv`, sehingga model cepat
konverge ke "selalu jawab nv". Ini bukan bug kode, tapi konsekuensi
langsung dari imbalance yang sudah diidentifikasi di EDA.

**Konsekuensi untuk metodologi:**
1. `class_weight` bukan opsional di proyek ini, tapi **wajib**.
2. Angka **F1 macro 0,115 = baseline** menjadi **penanda bahaya**: kalau
   sebuah model mencapai angka itu, artinya ia tidak belajar apa pun dan
   harus diperbaiki, bukan dilaporkan sebagai hasil.
3. Trade-off yang muncul: dengan class weight, **recall ganas tinggi
   (0,93) tapi specificity rendah**. Model lebih sering bilang "ganas".
   Untuk **skrining** ini trade-off yang benar (lebih baik daripada
   melanoma lolos), dan inilah alasan ada threshold "ganas" di tuning
   Fase 7.

**Catatan waktu training:** konfigurasi dengan class weight butuh 4.930
detik, sementara yang tanpa class weight hanya 278 detik. Penyebabnya
bukan class weight-nya, tapi model yang tanpa class weight **langsung
konverge ke triviale** (val_acc terkunci di epoch 1), jadi training
selesai jauh lebih cepat. Ini bukti tambahan bahwa model tanpa class
weight benar-benar tidak belajar.

---

## 7. Temuan Ketujuh: Pipeline `tf.data` di WSL2 perlu Data di Disk NATIVE

Pengukuran kecepatan pipeline (`tf.data`, batch 32, gambar 128x128):

| Lokasi data | Konfigurasi | Waktu per epoch |
|---|---|---|
| `/mnt/m/...` (drive Windows via 9p) | 218 step | ~1,626 detik/step |
| `/mnt/f/...` (disk native WSL) | 218 step | **0,08 detik/step** |

Faktor **20x**. Penyebabnya: pembacaan file lewat mount 9p/drvfs Windows
memiliki latensi tinggi, dan `tf.data` itu I/O-bound.

Tambahan penting: **run pertama selalu lambat** (7,4 detik/step) karena
TensorFlow melakukan JIT kompilasi kernel CUDA dari PTX. RTX 5070 adalah
Blackwell (compute capability 12.0), dan TensorFlow 2.21 dibangun dengan
CUDA 12.5 yang **tidak punya binary kernel untuk sm_120**, jadi kernel
harus dikompilasi saat runtime:

> TensorFlow was not built with CUDA kernel binaries compatible with
> compute capability 12.0a. CUDA kernels will be jit-compiled from PTX,
> which could take 30 minutes or longer.

Setelah warm-up, kecepatannya 0,08 detik/step. Ini bukan error, dan
tidak perlu dikhawatirkan saat demo.

**Rekomendasi:** salin dataset ke disk native WSL sebelum training:
```bash
cp -r "/mnt/m/.../data/raw" /mnt/f/ml_project_data/raw
```
Kalau tidak punya ruang di disk WSL, training tetap bisa jalan di
Windows (CPU), hanya lebih lambat.

---

## 8. Temuan Kedelapan: `tf.py_function` di dalam `tf.data` Bisa DEADLOCK

Ini adalah temuan yang **sangat penting** dan debugging yang benar-benar
membuang waktu. Gejalanya:

```
Training MobileNetV2 stuck. Thread stopped after 16 menit:
  State     : S (sleeping)
  Threads   : 172
  wchan     : futex_do_wait
  GPU       : 0% utilization, 2.512 MiB terpakai
  RSS       : 4,8 GB
```

Semua thread `tf.data` tertidur di `futex_do_wait`, artinya mereka saling
menunggu tanpa ada yang maju. **Deadlock, bukan sekadar lambat.**

### Akar masalahnya

Augmentasi kompresi JPEG versi pertama memanggil PIL lewat `tf.py_function`:

```python
result = tf.py_function(func=_compress_batch, inp=[inputs], Tout=tf.float32)
```

`tf.py_function` menjalankan kode Python di dalam pipeline `tf.data`.
Dikombinasikan dengan `num_parallel_calls=tf.data.AUTOTUNE`, ini
**tidak aman**: scheduler AUTOTUNE mengambil worker thread berdasarkan
throughput, sementara `tf.py_function` menahan GIL saat memanggil PIL.
Hasilnya: pool thread saling menunggu yang tidak pernah selesai.

### Perbaikannya

Ganti dengan **TensorFlow murni**:

```python
def _roundtrip(img):
    encoded = tf.io.encode_jpeg(img, quality=self.quality)
    decoded = tf.io.decode_jpeg(encoded, channels=3)
    return tf.cast(decoded, tf.float32)

return tf.map_fn(_roundtrip, img_u8, fn_output_signature=tf.float32)
```

`tf.io.encode_jpeg` dan `tf.io.decode_jpeg` adalah op TensorFlow native,
sehingga aman di graph mode dan bisa berjalan paralel.

**Konsekuensi yang harus diterima:** `tf.io.encode_jpeg` menuntut
`quality` berupa bilangan bulat **statis**, bukan tensor. Jadi kualitas
kompresi TIDAK bisa diacak per gambar di dalam graph. Yang tetap diacak
adalah cahaya, kontras, dan hue (layer lain). Ini trade-off yang
sengaja dan bisa dijelaskan saat tanya jawab.

### Cara mendeteksi deadlock di masa depan

1. `nvidia-smi` menunjukkan 0% GPU padahal proses berjalan
2. `top -H` menunjukkan semua thread di status S (sleeping)
3. `/proc/<pid>/wchan` menunjukkan `futex_do_wait`
4. RSS besar tapi pemakaian CPU ~0%

---

## 9. Temuan Kesembilan: Arsitektur CNN Perlu BatchNormalization dan Lebih Lebar

Eksperimen pada data validasi (1.529 gambar), 8 epoch:

| Varian | Parameter | F1 macro | Accuracy | Recall ganas | Kelas terpakai |
|---|---|---|---|---|---|
| 32-64 + GAP (tanpa BN) | 19.847 | 0,140 | 0,396 | 0,811 | **3 dari 7** |
| 32-64 + GAP (dengan BN) | 20.231 | 0,149 | 0,336 | 0,862 | 5 dari 7 |
| 32-64-128 + GAP (dengan BN) | 374.407 | **0,254** | 0,354 | 0,558 | **7 dari 7** |
| 32-64-128 + GAP (tanpa BN) | 374.407 | 0,103 | 0,315 | 0,526 | 3 dari 7 |

**Temuan utama:** varian 32-64 hanya memprediksi **3 dari 7 kelas**,
sedangkan varian 32-64-128 dengan BatchNormalization memprediksi
**SEMUA 7 kelas**.

**Penjelasan teknisnya:**
- Tanpa BatchNormalization, distribusi aktivasi tiap layer bergeser terus
  selama training. Optimizer selalu mengejar target yang bergerak, dan
  training jadi tidak stabil.
- Menambah lebar (jumlah filter) memberi model lebih banyak kapasitas
  untuk membedakan 7 kelas yang memang mirip satu sama lain.

**Konsekuensi untuk proyek:** arsitektur CNN di notebook 05 dilatih dalam
TIGA varian, dan perbandingan pooling (GAP vs Flatten) dilakukan pada
varian yang sama supaya FAIR.

---

## 10. Temuan Kesepuluh: Fine-Tuning yang Justru Memperburuk Model

Di Notebook 06, MobileNetV2 dilatih dalam dua tahap:

1. **Tahap 1**: classifier baru, backbone ImageNet dibekukan, lr 1e-3.
2. **Tahap 2**: 30 lapisan teratas dilepas, lr 1e-5.

Hasil di data validasi (angka final ada di `docs/logs/`):

| Metrik (data validasi) | Tahap 1 (backbone beku) | Tahap 2 (fine-tuning) |
|---|---|---|
| F1 macro | 0,3934 | 0,3938 |
| Accuracy | 0,6122 | 0,6841 |
| **Recall ganas** | **0,5545** | 0,3750 |
| **FN ganas** | **139** | 195 |
| FP (false alarm) | 118 | 44 |
| Epoch dilatih | 10 | 6 |
| Epoch terbaik | 5 | 1 |
| Waktu training | 14.8 menit | 12.0 menit |

Catatan penting soal cara membandingkan: **kedua tahap dinilai di data
validasi yang sama, dengan augmentasi training yang sama.** Hanya ada
satu variabel yang beda: apakah backbone-nya dibekukan, dan learning
rate-nya (1e-3 vs 1e-5). Ini penting, karena kalau tahap 1 memakai
augmentasi berbeda dari tahap 2, selisih hasil tidak bisa dipisahkan
antara "efek fine-tuning" dan "efek perubahan augmentasi".

Hasilnya: F1 macro praktis sama (0,3934 vs
0,3938), tapi **recall ganas tahap 2 jauh lebih
rendah** (0,5545 ->
0,3750, FN 139 ->
195). Fine-tuning justru membuat model lebih
sering misses lesi ganas.

Untuk aplikasi skrining, FN ganas adalah kesalahan paling mahal,
jadi **tahap 1 yang dipakai**.

> Angka lengkapnya ada di `docs/logs/model_mobilenetv2_validasi.json`.

Fine-tuning pada dataset sekecil ini justru **memperburuk** sensitivitas.
Dua sebab yang masuk akal:
- **Data terlalu sedikit** (6.959 gambar) untuk 30 lapisan pretrained
  yang di-fine-tune tanpa risiko overfit.
- **Learning rate kecil (1e-5) untuk 30 lapisan** menghasilkan
  perubahan yang tidak cukup berarti, tapi sudah cukup untuk menggeser
  bobot away dari representasi ImageNet yang memang bagus.

Kesimpulan diambil **berdasarkan angka**, bukan harapan: model tahap 1 yang
dipakai di aplikasi.

---

## 11. Bug Nyata yang Ditemukan Lewat Eksekusi (bukan asumsi)

Tujuh bug berikut ditemukan karena kodenya **dijalankan**, bukan karena
sekilas dibaca. Semuanya sudah diperbaiki dan ada skrip pemeriksa yang
mencegah kesalahan sejenis terulang.

| # | Bug | Gejala | Perbaikan |
|---|---|---|---|
| 1 | `tf.py_function` di `tf.data` | Deadlock, 172 thread sleep, GPU 0% | `tf.io.encode_jpeg`/`decode_jpeg` murni TF |
| 2 | `lambda` di `FunctionTransformer` | `PicklingError`, model tak bisa disimpan | kelas `HistogramScaler` (top-level) |
| 3 | `np.argmax(val_loss) * -1` | Indeks negatif, epoch terbaik salah | `np.argmin` |
| 4 | `model.predict_proba` di Keras 3 | `AttributeError` pada Sequential | `E.predict_with_proba` + verifikasi jumlah baris = 1 |
| 5 | `palambatte=` pada `sns.boxplot` | `TypeError: unexpected keyword argument` | `palette=` |
| 6 | Nama variabel `train_ds` vs `train_ds_smart` | `NameError` di notebook 06 | Diselaraskan, dicek `check_undefined_names.py` |
| 7 | Tanda gap overfitting terbalik | Overfitting parah dikategorikan "SEHAT" | `val_loss - train_loss` + gap relatif |
| 8 | `CalibratedClassifierCV(ensemble=False)` | Recall ganas SVM anjlok 0,49 -> 0,19 | `ensemble=True`, diukur di `scripts/cek_kalibrasi_svm.py` |
| 9 | Vektor fitur 1D masuk ke `predict` sklearn | `ValueError: X does not contain any features` | `.reshape(1, -1)` |
| 10 | Typo `gambar_uggahan` vs `gambar_unggahan` | `NameError` di panel hasil aplikasi | Penamaan diseragamkan + dicek `check_undefined_names.py` |
| 11 | `make_notebooks.py` menimpa output notebook | Output notebook hilang diam-diam | Lewati notebook yang sudah punya output, kecuali `--force` |
| 12 | Ambang model terpilih dipakai untuk semua model di uji eksternal | Perbandingan domain shift tidak adil | Ambang per-model diambil dari `hasil_evaluasi_final.json` |
| 13 | Band "area abu-abu" di `is_uncertain` tetap 0,15 | Hampir semua foto dilabel "belum meyakinkan" | Band mengikuti skala ambang (`0,30 x ambang`, min 0,02) |

Bug nomor 5 sampai 12 punya kesamaan: semuanya **tidak terlihat dari
membaca kode**, hanya dari menjalankannya.

Bug 12 deserves penjelasan khusus karena ia bisa membuat angka terlihat
**lebih bagus dari kenyataan**. `models/threshold_config.json` hanya
menyimpan ambang untuk model yang dipilih (0,11 untuk MobileNetV2).
Kode uji eksternal awalnya memakai ambang itu untuk keempat model,
padahal ambang masing-masing sudah dituning terpisah di validasi
(MobileNetV2 0,11 · CNN 0,12 · RF 0,20 · SVM 0,17). Efeknya: Random
Forest dan SVM dievaluasi jauh lebih permisif dari titik operasi yang
sebenarnya, sehingga recall di dataset eksternal melonjak dan
perbandingannya menyesatkan. Perbaikannya adalah mengambil ambang
per-model dari `hasil_evaluasi_final.json` -- file yang sama dengan
bagian "Recall HAM" di notebook uji eksternal, sehingga kedua sisi
perbandingan memakai ambang yang identik.

Bug 13 juga deserve penjelasan, karena ia memengaruhi apa yang dilihat
penonton video demo. `is_uncertain()` di aplikasi deciding memakai lebar
area abu-abu tetap 0,15 di sekitar ambang "ganas". Ketika ambang
MobileNetV2 berubah dari 0,25 menjadi 0,11 (setelah model dilatih ulang
dengan data augmentasi smartphone yang sama), seluruh rentang risiko
jinak -- 0 sampai 0,11 -- ikut masuk ke dalam area abu-abu itu. Hasilnya
**semua** foto lesi jinak diberi label "belum meyakinkan", dan kotak
hijau "Tidak ditemukan ciri ganas" nyaris tidak pernah muncul di
aplikasi.

Diperiksa dengan `scripts/cek_uncertain.py` pada empat contoh gambar:

| Contoh | Risiko | Probabilitas teratas | Logika lama | Logika baru |
|---|---|---|---|---|
| Nevus jinak | 0,070 | 0,469 | abu-abu | **pasti** |
| Melanoma (dermoskop) | 0,529 | 0,481 | abu-abu | **pasti** |
| Karsinoma sel basal | 0,610 | 0,309 | abu-abu | abu-abu |
| Melanoma (smartphone) | 0,024 | 0,951 | abu-abu | **pasti** |

Dengan logika lama, keempat contoh masuk kategori abu-abu. Dengan
logika baru, hanya satu -- dan itu memang kasus yang tepat untuk
diingatkan: pada foto melanoma versi smartphone, model justru **yakin**
bahwa itu nevus. Kegagalan itu adalah temuan domain shift yang
dilaporkan secara terbuka di bagian uji eksternal.

Karena itu proyek ini punya empat pemeriksa otomatis:

| Skrip | Yang diperiksa |
|---|---|
| `scripts/check_api_calls.py` | Argumen fungsi yang tidak ada di API library terpasang |
| `scripts/check_undefined_names.py` | Variabel yang dipakai sebelum didefinisikan |
| `scripts/check_indonesian_text.py` | Kalimat campuran Inggris / teks rusak |
| `scripts/sanity_check_text.py` | Karakter CJK dan BOM di semua file sumber |

---

## 12. Implikasi ke Laporan

Temuan-temuan di atas adalah **kontribusi intelektual proyek ini**, bukan
sekadar catatan teknis:

1. **Feature selection berbasis bukti, bukan asal pakai.** Kita bisa
   membuktikan dengan angka kenapa 1.812 fitur mentah tidak working dan
   apa yang menggantikannya.
2. **Justifikasi pemilihan model yang kuat.** "Model dengan accuracy
   tertinggi" dan "model terbaik untuk skrining" bisa berbeda, dan kita
   punya angka untuk keduanya.
3. **Menjawab pertanyaan "kenapa tidak pakai HOG saja?"** dengan jawaban
   berbasis eksperimen.
4. **Menjelaskan keterbatasan model klasik** dengan sebab yang terukur:
   HOG + histogram memang tidak cukup untuk membedakan 7 kelas dengan
   baik. Justru inilah yang memotivasi CNN dan MobileNetV2, yang belajar
   representasi yang lebih dalam.
5. **Menunjukkan bahwa bobot fitur tidak berlaku universal.** Random
   Forest dan SVM punya perilaku berbeda terhadap skala fitur, dan itu
   menjelaskan kenapa satu setelan bobot tidak bisa dipakai untuk semua
   model.

Semua eksperimen di atas dapat direproduksi dengan skrip di folder
`scripts/`.
