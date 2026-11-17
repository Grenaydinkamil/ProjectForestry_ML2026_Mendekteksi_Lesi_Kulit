# LAPORAN PROYEK AKHIR PEMBELAJARAN MESIN

## Klasifikasi Lesi Kulit Berbasis Citra dengan HOG-SVM, HOG-Random Forest, CNN, dan MobileNetV2, Termasuk Evaluasi Domain Shift pada Foto Smartphone

| Keterangan | Isi |
|---|---|
| Mata kuliah | Pembelajaran Mesin (ML2026) |
| Kelompok | Kelompok 1 — ML2026 |
| Anggota 1 | Arken Yusufa Ibrahim (J0404241166) |
| Anggota 2 | Ghazali Habibie A (J0404241145) |
| Anggota 3 | Nabil Zulfan Nurtsani (J0404241052) |
| Dosen pengampu | Dosen Pengampu ML2026 |
| Tanggal | November 2026 |
| Status dokumen | Final — seluruh angka diambil dari `docs/logs/*.json` |

---

## ABSTRAK

Proyek ini membangun dan membandingkan empat model pembelajaran mesin untuk
klasifikasi tujuh kelas lesi kulit dari foto dermoskop HAM10000: HOG-SVM,
HOG-Random Forest, Convolutional Neural Network (CNN) dari nol, dan
MobileNetV2 transfer learning. Dataset terdiri dari 10.015 foto yang
merepresentasikan 7.470 lesi unik, dengan ketidakseimbangan kelas 58,3:1
(`nv` 66,9% berbanding `df` 1,1%). Pembagian data dilakukan per `lesion_id`
menggunakan `GroupShuffleSplit` dengan seed 42 menjadi 6.959 data latih,
1.529 data validasi, dan 1.527 data uji, disertai tiga assert anti-kebocoran
yang semuanya bernilai nol.

Rekayasa fitur klasik dirancang berdasarkan eksperimen terukur, bukan
asumsi: HOG dengan sel 16×16 (1.764 dimensi) digabung dengan histogram warna
(48 dimensi), PCA hanya diterapkan pada bagian HOG, dan histogram diberi
bobot eksplisit tanpa normalisasi ulang. Untuk model deep learning,
class weight wajib digunakan--tanpanya CNN mengunci accuracy pada 0,672
dengan recall ganas nol. MobileNetV2 dilatih dua tahap; perbandingan adil
pada data validasi yang sama menunjukkan fine-tuning tidak membantu,
sehingga model tahap 1 yang dipilih.

Pada data uji yang dipakai sekali, MobileNetV2 mencapai F1 macro 0,3993,
accuracy 0,6176, recall ganas 0,4500, dan specificity 0,8720, mengungguli
tiga model lain pada F1 macro. Model ini juga paling tahan terhadap
domain shift: pada uji eksternal PAD-UFES-20 (2.298 foto smartphone),
recall ganasnya 0,8730 dengan specificity 0,4489--penurunan yang jauh
lebih terkendali dibanding model klasik. Aplikasi Streamlit dibangun
dengan preprocessing identik dengan training, ambang risiko ganas dari
data validasi, dan disclaimer medis yang tidak bisa dilewatkan.

Seluruh hasil disertai tiga belas temuan diagnostik dan perbaikan bug yang
ditemukan lewat eksekusi nyata, antara lain deadlock `tf.py_function`,
kalibrasi SVM yang menurunkan recall ganas dari 0,4904 ke 0,1891, serta
penggunaan satu ambang untuk semua model pada uji eksternal yang membuat
perbandingan tidak adil. Semua sudah diperbaiki dan diukur ulang.

**Kata kunci:** klasifikasi lesi kulit, HAM10000, PAD-UFES-20, HOG, SVM,
Random Forest, CNN, MobileNetV2, domain shift, class weight, F1 macro.

---

## DAFTAR ISI

- Bab 1 Pendahuluan
- Bab 2 Tinjauan Pustaka
- Bab 3 Metodologi Penelitian
- Bab 4 Hasil Penelitian
- Bab 5 Pembahasan
- Bab 6 Kesimpulan dan Saran
- Daftar Pustaka
- Lampiran A — Daftar Bug yang Ditemukan dan Diperbaiki
- Lampiran B — Status Proyek dan Tindak Lanjut

---

## BAB 1 — PENDAHULUAN

### 1.1 Latar Belakang

Kanker kulit adalah salah satu jenis kanker paling umum, dan melanoma
merupakan penyebab kematian terbesar di antara penyakit kulit ganas. Deteksi
dini secara signifikan meningkatkan peluang kesembuhan. Namun, pemeriksaan
dermoskopi memerlukan tenaga ahli dan peralatan khusus yang tidak tersedia
di semua fasilitas, terutama di daerah dengan akses terbatas.

Machine learning menawarkan potensi sebagai alat bantu skrining awal:
model dapat memberi perkiraan risiko keganasan dari sebuah foto, sehingga
membantu tenaga kesehatan memprioritaskan kasus yang perlu diperiksa lebih
lanjut. Tantangannya, data citra lesi kulit sangat tidak seimbang dan
berasal dari dua domain yang berbeda: foto dermoskop dengan pencahayaan
terkontrol, dan foto smartphone dengan pencahayaan dan kualitas bervariasi.

### 1.2 Rumusan Masalah

1. Bagaimana merancang pipeline klasifikasi yang adil untuk data dengan
   ketidakseimbangan kelas 58,3:1?
2. Bagaimana performa empat pendekatan--HOG-SVM, HOG-Random Forest, CNN
   dari nol, dan MobileNetV2--dibandingkan secara objektif?
3. Seberapa besar penurunan performa ketika model dilatih pada foto
   dermoskop lalu diuji pada foto smartphone?
4. Bagaimana merancang aplikasi yang preprocessing-nya identik dengan
   training sehingga hasil di aplikasi tidak menyimpang dari laporan?

### 1.3 Tujuan

- Membangun pipeline data tanpa kebocoran informasi antar split.
- Membandingkan empat model dengan metrik yang sesuai untuk data tidak
  seimbang (F1 macro dan recall ganas), bukan accuracy semata.
- Mengukur domain shift dari HAM10000 ke PAD-UFES-20 tanpa training ulang.
- Menghasilkan aplikasi web yang siap didemonstrasikan disertai panduan
  penggunaan.

### 1.4 Manfaat

Proyek ini menyumbang tiga hal yang dapat dipertanggungjawabkan:
**pertama**, desain fitur berbasis bukti eksperimen; **kedua**, prosedur
pemilihan model yang tidak terjebak pada accuracy; **ketiga**, dokumentasi
lengkap dari 13 bug nyata yang ditemukan lewat eksekusi, termasuk
perbaikannya, sehingga dapat dijadikan bahan pembelajaran.

### 1.5 Ruang Lingkup

Penelitian menggunakan dua dataset publik (HAM10000 dan PAD-UFES-20),
tujuh kelas diagnosis, dan empat arsitektur model. PAD-UFES-20 **hanya**
dipakai untuk pengujian eksternal, tidak pernah untuk training atau
tuning. Aplikasi ditujukan untuk keperluan edukasi, bukan diagnosis medis.

---

## BAB 2 — TINJAUAN PUSTAKA

Tujuh artikel digunakan sebagai dasar metodologi dan referensi domain.
Seluruhnya diverifikasi otomatis ke PubMed (pencocokan judul, tahun, dan
DOI) oleh `scripts/verify_literature.py`; hasilnya tersimpan di
`docs/logs/verifikasi_literature.json`.

| No | Penulis (tahun) | Judul / topik | Kegunaan dalam proyek |
|---|---|---|---|
| 1 | Tschandl dkk. (2018) | *The HAM10000 dataset: a large collection of multi-source dermatoscopic images of common pigmented skin lesions* | Sumber dataset utama; karakteristik 7 kelas dan ketidakseimbangan |
| 2 | Pacheco dkk. (2020) | *PAD-UFES-20: a skin lesion dataset composed of patient data and clinical images collected from smartphones* | Dataset uji eksternal untuk domain shift |
| 3 | Fogelberg dkk. (2023) | *Domain shifts in dermoscopic skin cancer datasets* | Landasan interpretasi penurunan performa lintas domain |
| 4 | Combalia dkk. (2022) | *Validation of artificial intelligence prediction models for skin cancer diagnosis* | Justifikasi evaluasi model pada data independen |
| 5 | Almaraz-Damian dkk. (2020) | *Melanoma and nevus skin lesion classification using handcraft and deep learning* | Perbandingan pendekatan fitur handcraft versus deep learning |
| 6 | Katsch dkk. (2022) | *Comparison of CNN architectures for robustness against common corruptions* | Dasar augmentasi gaya smartphone (kompresi JPEG, cahaya, warna) |
| 7 | Rotemberg dkk. (2021) | *A patient-centric dataset of images and metadata for identifying melanomas* | Referensi praktik pengumpulan data berbasis pasien dan isu privasi |

DOI masing-masing tercatat pada file log verifikasi. Tidak ada sitasi yang
dikarang: skrip verifikasi mengembalikan status `found`, `title_match`,
`year_match`, dan `doi_match` bernilai benar untuk ketujuh artikel.

---

## BAB 3 — METODOLOGI PENELITIAN

### 3.1 Dataset dan Pemeriksaan Integritas

HAM10000 berisi 10.015 foto RGB beresolusi 600×450 piksel dari 7.470 lesi
unik. Pemeriksaan integritas menunjukkan **0 file hilang** dan **0 file
kosong**. Nilai `age` kosong pada 57 gambar (0,57%) dan **tidak diimputasi**
karena tidak dipakai sebagai fitur. Sebanyak 53,3% label berasal dari
histopatologi. Seluruh gambar berasal dari kulit Fitzpatrick I-II--ini
menjadi keterbatasan demografis yang serius karena populasi Indonesia
didominasi Fitzpatrick III-V.

### 3.2 Pembagian Data Anti-Kebocoran

| Split | Gambar | Lesi | Proporsi |
|---|---:|---:|---:|
| Latih | 6.959 | 5.228 | 70% |
| Validasi | 1.529 | 1.121 | 15% |
| Uji | 1.527 | 1.121 | 15% |

Pembagian memakai `GroupShuffleSplit` per `lesion_id` dua tahap dengan seed
42. Tiga assert memeriksa tidak ada overlap lesi, gambar, dan path antar
split--ketiganya menghasilkan nol. Class weight dihitung **hanya dari data
latih** (`df` 2,39; `nv` 0,05). Test set disentuh sekali di Notebook 07,
setelah semua hyperparameter dan ambang batas dikunci dari validasi.

### 3.3 Rekayasa Fitur (Model Klasik)

Desain fitur ditentukan oleh eksperimen, bukan kebiasaan:

```
HOG (sel 16×16 → 1.764 dimensi)  →  StandardScaler  →  PCA(30)  →  StandardScaler
Histogram warna RGB (48 dimensi) →  dikali bobot 6  (TIDAK di-scale ulang)
                                          ↓
                                   digabung → SVC / RandomForest
```

Temuan yang membentuk desain ini:

- Random Forest pada 1.812 fitur mentah mencapai F1 macro **0,1467**,
  nyaris sama dengan baseline "selalu menebak nv" (**0,1148**), dengan
  recall ganas **0,0192**. Penyebabnya: RF memilih fitur secara acak saat
  membelah node, sehingga peluang memilih 48 fitur histogram hanya
  48/1.812 = **2,65%**.
- PCA yang diterapkan ke seluruh fitur lebih buruk: PCA(400) dari 1.812
  fitur hanya F1 **0,3069**, sedangkan PCA(30) pada HOG saja + histogram
  mencapai **0,3423**.
- Jumlah komponen PCA pada HOG sebaiknya 30: F1 turun monoton menjadi
  0,3218 (50), 0,2708 (100), 0,2371 (200), dan 0,1910 (400).
- Histogram warna saja lebih kuat dari HOG: SVC pada 48 fitur histogram
  mencapai F1 **0,3728** dan recall ganas **0,7115**. Ini mengonfirmasi
  Fisher Score di Notebook 03 yang 4,6× lebih besar untuk histogram.

Eksperimen ukuran sel HOG (`scripts/diagnostik_hog_cell.py`) memberi hasil
F1 0,2163 (8×8), 0,2118 (16×16), dan 0,2175 (32×32)--selisihnya hanya
0,0057. Jadi 16×16 dipilih bukan karena skornya tertinggi, melainkan karena
1.764 dimensi adalah 4,6× lebih hemat dari 8.100 dimensi tanpa
mengorbankan performa.

Bobot histogram dengan SVC naik sampai bobot 96 (F1 0,4096), tetapi pada
model SVM yang dipakai proyek (dengan kalibrasi) angkanya justru turun
di atas bobot 24. Bobot 6 dipertahankan karena diukur stabil pada model
aktual. Random Forest sama sekali tidak terpengaruh bobot, karena pohon
membelah berdasarkan urutan nilai, bukan jarak.

### 3.4 Pemodelan Deep Learning

**CNN dari nol.** Tiga varian dilatih dengan pengaturan identik kecuali
satu variabel: arsitektur 32-64 + GAP (20.231 parameter), 32-64-128 + GAP
(374.407), dan 32-64-128 + Flatten (2.207.623). BatchNormalization dan
class weight wajib: varian tanpa class weight mengunci accuracy 0,672
dengan recall ganas nol; varian 32-64 hanya memprediksi 3 dari 7 kelas.

**MobileNetV2.** Transfer learning dua tahap dengan input 224×224 dan
normalisasi `Rescaling(1/127.5, offset=-1.0)` di dalam model. Tahap 1:
classifier baru, backbone ImageNet dibekukan, lr 1e-3. Tahap 2: 30 lapisan
teratas dilepas dengan BatchNormalization tetap dikunci, lr 1e-5. Kedua
tahap memakai augmentasi training yang **sama** (termasuk gaya smartphone),
sehingga perbandingannya hanya mengukur efek fine-tuning.

### 3.5 Strategi Evaluasi

Metrik utama adalah **F1 macro** dan **recall ganas** (sensitivitas),
dilengkapi accuracy, specificity, AUC, FN, FP, dan waktu inferensi. FN ganas
adalah kesalahan paling mahal dalam skrining. Ambang batas "ganas/jinak"
dituning pada data validasi dengan target recall 90%, lalu dikunci sebelum
data uji dipakai. Setiap model memakai ambangnya sendiri; menggunakan satu
ambang untuk semua model akan membuat perbandingan tidak adil.

---

## BAB 4 — HASIL PENELITIAN

### 4.1 Hasil di Data Validasi

**Model klasik.** SVM F1 macro 0,2826 / accuracy 0,7037 / recall ganas
0,2917. Random Forest F1 macro 0,3491 / accuracy 0,7109 / recall ganas
0,1987.

**CNN dari nol.**

| Varian | Parameter | F1 macro | Accuracy | Recall ganas | Kelas terpakai |
|---|---:|---:|---:|---:|---:|
| 32-64 + GAP | 20.231 | 0,3067 | 0,5546 | 0,1795 | 7/7 |
| 32-64-128 + GAP | 374.407 | 0,0960 | 0,4088 | 0,4776 | 3/7 |
| **32-64-128 + Flatten** | 2.207.623 | **0,3420** | 0,5298 | **0,6346** | 7/7 |

**MobileNetV2--perbandingan adil dua tahap.**

| Metrik | Tahap 1 (backbone beku) | Tahap 2 (fine-tuning) |
|---|---:|---:|
| F1 macro | 0,3934 | 0,3938 |
| Accuracy | 0,6122 | 0,6841 |
| **Recall ganas** | **0,5545** | 0,3750 |
| **FN ganas** | **139** | 195 |
| FP | 118 | 44 |
| Epoch dilatih | 10 | 6 |
| Waktu training | 14,8 menit | 12,0 menit |

F1 macro praktis sama, tetapi recall ganas tahap 2 jauh lebih rendah dan
FN meningkat 56 kasus. Karena FN ganas paling mahal dalam skrining,
**tahap 1 yang dipilih**.

### 4.2 Hasil di Data Uji (1.527 gambar, dipakai sekali)

| Model | Accuracy | F1 macro | Recall ganas | Specificity | Threshold | FN | FP | Inferensi |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **MobileNetV2** | 0,6176 | **0,3993** | 0,4500 | 0,8720 | 0,11 | 39 | 550 | 59,3 ms |
| CNN from scratch | 0,5534 | 0,3634 | 0,5800 | 0,6862 | 0,12 | 28 | 635 | 9,6 ms |
| Random Forest (HOG) | 0,7132 | 0,3240 | 0,2600 | 0,9625 | 0,20 | 25 | 481 | 6,4 ms |
| SVM (HOG) | 0,6929 | 0,2642 | 0,2500 | 0,9470 | 0,17 | 42 | 559 | 10,2 ms |

MobileNetV2 terpilih dengan skor gabungan 0,7043 (bobot: recall ganas 35%,
F1 macro 25%, kesiapan domain shift 20%, efisiensi 20%). Perlu ditekankan:
model dengan **accuracy tertinggi** adalah Random Forest (0,7132), tetapi
recall ganasnya hanya 0,2600--ia melewatkan 75% lesi ganas. Ini bukti
bahwa accuracy tidak boleh jadi dasar pemilihan untuk skrining.

Analisis overfitting (selisih F1 train versus test): MobileNetV2 +0,0285,
CNN +0,0221, SVM +0,0396 (semua "sedikit overfitting"), sedangkan Random
Forest +0,6760 ("OVERFITTING PARAH": F1 latih 1,0000 versus uji 0,3240).

### 4.3 Uji Eksternal: PAD-UFES-20

Dataset PAD-UFES-20 (DOI 10.17632/zr7vgbcyr2, lisensi CC BY 4.0) terdiri
dari 2.298 foto smartphone dengan 6 diagnosis. Model tidak dilatih ulang;
ambang masing-masing diambil dari validasi.

| Model | Rec HAM | Rec PAD | Spec HAM | Spec PAD | FN PAD | FP PAD |
|---|---:|---:|---:|---:|---:|---:|
| CNN from scratch | 0,5800 | 0,9698 | 0,6862 | 0,0104 | 55 | 474 |
| **MobileNetV2** | 0,4500 | **0,8730** | 0,8720 | **0,4489** | 231 | 264 |
| SVM (HOG) | 0,2500 | 0,8686 | 0,9470 | 0,2589 | 239 | 355 |
| Random Forest (HOG) | 0,2600 | 0,8362 | 0,9625 | 0,2860 | 298 | 342 |

Recall ganas yang tinggi di PAD-UFES-20 **tidak** serta-merta berarti
model lebih baik. CNN memiliki recall 0,9698 tetapi specificity hanya
0,0104--ia menandai hampir semua foto sebagai ganas. MobileNetV2 memiliki
keseimbangan terbaik (recall 0,8730, specificity 0,4489).

Recall per diagnosis: ACK 0,8781 (n=730), BCC 0,8923 (n=845), MEL 0,6346
(n=52), NEV 0,5697 (n=244), SCC 0,8333 (n=192), SEK 0,3234 (n=235). SCC
tidak ada di HAM10000 sehingga merupakan kelas yang belum pernah dilihat
model ("unseen class"). MEL hanya 52 gambar, sehingga angkanya tidak stabil.

Perbedaan teknis kedua domain: resolusi 600×450 (HAM) versus 969×969 (PAD);
mean kanal merah 174,3 versus 168,9; ketajaman (varians Laplacian) 131,0
versus 224,2. Seluruhnya menggeser distribusi citra yang dilihat model.

### 4.4 Aplikasi Streamlit

Aplikasi `app/app.py` menyediakan unggah foto atau pemilihan contoh gambar,
tujuh bar probabilitas per kelas, kotak status tiga warna (aman, waspada,
tidak yakin), catatan privasi, dan disclaimer medis. Preprocessing dibaca
dari `models/preprocess_config.json` yang ditulis otomatis oleh Notebook
07 untuk model terpilih, sehingga ukuran gambar dan normalisasi tidak
bisa tidak sinkron. Pengujian memakai `streamlit.testing.v1.AppTest`:
kondisi kosong, unggah gambar (7 bar muncul), file rusak (pesan ramah), dan
fitur contoh gambar semuanya lulus.

---

## BAB 5 — PEMBAHASAN

### 5.1 Kontribusi Ilmiah

**Desain fitur berbasis bukti.** Kita dapat menunjukkan dengan angka
mengapa fitur mentah gagal, mengapa PCA global merugikan, dan mengapa
histogram warna lebih informatif. Seluruh angka dapat direproduksi dengan
`scripts/diagnostik_fitur.py`.

**Pemilihan model yang jujur.** Model accuracy tertinggi bukan model
terbaik untuk skrining. F1 macro dan recall ganas dipakai sebagai kriteria
utama, dengan FN ganas sebagai metrik pengingat paling penting.

**Fine-tuning tidak selalu membantu.** Perbandingan adil menunjukkan
fine-tuning 30 lapisan atas dengan lr 1e-5 menurunkan recall ganas dari
0,5545 menjadi 0,3750. Pada dataset sekecil ini, membuka representasi
pretrained justru merusak generalisasi.

**Hyperparameter bawaan tidak selalu benar.** Contoh resmi scikit-learn
menyarankan `CalibratedClassifierCV(ensemble=False)`, tetapi pengukuran
kita menunjukkan `ensemble=True` jauh lebih baik (recall ganas 0,2981
berbanding 0,1891). Penyebabnya: dengan `ensemble=False`, kalibrator
dihitung dari model lipatan, lalu estimator dilatih ulang pada seluruh
data sehingga skala decision function-nya bergeser.

**Model pohon kebal terhadap skala fitur.** Bobot histogram tidak
mengubah Random Forest sama sekali, karena pohon membelah berdasarkan
urutan nilai. Bobot hanya relevan untuk SVM yang menghitung jarak.

### 5.2 Bug yang Ditemukan Lewat Eksekusi

Tiga belas bug ditemukan dan diperbaiki. Empat di antaranya bersifat
kritis:

1. `tf.py_function` di dalam `tf.data` menyebabkan deadlock--172 thread
   tidur di `futex_do_wait` dengan GPU 0%. Diganti `tf.io.encode_jpeg`
   dan `tf.io.decode_jpeg` murni TensorFlow.
2. `lambda` di `FunctionTransformer` tidak bisa di-pickle, sehingga model
   tidak bisa disimpan. Diganti kelas `HistogramScaler`.
3. `np.argmax(val_loss) * -1` menghasilkan indeks negatif sehingga epoch
   terbaik salah. Diganti `np.argmin`.
4. `CalibratedClassifierCV(ensemble=False)` menurunkan recall ganas SVM
   dari 0,4904 ke 0,1891. Diganti `ensemble=True`.

Bug lengkap ada di Lampiran A dan `docs/CATATAN_TEMUAN_DIAGNOSTIK.md`.

### 5.3 Keterbatasan

- Dataset hanya mencakup kulit Fitzpatrick I-II; tidak ada representasi
  kulit gelap, sementara populasi sasaran di Indonesia terutama III-V.
- Kelas `df` dan `vasc` sangat sedikit, sehingga metrik per kelasnya
  tidak stabil.
- Model belum divalidasi klinis. Semua angka adalah performa pada dataset
  publik, bukan pada pasien nyata.
- Presisi kelas minoritas rendah; model sering mengira lesi jinak sebagai
  ganas (banyak false alarm).
- Domain shift tetap terukur: performa pada foto smartphone lebih rendah,
  dan satu contoh melanoma smartphone diprediksi sebagai nevus dengan
  keyakinan tinggi--ini ditampilkan jujur di panduan video demo.

### 5.4 Ancaman Validitas dan Cara Mengatasinya

- **Kebocoran lesi:** diatasi dengan split per `lesion_id` dan tiga assert.
- **Kebocoran scaler:** scaler di-fit hanya pada data latih, di dalam
  pipeline sklearn.
- **Double normalization:** normalisasi ada di dalam model `.keras`, tidak
  diulang di aplikasi.
- **Tuning pada test:** ambang dan hyperparameter dikunci dari validasi;
  test dipakai sekali.
- **Perbandingan tidak adil:** setiap model dievaluasi pada ambangnya sendiri
  di kedua domain.

---

## BAB 6 — KESIMPULAN DAN SARAN

### 6.1 Kesimpulan

1. Empat model berhasil dilatih dan dibandingkan pada test set yang sama
   dengan metrik yang sesuai untuk data tidak seimbang.
2. MobileNetV2 tahap 1 adalah model terbaik dengan F1 macro 0,3993 dan
   recall ganas 0,4500 di data uji, serta paling tahan terhadap domain
   shift (recall 0,8730 di PAD-UFES-20).
3. Accuracy bukan kriteria yang tepat: Random Forest memiliki accuracy
   tertinggi (0,7132) tetapi recall ganas terendah (0,2600).
4. Class weight dan BatchNormalization adalah keharusan teknis, bukan
   pilihan estetik.
5. Fine-tuning pada dataset kecil justru menurunkan sensitivitas.
6. Aplikasi Streamlit berhasil dibangun dengan preprocessing identik
   dengan training dan disclaimer yang tidak bisa dilewatkan.

### 6.2 Saran Pengembangan Lanjutan

- Mengumpulkan data dengan kulit Fitzpatrick III-V agar model relevan
  untuk populasi Indonesia.
- Mencoba augmentasi smartphone yang lebih agresif dan fine-tuning dengan
  learning rate lebih kecil dengan validasi yang lebih besar.
- Menambahkan kalibrasi probabilitas yang lebih stabil dan evaluasi
  threshold berbasis biaya (FN lebih mahal dari FP).
- Melakukan uji klinis terbatas dengan dokter kulit sebelum penggunaan
  nyata.
- Menambahkan penjelasan model (mis. Grad-CAM) agar keputusan model bisa
  diaudit secara visual.

---

## DAFTAR PUSTAKA

1. Tschandl, P., Rosendahl, C., dan Kittler, H. (2018). The HAM10000
   dataset. *Scientific Data*, 5, 180214. DOI 10.1038/sdata.2018.161.
2. Pacheco, A. G. C. dkk. (2020). PAD-UFES-20. *Data in Brief*. DOI
   10.1016/j.dib.2020.106221.
3. Fogelberg, A. dkk. (2023). Domain shifts in dermoscopic skin cancer
   datasets. *New Biotechnology*. DOI 10.1016/j.nbt.2023.04.006.
4. Combalia, M. dkk. (2022). Validation of AI prediction models for skin
   cancer diagnosis. DOI terverifikasi di `docs/logs/verifikasi_literature.json`.
5. Almaraz-Damian, G. dkk. (2020). Melanoma and nevus classification using
   handcraft and deep learning.
6. Katsch, F. dkk. (2022). Comparison of CNN architectures for robustness
   against common corruptions. *Dermatology Practical & Conceptual*. DOI
   10.5826/dpc.1203a126.
7. Rotemberg, V. dkk. (2021). A patient-centric dataset of images and
   metadata for identifying melanomas. *Scientific Data*. DOI
   10.1038/s41597-021-00815-z.

Seluruh DOI dan judul diverifikasi otomatis ke PubMed.

---

## LAMPIRAN A — DAFTAR BUG YANG DITEMUKAN DAN DIPERBAIKI

| # | Bug | Dampak | Perbaikan |
|---|---|---|---|
| 1 | `tf.py_function` di `tf.data` | Deadlock, GPU 0% | `tf.io.encode_jpeg`/`decode_jpeg` |
| 2 | `lambda` di `FunctionTransformer` | Model tak bisa di-pickle | Kelas `HistogramScaler` |
| 3 | `np.argmax(val_loss) * -1` | Epoch terbaik salah | `np.argmin` |
| 4 | `model.predict_proba` di Keras 3 | `AttributeError` | `E.predict_with_proba` |
| 5 | `palambatte=` pada `sns.boxplot` | `TypeError` | `palette=` |
| 6 | `train_ds` vs `train_ds_smart` | `NameError` di Notebook 06 | Diselaraskan |
| 7 | Tanda gap overfitting terbalik | Overfit parah disebut "SEHAT" | `val - train` + gap relatif |
| 8 | `CalibratedClassifierCV(ensemble=False)` | Recall ganas 0,49 → 0,19 | `ensemble=True` |
| 9 | Vektor fitur 1D ke `predict` | `ValueError` | `.reshape(1, -1)` |
| 10 | Typo `gambar_uggahan`/`gambar_unggahan` | `NameError` di panel hasil | Diseragamkan |
| 11 | `make_notebooks.py` menimpa output | Output hilang | Skip kecuali `--force` |
| 12 | Satu ambang untuk semua model di uji eksternal | Perbandingan tidak adil | Ambang per-model |
| 13 | Band `is_uncertain` tetap 0,15 | Semua foto "tidak yakin" | Band mengikuti ambang |

---

## LAMPIRAN B — STATUS PROYEK DAN TINDAK LANJUT

### Sudah selesai

| Komponen | Status | Bukti |
|---|---|---|
| 8 notebook (01–08) | Dieksekusi, semua punya output | `notebooks/*.ipynb` |
| 4 model terlatih | Tersimpan | `models/` |
| Uji test set sekali | Selesai | `docs/logs/hasil_evaluasi_final.json` |
| Uji eksternal PAD-UFES-20 | Selesai | `docs/logs/hasil_uji_eksternal_pad_ufes.json` |
| Aplikasi Streamlit | Lulus AppTest | `scripts/test_app_content.py` |
| Laporan dan rekap | Final | `docs/REKAP_LAPORAN.md` |
| PPT versi 1 (14 slide) | Final | `docs/PPT_Laporan_Project.pptx` |
| PPT versi 2 (12 slide) | Final | `docs/PPT_Laporan_Project_v2.pptx` |
| Panduan penggunaan | Final | `docs/PANDUAN_PENGGUNAAN.md` |
| 7 sitasi PubMed | Terverifikasi | `docs/logs/verifikasi_literature.json` |
| 9 pemeriksaan otomatis | Semua lulus | `scripts/verify_final.py` |

### Masih perlu dilakukan mahasiswa (tidak bisa diwakilkan)

1. **Login Kaggle** dan menerima lisensi dataset HAM10000/PAD-UFES-20.
2. **Login GitHub** dan mengunggah repositori (data tidak di-commit).
3. **Rekam video demo** maksimal 7 menit memakai
   `docs/CHECKLIST_VIDEO_DEMO.md`.
4. **Finalisasi nama dosen pengampu** di `docs/info_kelompok.json` bila
   berbeda dari placeholder.

### Cara memverifikasi seluruh proyek

```bash
python scripts/verify_final.py
```

Perintah ini menjalankan sembilan pemeriksaan sekaligus dan membangun
ulang PPT serta rekap laporan dari file log, sehingga tidak ada angka
yang diketik manual.
