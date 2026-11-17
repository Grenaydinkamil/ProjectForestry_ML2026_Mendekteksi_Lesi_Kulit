# Persiapan Tanya Jawab Individu

Dokumen ini adalah bahan latihan mandiri sebelum sesi tanya jawab. Setiap
pertanyaan di bawah **harus bisa dijawab tanpa membuka catatan**, dan
jawabannya harus bisa dipertanggungjawabkan ke kode.

Format: **Pertanyaan** -> jawaban dalam 2-4 kalimat -> **Bukti di kode**
(nama file + nama fungsi).

---

## A. Konsep Machine Learning

### A1. Apa itu supervised learning, dan bagaimana project ini masuk kategori itu?

Supervised learning adalah paradigm di mana model belajar dari data yang
sudah berlabel: input dan output-nya diketahui. Di project ini, setiap foto
lesi kulit dipasangkan dengan label diagnosisnya (`dx` dalam
`HAM10000_metadata.csv`), lalu model belajar memetakan citra ke salah satu
dari 7 kelas diagnosis.

Ini berbeda dari unsupervised learning yang mencari pola tanpa label, dan
dari reinforcement learning yang belajar dari reward.

**Bukti:** `src/data_loader.py` (fungsi `load_metadata`),
`notebooks/02_split.ipynb`.

### A2. Kenapa dataset ini di-split, dan apa itu data leakage?

Split membagi data jadi train, validasi, dan test. Tujuannya: model
belajar di train, pilihan hyperparameter diambil di validasi, dan test
dipakai **sekali** di akhir sebagai pengukuran jujur.

Data leakage adalah ketika informasi dari test set "bocor" ke proses
training, sehingga angka test jadi terlalu optimistis dan tidak lagi
mengukur kemampuan generalisasi.

**Bukti:** `notebooks/02_split.ipynb` bagian 1 dan 3.

### A3. Kenapa accuracy bukan metrik utama di project ini?

Karena datanya sangat tidak seimbang: kelas `nv` (mola) adalah 66,9% data,
kelas `df` (dermatofibroma) cuma 1,1%. Model yang selalu menebak "nv"
mendapat accuracy 67% tanpa belajar apa pun.

Untuk aplikasi skrining, kesalahan yang paling berbahaya adalah melanoma
lolos diklasifikasi jinak (false negative). Jadi metrik utama adalah
**F1 macro** (menghukum kelas yang diabaikan) dan **recall kelas ganas**.

**Bukti:** `notebooks/01_eda.ipynb` bagian 3, `notebooks/07_evaluasi_perbandingan.ipynb`.

### A4. Apa itu class weight, dan kenapa dipakai di project ini?

Class weight adalah bobot per kelas di fungsi loss, supaya kesalahan pada
kelas minoritas dihukum lebih berat. Dihitung dari frekuensi kelas di data
train saja.

Tanpa class weight, CNN mengunci accuracy di 0,672 (persis proporsi `nv`)
dengan recall ganas nol: model tidak belajar apa pun. Dengan class weight,
recall ganas naik ke 0,93, dengan trade-off specificity yang turun.

**Bukti:** `notebooks/02_split.ipynb` bagian 6, `src/modeling.py` (parameter `class_weight`).

---

## B. Dataset

### B1. Ceritakan dataset HAM10000.

Dataset dermatologi berisi 10.015 foto dermoskopi (foto lewat mikroskop
khusus) dari 7.470 lesi kulit, dengan 7 kelas diagnosis. Dikumpulkan dari
Albania, Australia, dan Austria. Dipublikasikan Tschandl dkk. (2018) di
*Scientific Data*.

Frasa "Human Against Machine with 10.000 training images" merujuk pada
tantangan awal: dataset dermatologi yang cukup besar dan beragam untuk
melatih jaringan saraf.

**Bukti:** `docs/CATATAN_TEMUAN_DIAGNOSTIK.md`, `scripts/literature.py`.

### B2. Kenapa split harus per `lesion_id`, bukan per `image_id`?

Karena 10.015 foto itu sebenarnya cuma 7.470 lesi: **2.545 foto adalah lesi
yang sama** yang difoto dari sudut berbeda. Kalau di-split per foto, foto
lesi yang sama bisa masuk train DAN test. Model lalu "menghafal" lesi itu,
dan angka test jadi tidak mengukur kemampuan generalisasi.

Jadi `GroupShuffleSplit` dengan `groups=lesion_id` dipakai, dan hasilnya
diverifikasi dengan `assert` bahwa tidak ada `lesion_id` yang overlap.

**Bukti:** `notebooks/02_split.ipynb` bagian 2 dan 3 (tiga blok `assert`).

### B3. Kenapa `age` tidak diimputasi ada 57 nilai kosong?

Karena tiga alasan: nilainya hanya 0,57% dari data, `age` **tidak dipakai
sebagai feature model** (model hanya memproses gambar), dan mengimputasi
nilai yang tak dipakai hanya menambah pekerjaan tanpa manfaat.

Ini keputusan sadar, bukan kelalaian, dan bisa dipertanggungjawabkan.

**Bukti:** `notebooks/01_eda.ipynb` bagian 5.

### B4. Apa keterbatasan demografis paling serius dari dataset ini?

Seluruh foto berasal dari pigmentasi Fitzpatrick I-II (kulit putih).
**Tidak ada satu pun gambar kulit gelap.** Padahal populasi sasaran di
Indonesia didominasi Fitzpatrick III-V.

Akibatnya: model mungkin tidak bisa mengenali lesi pada kulit gelap, dan
k groupes.application ini mungkin salah memberi reassurance pada pengguna
kulit gelap. Ini harus dinyatakan terbuka, bukan disembunyikan.

**Bukti:** `notebooks/01_eda.ipynb` bagian 5 dan 6.

---

## C. Preprocessing dan Rekayasa Fitur

### C1. Apa itu HOG dan histogram warna, dan kenapa keduanya dipakai?

**HOG** (Histogram of Oriented Gradients) mengukur arah dan kekuatan
garis di setiap area gambar. Analoginya: menyinari foto dari samping,
bayangan di tepi lesi menunjukkan arah garis. Lesi dengan tepi tidak rata
(melanoma) punya pola garis yang tidak teratur, sedangkan mola punya pola
radial yang simetris.

**Histogram warna** menghitung berapa persen area gambar pada tiap rentang
warna per kanal (R, G, B). Jadi HOG menjawab "pola bentuk", histogram
menjawab "pola sebaran warna".

Keduanya dipakai karena keduanya menangkap sisi berbeda dari pembeda
lesi, dan eksperimen menunjukkan gabungan keduanya lebih baik daripada
salah satu saja.

**Bukti:** `src/features.py`, `notebooks/03_fitur_klasik.ipynb` bagian 1.

### C2. Temuan paling menarik dari eksperimen fitur Anda.

Random Forest pada 8.148 fiturMENTAH: F1 macro 0,115, **persis sama
dengan baseline "selalu menebak nv"**, dengan recall ganas nol.

Penyebabnya: RF memilih fitur **secara acak** saat membelah node. Dengan
8.100 fitur HOG yang saling berkorelasi dan hanya 48 fitur histogram yang
diskriminatif, peluang RF memakai fitur histogram saat satu split hanya
48/8.148 = 0,59%. Dari ribuan split, fitur paling informatif praktis
tidak pernah dipakai.

Perbaikannya: PCA hanya pada bagian HOG (bukan seluruh fitur), histogram
dibiarkan utuh dengan bobot eksplisit.

**Bukti:** `scripts/diagnostik_fitur.py`, `docs/CATATAN_TEMUAN_DIAGNOSTIK.md` bagian 1-3.

### C3. Kenapa ukuran sel HOG 16x16, bukan 8x8 atau 32x32?

Saya mengukurnya dengan model yang sama untuk ketiga ukuran sel
(`scripts/diagnostik_hog_cell.py`):

| Ukuran sel | Dimensi HOG | F1 macro | Accuracy | Recall ganas |
|---|---|---|---|---|
| 8x8 | 8.100 | 0,2163 | 0,6900 | 0,2179 |
| **16x16** | **1.764** | 0,2118 | 0,6880 | 0,1859 |
| 32x32 | 324 | 0,2175 | 0,6913 | 0,1667 |

**Jawaban jujurnya: ukuran sel hampir tidak berpengaruh pada performa
akhir.** Selisih F1 macro antar ketiganya cuma 0,0057, jauh
lebih kecil daripada selisih antar model (0,21 sampai 0,42).

Jadi alasan saya memilih 16x16 **bukan** karena skornya paling tinggi,
tapi karena **hemat tanpa mengorbankan performa**:

1. 1.764 dimensi adalah **4,6x lebih kecil** dari 8.100 dimensi, sehingga
   ekstraksi fitur, PCA, dan training jauh lebih cepat.
2. Sel 8x8 menghasilkan banyak dimensi yang saling berkorelasi (noise),
   sedangkan sel 32x32 menghilangkan bentuk dan tepi lesi yang justru
   penting untuk membedakan melanoma dari mola.
3. Karena performanya setara, saya pilih yang paling murah.

**Bukti:** `src/config.py` (HOG_PIXELS_PER_CELL + komentar),
`scripts/diagnostik_hog_cell.py`,
`docs/logs/diagnostik_hog_cell.json`.

### C4. Kenapa `CalibratedClassifierCV` dan bukan `SVC(probability=True)`?

Di scikit-learn 1.9 parameter `probability=True` pada SVC sudah
**deprecated**. Peringatan resmi library:
"Use `CalibratedClassifierCV(SVC(), ensemble=False)` instead of
`SVC(probability=True)`".

`CalibratedClassifierCV` melakukan hal yang sama (kalibrasi Platt dengan
5-fold CV internal) lewat API yang stabil ke depan.

**Dan `ensemble=True` tidak asal dipilih.** Saya mengukur keempat cara
di data validasi (`scripts/cek_kalibrasi_svm.py`):

| Cara | F1 macro | Recall ganas |
|---|---|---|
| SVC polos (tanpa probabilitas, hanya acuan) | 0,3437 | 0,4904 |
| `SVC(probability=True)` | 0,2957 | 0,3365 |
| `CalibratedClassifierCV(ensemble=False)` | 0,2151 | 0,1891 |
| **`CalibratedClassifierCV(ensemble=True)`** | **0,2713** | **0,2981** |

Kenapa `ensemble=False` (contoh resmi sklearn) justru lebih buruk?
Dengan `ensemble=False`, kalibrator dihitung dari prediksi out-of-fold,
lalu estimator dasarnya **dilatih ulang** pada seluruh data latih.
Decision function model yang dilatih ulang bisa punya skala berbeda dari
model lipatan, sehingga sigmoid kalibrator tidak lagi seimbang antar
kelas dan argmax probabilitas berubah. Efeknya besar: recall ganas
turun dari 0,4904 ke 0,1891.

Dengan `ensemble=True`, prediksi adalah rata-rata probabilitas dari 5
model lipatan, yang skalanya konsisten dengan kalibrator.

Jadi alasan saya memilih `ensemble=True`: (1) API-nya yang resmi
disarankan library, bukan yang deprecated; (2) performanya jauh lebih
baik; (3) kita **mengukur** sendiri, bukan memakai nilai bawaan.

**Bukti:** `src/classic_models.py` (fungsi `build_svm`),
`scripts/cek_kalibrasi_svm.py`,
`docs/logs/cek_kalibrasi_svm.json`.

### C5. Kenapa normalisasi tidak boleh dua kali?

Karena MobileNetV2 dilatih dengan `preprocess_input` yang mengubah nilai
0..255 menjadi -1..1. Kalau kita menambah `Rescaling(1/255)` di atasnya,
input model berada di rentang yang **tidak pernah dilihat** saat
pretraining, dan performanya anjlok tanpa error yang terlihat.

Karena itu normalisasi di project ini **ditempatkan di dalam model**
(layer `Rescaling`), sehingga:
- aturannya tidak bisa "lupa" (ikut tersimpan di file .keras)
- file .keras bisa dimuat tanpa `custom_objects`, penting untuk deploy

**Bukti:** `src/modeling.py` (fungsi `build_mobilenetv2_stage1`).

### C6. Kenapa augmentasi hanya di data latih?

Augmentasi menciptakan variasi dari gambar training (rotasi, zoom, flip,
perubahan pencahayaan). Terapkan ke validation atau test, kita mengukur
performa pada data yang **tidak realistis** dan angkanya jadi tidak jujur.

Karena itu di kode, augmentasi ada di `make_tf_dataset` dengan parameter
`training=True`, yang hanya bernilai True untuk split latih.

**Bukti:** `src/preprocessing.py`.

---

## D. Algoritma

### D1. Kenapa memilih 4 model ini?

Keempatnya mewakili dua keluarga yang berbeda, sehingga perbandingannya
bernada:

- **HOG-SVM** dan **HOG-Random Forest**: model klasik berbasis fitur
  buatan tangan. Cepat, bisa dijelaskan sepenuhnya, tapi dibatasi oleh
  kualitas fitur.
- **CNN dari nol**: deep learning dengan representasi yang dipelajari dari
  data sendiri. jumpedshows jika arsitektur dan augmentasinya tepat.
- **MobileNetV2**: transfer learning. Bobotnya sudah dilatih pada 1,4
  juta gambar ImageNet, jadi hanya classifier yang perlu disesuaikan.

**Bukti:** `notebooks/04_svm_rf.ipynb`, `notebooks/05_cnn.ipynb`, `notebooks/06_mobilenetv2.ipynb`.

### D2. Apa itu transfer learning dan kenapa MobileNetV2?

Transfer learning memakai bobot yang sudah dilatih pada dataset besar
untuk dataset yang lebih kecil dan mirip.

MobileNetV2 sudah dilatih pada 1,4 juta gambar ImageNet, jadi sudah "tahu"
cara mengenali garis, tepi, dan tekstur. Kita hanya perlu menyesuaikan
**kepala classifier**-nya ke lesi kulit.

Karena itu training dilakukan **dua tahap**: tahap 1 classifier saja
(backbone dibekukan, cepat dan stabil), tahap 2 membuka 30 lapisan teratas
dengan learning rate sangat kecil (1e-5).

**Bukti:** `notebooks/06_mobilenetv2.ipynb` bagian 1 dan 5.

### D3. Kenapa BatchNormalization dikunci saat fine-tuning?

BatchNormalization punya dua mode. Dalam mode training, ia menghitung
statistik dari batch saat ini dan **mengubah** moving mean/variance setiap
epoch. Dalam mode inference, ia memakai statistik yang tersimpan dan tidak
mengubah apa pun.

Kalau BN ikut di-training: batch size kita cuma 32 dan distribusinya
berbeda dari ImageNet, jadi statistiknya jadi bising dan prediksi bisa
**berayak** antar epoch. Mengunci BN mempertahankan statistik yang stabil.

**Bukti:** `src/modeling.py` (fungsi `unfreeze_top_layers`), termasuk
`assert` yang memverifikasi tidak ada BN yang trainable.

### D4. Kenapa learning rate tahap 2 jauh lebih kecil (1e-5)?

Bobot backbone hasil latihan 1,4 juta gambar itu sangat berharga. Dengan
learning rate besar, Adam akan membuat langkah besar yang **menghancurkan**
fitur itu (catastrophic forgetting).

Tahap 1 memakai lr 1e-3 karena classifier-nya baru dan perlu bergerak
cepat. Tahap 2 memakai 1e-5 karena kita hanya ingin menyempurnakan.

**Bukti:** `notebooks/06_mobilenetv2.ipynb` bagian 5.

### D5. Apa itu GlobalAveragePooling dan kenapa Anda membandingkannya dengan Flatten?

GAP mengambil rata-rata setiap peta fitur, jadi mengubah (32, 32, 64)
menjadi 64 angka. Flatten mengubahnya jadi 65.536 angka.

Dengan GAP, Dense pertama hanya punya 64 x 64 = 4.160 parameter. Dengan
Flatten, 65.536 x 64 = 4,2 juta parameter, pada dataset cuma 6.959 gambar.

Jadi perbandingannya bukan cuma "mana yang lebih akurat", tapi "mana yang
lebih masuk akal untuk dataset sekecil ini".

**Bukti:** `notebooks/05_cnn.ipynb` bagian 4 dan 8.

### D6. Kenapa `PCA` dipakai untuk SVM?

Dua alasan yang saling strengthening:
1. **Waktu**: SVM RBF menghitung jarak antar semua pasangan data, dan
   tiap jarak involve semua dimensi. Dari 8.148 fitur ke 78 fitur membuat
   training 7x lebih cepat.
2. **Overfitting**: 8.148 fitur bebas untuk 6.959 data sangat prone overfit.

**Bukti:** `src/classic_models.py` (fungsi `build_preprocessor`).

---

## E. Metrik dan Evaluasi

### E1. Metrik apa yang Anda pakai dan kenapa?

- **F1 macro** (utama): rata-rata harmonic mean precision & recall untuk
  semua kelas. Menghukum kelas yang diabaikan, berbeda dengan accuracy.
- **Recall kelas ganas** (kritis): dari semua lesi ganas, berapa persen
  yang terdeteksi. Untuk skrining, ini yang paling penting.
- **Precision, specificity, confusion matrix** (pendukung): melihat
  kesalahan per kelas dan ke mana kesalahan itu pergi.
- **AUC** (opsional): kemampuan memisahkan kelas secara global.

**Bukti:** `src/evaluation.py` (fungsi `evaluate_model`).

### E2. Bagaimana Anda menentukan model terbaik?

Bukan dari satu angka, tapi dari empat kriteria berbobot:
recall kelas ganas (35%), F1 macro (25%), kesiapan domain shift (20%), dan
kecepatan serta ukuran model (20%).

Kriteria domain shift dan efisiensi mengandung penilaian yang partly
subjektif, jadi kami nyatakan terbuka di laporan, dan diperkuat dengan
hasil uji eksternal di Fase 7.

**Bukti:** `notebooks/07_evaluasi_perbandingan.ipynb` bagian 13.

### E3. Bagaimana cara menentukan ambang "ganas"?

Dari **data validasi**, dengan target recall 90%. Kita mencari ambang
terkecil yang mencapai target itu, lalu ambil yang specificity terbaik di
sekitarnya.

Threshold **tidak diubah** setelah melihat hasil test, karena kalau
threshold-nya "di-tuning" dengan data test, hasil test tidak lagi jadi
pengukuran yang jujur.

**Bukti:** `src/evaluation.py` (fungsi `tune_binary_threshold`),
`notebooks/07_evaluasi_perbandingan.ipynb` bagian 6.

### E4. Kenapa test set hanya dipakai sekali?

Karena test set adalah "ujian". Kalau kita melihat hasilnya dan lalu
mengubah apa pun (threshold, hyperparameter, pilihan model), kita sudah
"membocorkan" soal ujian ke proses belajar, dan angkanya jadi tidak lagi
cerminan kemampuan model di dunia nyata.

Di project ini, semua keputusan diambil di validasi, dan test baru
disentuh di Notebook 07.

**Bukti:** `notebooks/07_evaluasi_perbandingan.ipynb` bagian 3.

### E5. Bagaimana Anda mendiagnosis overfitting?

Dari selisih performa train dan test, dan dari kurva training:
- train turun, validation naik setelah epoch tertentu -> overfitting
- train dan validation turun bersama dan masih turun -> underfitting
- train loss mendekati 0 sementara validation jauh lebih tinggi ->
  overfitting parah (model menghafal)

EarlyStopping dengan `restore_best_weights=True` dipakai supaya model
yang disimpan adalah yang terbaik, bukan yang terakhir.

**Bukti:** `src/evaluation.py` (fungsi `plot_training_curves`),
`notebooks/05_cnn.ipynb` bagian 6.

---

## F. Kode dan Reproduksi

### F1. Bagaimana cara mereproduksi hasil Anda?

Semua angka berasal dari kode yang dijalankan, bukan dari atur.
Langkah-langkahnya ada di README, dan `scripts/check_environment.py`
memverifikasi kesiapan lingkungan.

Semua random seed ditetapkan ke 42 di satu tempat (`src/config.py`), dan
split disimpan ke CSV supaya notebook berikutnya memakai partisi yang
sama persis.

**Bukti:** `src/config.py` (SEED), `notebooks/02_split.ipynb` bagian 7.

### F2. Kenapa normalisasi dan scalerplaced di dalam pipeline?

Supaya tidak mungkin ada kebocoran informasi. Kalau scaler di-fit manual,
sangat mungkin kode secara tidak sengaja memanggil `fit_transform` pada data
validasi. `Pipeline` sklearn menjamin `fit` hanya dipanggil pada data
training.

**Bukti:** `src/classic_models.py`.

### F3. Apa yang terjadi kalau di-running di laptop tanpa GPU?

Bisa. Hasilnya identik secara metodologis, hanya lebih lambat. CNN dan MobileNetV2 butuh jauh lebih lama di CPU.

Ada catatan penting untuk RTX 50 series: TensorFlow 2.21 dibangun dengan
CUDA 12.5 yang tidak punya binary kernel untuk sm_120 (Blackwell), jadi
kernel di-JIT dari PTX. **Run pertama** sangat lambat (10-30 menit),
setelah itu di-cache dan kecepatannya normal. Itu normal, bukan error.

**Bukti:** `README.md` bagian "Strategi GPU", `docs/CATATAN_TEMUAN_DIAGNOSTIK.md` bagian 6.

### F4. Apa saja batasan project ini?

- Dataset: hanya Fitzpatrick I-II, tanpa kulit gelap
- Model: belum divalidasi klinis, hanya diuji pada dataset
- Kelas minoritas (df, vasc) selalu jadi masalah
- Aplikasi: performa pada foto smartphone lebih rendah
- Belum ada validasi oleh dokter

**Bukti:** `notebooks/08_uji_eksternal_pad_ufes.ipynb` bagian 13-14,
`docs/PANDUAN_DEPLOY.md`.

---

## G. Aplikasi dan Etika

### G1. Kenapa aplikasi menampilkan disclaimer besar?

Karena model ini **belum divalidasi klinis** dan bisa salah. Aplikasi
skrining yang menyesatkan pengguna adalah bahaya, bukan bantuan.

Yang ditampilkan: alat bantu belajar, bukan diagnosis; "tidak ganas"
bukan berarti aman; segera ke dokter bila lesi berubah bentuk, warna,
ukuran, berdarah, atau terasa tidak nyaman; model dilatih pada data
dermoskopi dengan populasi terbatas dan tanpa cakupan kulit gelap.

**Bukti:** `app/app.py` (fungsi `render_disclaimer`).

### G2. Bagaimana privasi foto pengguna dijaga?

Foto diproses di memori selama sesi saja. Tidak disimpan di server, tidak
dicatat, dan tidak dikirim ke pihak ketiga. Statistik usage Streamlit juga
dimatikan di `.streamlit/config.toml`.

**Bukti:** `app/app.py` (sidebar privasi), `.streamlit/config.toml`.

### G3. Bagaimana memastikan preprocessing di aplikasi sama dengan training?

Aturan preprocessing disimpan di `models/preprocess_config.json`, yang
ditulis **otomatis** oleh notebook training (bukan diketik manual).
Aplikasi membacanya. Normalisasi sendiri berada di dalam model, jadi
ikut tersimpan di file .keras dan mustahil lupa.

**Bukti:** `src/predictor.py`, `src/modeling.py` (fungsi `save_model`).

### G4. Kenapa ambang "ganas" di aplikasi berasal dari validasi?

Karena ambang yang disetel dari test set akan membuat hasil test tidak
lagi jujur. Ambang dari validasi adalah pilihan yang bisa dipertanggungjawabkan
dan bisa dijelaskan: "Kami ingin minimal 90% lesi ganas terdeteksi, lalu
meminimalkan false alarm".

**Bukti:** `src/evaluation.py` (fungsi `tune_binary_threshold`).

### G5. Apa yang terjadi jika model tidak tersedia saat aplikasi dibuka?

Aplikasi menampilkan pesan error yang jelas, bukan crash. Ada tiga kondisi
tampilan yang lengkap: kosong (petunjuk sebelum upload), sedang memproses
(spinner), dan gagal (pesan ramah plus detail teknis).

**Bukti:** `app/app.py` (blok `try/except`).

---

## H. Pertanyaan yang Sulit (siapkan diri)

### H1. Kenapa model klasik Andaperformanya rendah?

Karena fitur handcraft punya batas. HOG menangkap arah garis dan histogram
menangkap sebaran warna, tapi melanoma versus mola sering hanya berbeda
pada simetri dan gradasi warna halus yang tidak tertangkap fitur
tersebut.

Justru inilah temuan yang paling bernilai: Random Forest pada 8.148 fitur
tidak belajar apa pun sama sekali, yang membuktikan bahwazah Historically
memilih fitur klasik tanpa pengukuran adalah pilihan yang berisiko.

**Bukti:** `docs/CATATAN_TEMUAN_DIAGNOSTIK.md`.

### H2. Kenapa tidak pakai model yang lebih akurat saja, tanpa model klasik?

Assignment ini secara eksplisit meminta perbandingan minimal tiga
algoritma dengan alasan yang jelas. Model klasik adalah pembanding yang
menjawab pertanyaan "apakah feature engineering cukup, atau kita benar-benar
perlu deep learning".

Ternyata jawabannya: **keduanya perlu**, dan perbandingan itu yang
memberikan penjelasan, bukan sekadar angka.

### H3. Bagaimana Anda tahu model Anda tidak overfitting ke data test?

  Test set tidak pernah dipakai untuk keputusan apa pun: bukan untuk
memilih hyperparameter, bukan untuk menyetel threshold, bukan untuk
memilih model. Semua itu dilakukan di validasi.

Dan split dilakukan per `lesion_id`, dengan `assert` yang membuktikan tidak
ada lesi yang muncul di dua subset. Tanpa itu, overfitting ke lesi yang
sama bisa terjadi dan kita tidak akan sadari.

**Bukti:** `notebooks/02_split.ipynb` bagian 3, `notebooks/07_evaluasi_perbandingan.ipynb` bagian 3.

### H4. Kalau diberi waktu lebih, apa yang akan Anda perbaiki?

1. Fine-tuning dengan augmentasi smartphone yang lebih agresif (Fase 8 baru
   menunjukkan penurunan domain shift yang terukur)
2. Validasi pada data kulit gelap Fitzpatrick IV-VI
3. Feature engineering khusus untuk `df` dan `vasc` yang selalu gagal
4. Threshold adaptif per tingkat confidence, bukan satu ambang global
5. Validasi klinik bersama dokter kulit

**Bukti:** `notebooks/08_uji_eksternal_pad_ufes.ipynb` bagian 13-14.

### H5. Siapa yang mengerjakan bagian ini, dan apakah Anda paham semuanya?

Pembagian tugas ada di README dan PPT. Karena penilaiannya individu,
setiap anggota wajib memahami keseluruhan proyek, bukan hanya bagiannya.
Saya sudah mendokumentasikan keputusan dan alasannya di setiap notebook
supaya bisa ditelusuri, dan semua eksperimen yang menghasilkan angka
tercantum di `docs/CATATAN_TEMUAN_DIAGNOSTIK.md` beserta skripnya.

---

## Cara Berlatih

1. Bacalah pertanyaan, lalu **jawab dengan lisan** tanpa membuka file.
2. Kalau bingung, buka **hanya** file yang disebut di bagian "Bukti".
3. Kalau ditanya "kenapa" untuk angka yang muncul, jawab dengan eksperimen
   yang menundukkan (bukan dengan ="sepeda umumnya").
4. Kalau angkanya tidak bisa dijelaskan, katakan terus terang. Lebih baik
   jujur daripada mengarang alasan.
