# Rekap Laporan Proyek

**Status per 17 November 2026, 22:20.** Seluruh angka di dokumen ini diambil otomatis dari `docs/logs/*.json`, yang ditulis notebook saat dijalankan. Tidak ada angka yang diketik manual.

---

## 1. Hasil Utama: Evaluasi di Data Uji

Test set berisi **1527 gambar** dan dipakai **SEKALI** di Notebook 07, setelah seluruh hyperparameter dan ambang batas terkunci dari data validasi.

| Model | Accuracy | F1 macro | Recall ganas | Specificity | FN | FP | Inferensi |
|---|---|---|---|---|---|---|---|
| MobileNetV2 | 0,6176 | 0,3993 | 0,4500 | 0,8720 | 39 | 550 | 59,3 ms |
| CNN (from scratch) | 0,5534 | 0,3634 | 0,5800 | 0,6862 | 28 | 635 | 9,6 ms |
| Random Forest (HOG) | 0,7132 | 0,3240 | 0,2600 | 0,9625 | 25 | 481 | 6,4 ms |
| SVM (HOG) | 0,6929 | 0,2642 | 0,2500 | 0,9470 | 42 | 559 | 10,2 ms |

**FN** = lesi ganas yang LOLOS (kesalahan paling berbahaya untuk skrining). **FP** = false alarm pada lesi tidak ganas.

### Ambang batas "ganas" (dihitung dari DATA VALIDASI)

Target recall **90%**. Ambang ini dikunci SEBELUM test disentuh, supaya test tetap bersih sebagai ukuran yang jujur.

| Model | Ambang |
|---|---|
| CNN (from scratch) | 0,12 |
| MobileNetV2 | 0,11 |
| Random Forest (HOG) | 0,20 |
| SVM (HOG) | 0,17 |

### Analisis overfitting (selisih F1 train vs test)

| Model | F1 train | F1 test | Selisih | Diagnosis |
|---|---|---|---|---|
| CNN (from scratch) | 0,3855 | 0,3634 | +0.0221 | sedikit overfitting |
| MobileNetV2 | 0,4278 | 0,3993 | +0.0285 | sedikit overfitting |
| Random Forest (HOG) | 1,0000 | 0,3240 | +0.6760 | OVERFITTING PARAH |
| SVM (HOG) | 0,3038 | 0,2642 | +0.0396 | sedikit overfitting |

**Model yang dipilih untuk aplikasi: MobileNetV2.**

Bobot kriteria: recall_ganas 35%, f1_macro 25%, domain_shift 20%, efisiensi 20%.

## 2. Hasil di Data Validasi

### Model klasik

| Model | F1 macro | Accuracy | Recall ganas |
|---|---|---|---|
| HOG-SVM | 0,2826 | 0,7037 | 0,2917 |
| HOG-Random Forest | 0,3491 | 0,7109 | 0,1987 |

### CNN dari nol (tiga varian, perbandingan yang adil)

| Varian | Parameter | F1 macro | Accuracy | Recall ganas | Kelas terpakai |
|---|---|---|---|---|---|
| CNN spesifik + GAP | 20,231 | 0,3067 | 0,5546 | 0,1795 | 7/7 |
| CNN lebar + GAP | 374,407 | 0,0960 | 0,4088 | 0,4776 | 3/7 |
| CNN lebar + Flatten | 2,207,623 | 0,3420 | 0,5298 | 0,6346 | 7/7 |

Pemenang: **CNN lebar + Flatten** (F1 macro 0,3420, recall ganas 0,6346).

Diagnosis overfitting: **selisih jauh. Model menghafal data latih.**

### MobileNetV2 (dua tahap, dinilai di data validasi yang sama)

| Tahap | F1 macro | Accuracy | Recall ganas | FN ganas |
|---|---|---|---|---|
| 1 (classifier, backbone beku) | 0,3934 | 0,6122 | 0,5545 | 139 |
| 2 (fine-tuning 30 layer atas) | 0,3938 | 0,6841 | 0,3750 | 195 |

Kedua tahap dinilai di data validasi yang sama dengan augmentasi training yang sama, jadi perbandingannya adil: satu-satunya yang beda adalah apakah backbone-nya dibekukan dan learning rate-nya (1e-3 vs 1e-5).

**DIPILIH: tahap 1.** Fine-tuning justru menurunkan sensitivitas, dan ini temuan yang layak dibahas: pada dataset sekecil ini, membuka lapisan pretrained memperburuk generalisasi.

## 3. Uji Eksternal: Domain Shift ke Foto Smartphone

**PAD-UFES-20** (DOI 10.17632/zr7vgbcyr2, lisensi CC BY 4.0): foto smartphone, tanpa training ulang, ambang tetap dari validasi.

| Model | Rec ganas HAM | Rec ganas PAD | Selisih | Spec HAM | Spec PAD | FN HAM | FN PAD |
|---|---|---|---|---|---|---|---|---|
| CNN (from scratch) | 0,5800 | 0,9698 | +0.3898 | 0,6862 | 0,0104 | 28 | 55 |
| MobileNetV2 | 0,4500 | 0,8730 | +0.4230 | 0,8720 | 0,4489 | 39 | 231 |
| SVM (HOG) | 0,2500 | 0,8686 | +0.6186 | 0,9470 | 0,2589 | 42 | 239 |
| Random Forest (HOG) | 0,2600 | 0,8362 | +0.5762 | 0,9625 | 0,2860 | 25 | 298 |

**Cara membaca tabel ini:** recall ganas yang TINGGI di PAD-UFES-20 tidak otomatis berarti model lebih baik. Lihat kolom specificity: di dataset eksternal, specificity semua model turun drastis (0,01 sampai 0,45). Artinya model mendeteksi hampir semua lesi ganas **karena ia menandai hampir semua foto sebagai ganas** -- bukan karena lebih teliti. Pada aplikasi skrining, trade-off ini harus dibuat eksplisit, bukan disembunyikan di balik angka recall yang menarik.

### Recall per diagnosis di PAD-UFES-20

| Diagnosis | n | Recall |
|---|---|---|
| ACK | 730 | 0,8781 |
| BCC | 845 | 0,8923 |
| MEL | 52 | 0,6346 |
| NEV | 244 | 0,5697 |
| SCC | 192 | 0,8333 |
| SEK | 235 | 0,3234 |

## 4. Temuan Eksperimen (intis kontribusi laporan)

**A. Random Forest pada fitur mentah hampir tidak belajar.** F1 macro 0,1467 vs baseline 0,1148, recall ganas 0,0192. Penyebab: RF memilih fitur acak saat split, jadi peluang fitur histogram (48 dari 1.812 = 2,65%) praktis tak pernah dipakai.

**B. Ukuran sel HOG ternyata hampir tidak berpengaruh.** 8x8 -> F1 0,2163; 16x16 -> F1 0,2118; 32x32 -> F1 0,2175. Yang dipakai 16x16 karena 4,6x lebih hemat dimensi (1.764 vs 8.100) dengan performa setara.

**C. Cara SVM mengambil probabilitas sangat memengaruhi performa.**

| Cara | F1 macro | Recall ganas |
|---|---|---|
| SVC polos (acuan) | 0,3437 | 0,4904 |
| SVC(probability=True) | 0,2957 | 0,3365 |
| CalibratedClassifierCV(ensemble=False) | 0,2151 | 0,1891 |
| CalibratedClassifierCV(ensemble=True) | 0,2713 | 0,2981 |

Konsekuensi: dipakai `CalibratedClassifierCV(ensemble=True)`, karena API-nya yang resmi disarankan library DAN performanya jauh lebih baik daripada `ensemble=False` yang ada di contoh resmi sklearn.

**D. Bobot histogram hanya berpengaruh pada SVM, bukan pada Random Forest.** Pohon membelah berdasarkan urutan nilai, bukan jarak, jadi mengalikan fitur dengan konstanta tidak mengubah partisi.

## 5. Dokumentasi dan Artefak

| Artefak | Isi |
|---|---|
| `README.md` | Struktur repo, aturan metodologi, cara jalankan |
| `docs/CATATAN_TEMUAN_DIAGNOSTIK.md` | Semua eksperimen + 11 bug yang ditemukan lewat eksekusi |
| `docs/PERSIAPAN_TANYA_JAWAB.md` | 40+ soal potentiall beserta bukti file |
| `docs/PANDUAN_DEPLOY.md` | Panduan deploy aplikasi |
| `docs/CHECKLIST_VIDEO_DEMO.md` | Checklist rekaman video demo |
| `docs/PPT_Laporan_Project.pptx` | Slide presentasi (14 slide) |

**Literatur:** 7 artikel diverifikasi otomatis ke PubMed (PMID + DOI + judul cocok), lihat `docs/logs/verifikasi_literature.json`.

---

## 6. Pemeriksaan Otomatis

Empat skrip memeriksa repo sebelum submit. Semuanya harus lulus:

| Skrip | Yang diperiksa |
|---|---|
| `scripts/check_api_calls.py` | Argumen fungsi yang tidak ada di API library terpasang |
| `scripts/check_undefined_names.py` | Variabel yang dipakai sebelum didefinisikan |
| `scripts/check_indonesian_text.py` | Kalimat campuran Inggris / teks rusak |
| `scripts/sanity_check_text.py` | Karakter CJK dan BOM |

Jalankan: `python scripts/check_api_calls.py`, lalu yang lainnya.

Dokumen ini dibuat ulang oleh `python scripts/build_rekap.py`.
