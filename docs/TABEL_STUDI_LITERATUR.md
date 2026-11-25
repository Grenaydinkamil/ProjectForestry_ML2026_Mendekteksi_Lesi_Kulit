# Tabel studi literatur: tujuh artikel jurnal

Laporan kami bertumpu pada tujuh artikel jurnal yang dipilih karena menjawab
pertanyaan yang sama: bagaimana cara memprediksi diagnosis lesi kulit dari
citra, dan seberapa jauh hasilnya bertahan ketika fotoya diambil dengan
smartphone, bukan dermatoskop.

Metadata setiap artikel (judul, penulis, tahun, jurnal, DOI) diverifikasi
otomatis terhadap NCBI PubMed E-utilities, bukan diketik dari ingatan. Hasil
verifikasi tersimpan di `docs/logs/verifikasi_literature.json` dan dapat
diulang dengan `python scripts/verify_literature.py`.

| No | Penulis & Tahun | Permasalahan | Dataset | Metode/Algoritma | Metrik Evaluasi | Hasil Utama | Perbedaan dengan Project | DOI |
|---|---|---|---|---|---|---|---|---|
| R1 | Tschandl, P.; Rosendahl, C.; Kittler, H. (2018) | Dataset dermoskopi yang tersedia untuk melatih jaringan saraf masih sedikit dan kurang beragam. | HAM10000: 10.015 citra dermoskopi multi-sumber dari 7.470 lesi, 7 kelas. | Artikel deskriptif data (data descriptor), bukan model ML. | Tidak ada metrik model; fokus pada kelengkapan data, distribusi kelas, dan konsistensi label. | Menyediakan benchmark standar klasifikasi lesi pigmented. 67% data berkelas nv (mola), jadi penilaian harus memperhitungkan ketidakseimbangan. | Kami memakai HAM10000 sebagai data latih dengan pembagian per `lesion_id` (GroupShuffleSplit dua tahap, seed 42), bukan per gambar, dan tidak memakai metadata klinis yang baru tersedia setelah diagnosis. | https://doi.org/10.1038/sdata.2018.161 |
| R2 | Pacheco, A. G. C.; Lima, G. R.; Salomão, A. S.; dkk. (2020) | Sebagian besar sistem diagnosis kulit hanya bekerja pada citra dermoskopi, sehingga minim arsip citra klinis publik untuk mengevaluasi sistem tersebut. | PAD-UFES-20: 2.298 citra klinis dari 1.641 lesi pada 1.373 pasien, 6 diagnosis, foto smartphone, Fitzpatrick I-VI. | Artikel deskriptif data (data descriptor), bukan model ML. | Tidak ada metrik model; fokus pada kelengkapan data klinis, distribusi diagnosis, dan variasi Fitzpatrick. | Menyediakan benchmark foto smartphone untuk lesi pigmented beserta metadata klinis, dan memuat jenis lesi yang tidak pernah ada di dataset dermoskopi. | Kami memakai PAD-UFES-20 hanya sebagai uji eksternal: model tidak dilatih ulang dan ambang batas tidak diubah dari hasil validasi HAM10000. | https://doi.org/10.1016/j.dib.2020.106221 |
| R3 | Fogelberg, K.; Chamarthi, S.; Maron, R. C.; dkk. (2023) | Kemampuan CNN untuk menggeneralisasi ke gambar dari domain yang belum pernah dilihat, terutama untuk tugas klinis yang bersifat safety-critical. | Berbagai dataset dermoskopi skin cancer, termasuk HAM10000, dievaluasi pada kondisi domain yang berbeda. | Evaluasi eksperimental lintas-dataset (cross-dataset evaluation). | Metrik performa klasifikasi lintas-dataset, dengan fokus pada degradasi performa antar domain. | Menunjukkan bahwa performa model menurun nyata ketika dievaluasi pada domain berbeda, sekaligus mengidentifikasi faktor-faktor penyebabnya. | Kami mengukur domain shift secara kuantitatif pada model kami sendiri, dari dermoskop ke foto smartphone: specificity turun dari 87,2% pada HAM10000 menjadi 44,9% pada PAD-UFES-20 dengan ambang yang sama. | https://doi.org/10.1016/j.nbt.2023.04.006 |
| R4 | Combalia, M.; Codella, N.; Rotemberg, V.; dkk. (2022) | Klaim performa AI dermatologi yang setara atau lebih baik dari dermatolog belum dinilai pada skenario yang realistis secara klinis. | Dataset dermatitis dan lesi kulit dari International Skin Imaging Collaboration 2019. | Validasi eksternal multi-pusat dengan sub-analisis pada kasus yang sulit. | Sensitivity, specificity, dan AUC pada subset kasus yang tidak ada dalam data training. | Performa AI menurun nyata pada kasus yang tidak pernah ada di training, sehingga validasi yang jujur dibutuhkan sebelum klaim klinis apa pun. | Kami menerapkan prinsip yang sama pada konteks yang berbeda, yaitu ganti perangkat kamera, bukan ganti pusat medis, dan mengukur biaya false negative ganas secara eksplisit lewat ambang 0,11 yang mengutamakan sensitivitas. | https://doi.org/10.1016/S2589-7500(22)00021-8 |
| R5 | Almaraz-Damian, J. A.; Ponomaryov, V.; Sadovnychiy, S.; dkk. (2020) | Klasifikasi melanoma versus nevus dengan sistem CAD yang menggabungkan fitur buatan tangan (aturan ABCD) dan fitur deep learning. | Citra lesi kulit dermoskopi untuk melanoma dan nevus. | Fusi fitur handcraft dan fitur deep learning dengan ukuran mutual information. | Accuracy dan metrik klasifikasi biner melanoma versus nevus. | Menunjukkan bahwa fitur handcraft tradisional dan fitur deep learning saling melengkapi serta meningkatkan performa bila digabung. | Kami membandingkan HOG + SVM dan HOG + Random Forest sebagai model terpisah, sehingga terlihat siapa yang lebih kuat dan apakah gabungannya lebih baik. Kami tidak menggabungkan fitur handcraft dengan deep learning dalam satu model. | https://doi.org/10.3390/e22040484 |
| R6 | Katsch, F.; Rinner, C.; Tschandl, P.; dkk. (2022) | Klasifikasi citra dermoskopi bisa terganggu oleh artefak umum seperti tanda di kulit dan penggaris. | Citra dermoskopi dengan artefak yang dianotasi. | Perbandingan arsitektur CNN (ResNet-34, Faster R-CNN, Mask R-CNN). | Performa klasifikasi pada gambar dengan artefak dibandingkan tanpa artefak. | Ketahanan terhadap artefak bervariasi antar arsitektur, dan artefak bisa menurunkan performa secara signifikan. | Kami menguji ketahanan pada pergeseran domain yang lebih besar, yaitu dermoskop ke foto smartphone dengan perbedaan resolusi, pencahayaan, dan kualitas gambar, bukan pada artefak tanda dan penggaris. | https://doi.org/10.5826/dpc.1203a126 |
| R7 | Rotemberg, V.; Kurtansky, N.; Betz-Stablein, B.; dkk. (2021) | Dataset kulit yang ada belum membahas informasi tingkat pasien, padahal dermatolog menilai keseluruhan pasien, bukan gambar per lesi. | SIIM-ISIC Melanoma dengan metadata klinis per pasien. | Artikel deskriptif data (data descriptor). | Tidak ada metrik model; fokus pada kelengkapan metadata. | Menyediakan metadata klinis dan pengelompokan lesi per pasien, yang dibutuhkan untuk menilai model secara realistis. | Kami membagi data per `lesion_id` untuk mencegah kebocoran dari lesi yang sama, dan memakai dataset tanpa metadata klinis per pasien, sehingga konteks pasien tidak tersedia bagi model maupun bagi kami. | https://doi.org/10.1038/s41597-021-00815-z |

## Catatan pembacaan

- Angka dan klaim jurnal adalah laporan penulis artikel, bukan hasil replikasi
  kami. Perbedaan split, dataset, preprocessing, dan metrik membuat peringkat
  lintas artikel tidak setara, jadi tabel ini tidak dipakai untuk melihat
  artikel mana yang "lebih baik".
- Kolom "Perbedaan dengan Project" adalah analisis penulis laporan ini,
  bukan kutipan dari artikel.
- Tanggal verifikasi: 11 November 2026, dan diulang pada 25 November 2026.
- Konfirmasi kelayakan referensi kepada dosen; tidak ada klaim persetujuan.

## Kriteria pemilihan artikel

1. Artikel terbit pada jurnal yang dapat diverifikasi lewat PubMed, sehingga
   judul, penulis, tahun, dan DOI dapat dicek otomatis.
2. Artikel menyentuh salah satu dari tiga topik yang dipakai laporan ini:
   dataset lesi kulit, pergeseran domain antardataset, atau cara memvalidasi
   model AI dermatologi.
3. Artikel memakai metrik yang bisa dibandingkan dengan metrik kami, yaitu
   accuracy, precision, recall, F1, sensitivity, specificity, atau AUC.
4. Artikel tersedia untuk dibaca dan tidak memuat klaim yang tidak dapat
   kami pertanggungjawabkan sebagai rujukan.

## Cara memverifikasi ulang

```bash
python scripts/verify_literature.py
```

Skrip itu mengambil setiap entri di `scripts/literature.py`, mencocokkan
judul, tahun, dan DOI dengan indeks PubMed, lalu menulis ulang hasilnya ke
`docs/logs/verifikasi_literature.json`. Tabel di dokumen ini disusun dari
entri yang sama, sehingga angka pada laporan dan isi tabel tidak mungkin
berbeda satu karakter pun tanpa ketahuan.

## Sitasi lengkap

1. Tschandl, P., Rosendahl, C., dan Kittler, H. (2018). The HAM10000 dataset,
   a large collection of multi-source dermatoscopic images of common
   pigmented skin lesions. *Scientific Data*, 5, 180161. DOI
   10.1038/sdata.2018.161.
2. Pacheco, A. G. C., Lima, G. R., Salomão, A. S., dkk. (2020). PAD-UFES-20:
   A skin lesion dataset composed of patient data and clinical images
   collected from smartphones. *Data in Brief*, 32, 106221. DOI
   10.1016/j.dib.2020.106221.
3. Fogelberg, K., Chamarthi, S., Maron, R. C., dkk. (2023). Domain shifts in
   dermoscopic skin cancer datasets: Evaluation of essential limitations for
   clinical translation. *New Biotechnology*, 76, 106-117. DOI
   10.1016/j.nbt.2023.04.006.
4. Combalia, M., Codella, N., Rotemberg, V., dkk. (2022). Validation of
   artificial intelligence prediction models for skin cancer diagnosis using
   dermoscopy images: the 2019 International Skin Imaging Collaboration Grand
   Challenge. *The Lancet Digital Health*, 4, e330-e339. DOI
   10.1016/S2589-7500(22)00021-8.
5. Almaraz-Damian, J. A., Ponomaryov, V., Sadovnychiy, S., dkk. (2020).
   Melanoma and nevus skin lesion classification using handcraft and deep
   learning feature fusion via mutual information measures. *Entropy*, 22(4),
   484. DOI 10.3390/e22040484.
6. Katsch, F., Rinner, C., Tschandl, P., dkk. (2022). Comparison of
   convolutional neural network architectures for robustness against common
   artefacts in dermatoscopic images. *Dermatology Practical & Conceptual*,
   12(3), e126. DOI 10.5826/dpc.1203a126.
7. Rotemberg, V., Kurtansky, N., Betz-Stablein, B., dkk. (2021). A
   patient-centric dataset of images and metadata for identifying melanomas
   using clinical context. *Scientific Data*, 8, 34. DOI
   10.1038/s41597-021-00815-z.
