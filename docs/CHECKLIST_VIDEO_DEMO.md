# Checklist Video Demo (Maksimum 7 Menit)

Dokumen ini adalah panduan rekaman video demo aplikasi. Semua langkah di
bawah ini sudah diverifikasi berjalan di proyek ini.

**Aturan utama:** video harus menunjukkan aplikasi **benar-benar berjalan**,
bukan slide. Waktu total tidak boleh melebihi **7 menit**.

---

## 1. Persiapan (sebelum merekam)

| Yang perlu | Detail |
|---|---|
| Aplikasi jalan | `streamlit run app/app.py` (dari folder utama repo) |
| Model tersedia | `models/mobilenetv2_best.keras` + `models/preprocess_config.json` + `models/threshold_config.json` |
| Contoh foto | **Sudah disediakan di aplikasi**: kotak "Atau pakai contoh" di panel kiri berisi melanoma, karsinoma sel basal, nevus jinak, dan melanoma foto smartphone. Tidak perlu mencari file sendiri. |
| Browser | Chrome/Edge, zoom 100%, satu tab saja |
| Rekaman | OBS Studio atau Screen Recorder, 1080p, mikrofon bersih |
| Angka acuan | Buka `docs/REKAP_LAPORAN.md` di tab lain untuk pengecekan angka |

> Penting: jangan rekam saat aplikasi baru dibuka. Streamlit memuat model
> sekali saat halaman pertama dibuka. Buka halaman **sebelum** mulai
> merekam supaya tidak ada jeda "Memuat model..." di tengah video.

---

## 2. Susunan Video (total 6 menit 30 detik)

### Bagian 1 - Pembuka (0:00 - 0:30) - 30 detik

- Tampilkan halaman aplikasi dengan disclaimer terlihat.
- Satu kalimat pembuka: "Ini aplikasi skrining risiko lesi kulit berbasis
  machine learning, dibangun dari dataset HAM10000 berisi 10.015 foto
  dermoskop."

### Bagian 2 - Cara memotret (0:30 - 1:00) - 30 detik

- Tunjuk panel **Panduan memotret** di sidebar (5 poin).
- Tunjuk blok **Privasi**: foto diproses di memori, tidak disimpan.

### Bagian 3 - Unggah foto jinak (1:00 - 2:00) - 60 detik

1. Klik kotak "**Atau pakai contoh**", pilih **Nevus jinak (dermoskop)**,
   centang "Pakai contoh ini". (Atau unggah file sendiri lewat area
   unggahan.)
2. Tunggu hasil muncul.
3. Tunjuk:
   - kotak status hijau "Tidak ditemukan ciri ganas"
   - nama diagnosis yang tertebalkan
   - **7 bar probabilitas per kelas** (tunjuk 2-3 bar saja)
   - jelaskan: "Bar merah menandai kelas ganas: melanoma, karsinoma sel
     basal, dan keratosis atipis."
4. Tunjuk angka ambang: "Ambang {ANGKA}, dihitung dari data validasi,
   bukan data uji." (Ambang diambil dari `models/threshold_config.json`.)

### Bagian 4 - Unggah foto ganas (2:00 - 3:00) - 60 detik

1. Ganti contoh ke **Melanoma (dermoskop)** atau **Melanoma (foto
   smartphone)**, lalu centang "Pakai contoh ini".
2. Tunjuk:
   - kotak status merah "Ciri ganas terdeteksi"
   - persentase risiko ganas di atas ambang
   - bar merah pada kelas ganas
3. Tunjukkan juga **satu** bar warna yang menjelaskan kenapa model
   berpikir jinak (misalnya "nevus 62%, melanoma 21%").
4. Kalau sempat, pakai contoh **foto smartphone** untuk menunjukkan
   perbedaan domain: foto yang berasal dari dermal scope dan foto
   yang diambil dengan kamera HP memberikan hasil berbeda. Ini bagian yang
   paling menarik untuk ditanyakan.

### Bagian 5 - Kasus tidak yakin (3:00 - 3:30) - 30 detik

- Unggah foto yang hasilnya "belum meyakinkan" (area abu-abu).
- Tunjuk kotak kuning beserta penjelasan "periksa ke dokter kulit".
- Satu kalimat: "Aplikasi sengaja tidak memaksa jawaban kalau modelnya
  tidak yakin."

### Bagian 6 - Penanganan error (3:30 - 4:00) - 30 detik

- Unggah file **bukan gambar** (misalnya berkas teks yang diberi ekstensi
  `.png`).
- Tunjuk pesan ramah "File yang dipilih bukan gambar yang bisa dibaca".
- Ini menunjukkan edge case sudah ditangani, bukan crash.

### Bagian 7 - Informasi model (4:00 - 4:45) - 45 detik

- Tunjuk tabel **Model yang dipakai** di bawah: model, jenis, ukuran
  input, ambang risiko ganas, sumber dataset.
- Tunjuk footer: "Tschandl dkk. (2018), Scientific Data, lisensi
  CC BY-NC 4.0."

### Bagian 8 - Angka kunci (4:45 - 5:45) - 60 detik

Tampilkan layar split: buka tab `docs/REKAP_LAPORAN.md` atau slide
perbandingan model di PPT. Sebutkan empat angka saja:

1. **F1 macro** model terbaik di data uji
2. **Recall ganas** (sensitivitas) di data uji
3. **Jumlah FN** (lesi ganas yang lolos) di data uji
4. Hasil **uji eksternal PAD-UFES-20**: recall ganas model terbaik di foto
   smartphone

Kalimat kunci: "Model diuji dua kali: di data uji HAM10000, dan di dataset
eksternal PAD-UFES-20 yang foto-fotonya diambil dengan smartphone. Di
dataset eksternal, angka recall ganas turun dari X ke Y. Itu tradeoff yang
kami laporkan dengan jujur, bukan disembunyikan."

### Bagian 9 - Penutup (5:45 - 6:15) - 30 detik

- Kembali ke tab aplikasi.
- Tunjuk disclaimer sekali lagi.
- Satu kalimat penutup: "Aplikasi ini alat bantu belajar, bukan alat
  diagnosis. Diagnosis tetap oleh dokter."

### Bagian 10 - Demo kode (6:15 - 7:00) - 45 detik (opsional tapi disarankan)

Buka terminal, jalankan **satu** perintah yang pendek:

```bash
streamlit run app/app.py
```

atau tunjukkan struktur folder:

```bash
tree notebooks src app scripts
```

Sebutkan: "Delapan notebook, satu folder src bersama yang dipakai notebook
dan aplikasi, sehingga preprocessing di aplikasi dijamin sama dengan saat
training."

---

## 3. Larangan

- Jangan mulai langsung unggah foto tanpa penjelasan pembuka.
- Jangan menampilkan proses training (terlalu lama).
- Jangan melewati bagian disclaimer di pembuka maupun penutup.
- Jangan memakai foto yang melanggar privasi pasien. Semua foto di repo
  berasal dari dataset publik HAM10000 dan PAD-UFES-20.
- Jangan menyebut angka yang tidak ada di `docs/logs/*.json`.

---

## 4. Checklist Teknis Sebelum Kirim

- [ ] Durasi video tidak lebih dari 7 menit
- [ ] Audio jernih, tidak ada suara napas yang keras
- [ ] Semua teks di aplikasi terbaca (zoom cukup)
- [ ] Tidak ada error merah di terminal saat aplikasi dijalankan
- [ ] Disclaimer terlihat di frame pertama
- [ ] Angka yang disebut cocok dengan `docs/logs/hasil_evaluasi_final.json`
- [ ] Nama file video rapi, contoh:
      `Video_Demo_ML2026_Kelompok_1.mp4`

---

## 5. Angka yang Wajib Benar

Jangan menghafal angka. Buka file berikut saat merekam:

| File | Isi |
|---|---|
| `docs/logs/hasil_evaluasi_final.json` | Metrik data uji dan model terpilih |
| `docs/logs/hasil_uji_eksternal_pad_ufes.json` | Hasil uji PAD-UFES-20 |
| `docs/logs/model_klasik_validasi.json` | Metrik model klasik di validasi |
| `models/threshold_config.json` | Ambang risiko ganas yang dipakai aplikasi |

Ambang risiko ganas **dihitung dari data validasi**, supaya data uji tetap
dipakai sekali saja. Ini bagian yang sering ditanyakan dosen, jadi siapkan
penjelasan singkatnya.
