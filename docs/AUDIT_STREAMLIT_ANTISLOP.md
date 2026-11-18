# Audit Streamlit + Antislop: DermAssist Clinical

| Keterangan | Isi |
|---|---|
| Tanggal | 19 November 2026 |
| Mode antislop | **DURING** (session override, dipilih pengguna) |
| Skill Streamlit | `streamlit/agent-skills@developing-with-streamlit` |
| Sumber skill | Repo GitHub resmi `streamlit` (227 bintang, Apache-2.0, 3,5K install) |
| Hasil pemeriksaan keamanan skill | Safe, 0 alert Socket, Low Risk Snyk |
| Versi Streamlit terpasang | 1.67.0 (referensi skill dimuat dari paket, cocok versi) |
| Objek audit | `app/app.py`, `app/requirements.txt`, `.streamlit/config.toml`, `app/contoh/` |
| Versi UI | 1.0 (rombak tampilan 18-19 November 2026) |

Dokumen ini menggantikan hasil audit 17 November 2026 karena antarmuka
dirombak total. Bagian yang berubah: header, tema, kartu, tab, pratinjau,
kartu metrik, lencana status, dan grafik probabilitas. Bagian prediksi
tidak berubah; buktinya ada di Bagian 5.

---

## 1. Cara skill diperoleh dan diverifikasi

Skill tidak diambil dari sumber sembarangan. Saya mencari lewat
`npx skills find streamlit`, lalu memeriksa tiga kandidat sebelum memasang:

| Kandidat | Install | Repo | Keputusan |
|---|---|---|---|
| `streamlit/agent-skills@developing-with-streamlit` | 3.5K | resmi `streamlit`, 227 bintang, Apache-2.0 | **Dipilih** |
| `streamlit/streamlit@debugging-streamlit` | 501 | resmi `streamlit` | Cadangan, tidak dipasang |
| `silvainfm/claude-skills@streamlit` | 309 | pihak ketiga | Ditolak (sumber tidak resmi) |

Perintah pemasangan: `npx skills add streamlit/agent-skills@developing-with-streamlit -g -y`.
Deskripsi skill menyatakan bahwa ini adalah routing resmi yang dikirim
bersama Streamlit >= 1.57, dan `discover.py` menemukan `SKILL.md` di dalam
paket `streamlit` yang terpasang, jadi referensinya benar-benar cocok
dengan versi 1.67.0.

Referensi yang dipakai pada rombakan ini: `api-reference.md` (tema lewat
`config.toml`, `st.container(border=True)`, `st.tabs`, `st.caption`),
`testing.md` (AppTest), dan `best-practices.md` (jangan memakai CSS untuk
hal yang bisa diatur tema).

---

## 2. Keputusan desain dan alasannya

| Keputusan | Alasan |
|---|---|
| Tema Teal & Warm Alabaster | Teal menandakan tenang dan klinis; latar alabaster (`#FAF9F6`) mengurangi silau dibanding putih murni, cocok untuk membaca hasil skrining |
| Judul `#0F766E`, aksen `#0D9488` | Dua tingkat teal: judul lebih gelap agar kontras 5,20:1, aksen untuk kontrol dan grafik |
| Rose `#E11D48` khusus kelas ganas | Merah dipakai hanya untuk makna bahaya, bukan hiasan; kontras terhadap putih 4,70:1 |
| Header tiga baris + lencana versi di kanan | Menghapus label huruf besar di atas judul; hierarki dibaca dari ukuran, bukan dari dekorasi |
| Panduan memotret hanya di sidebar | Menghilangkan duplikasi; konten utama fokus pada foto dan hasil |
| Dua kartu berbingkai | Memisahkan input dan hasil tanpa kotak peringatan bertepi kiri tebal |
| Tab `Unggah Foto` / `Gunakan Sampel` | Dua cara memasukkan foto tidak perlu tampil bersamaan |
| Pratinjau maksimal 360 px | Foto tidak mendominasi kartu dan tidak mendorong hasil ke bawah |
| Grafik Plotly tanpa gridline | Panjang batang dan angka sudah cukup; gridline menambah tinta tanpa informasi |
| Disclaimer satu kali di bawah | Tidak mengulang peringatan yang sama di beberapa kotak |

Tipografi memakai font bawaan tema Streamlit (tanpa font kustom) supaya
tidak menambah unduhan dan tetap konsisten dengan komponen shadcn.
Lencana versi memakai varian `outline`; lencana ganas memakai
`destructive` + `outline` karena teks putih di atas teal `#0D9488` hanya
3,74:1 (gagal 4,5:1 untuk teks normal).

---

## 3. Perubahan dibanding versi 17 November

| Area | Sebelum | Sesudah |
|---|---|---|
| Header | Label huruf besar + judul + subjudul, tanpa versi | Tiga baris (judul, subjudul, caption) + lencana `Versi 1.0` di kanan |
| CSS kustom | 260 baris, termasuk patch sidebar dan tombol | 2 aturan: sembunyikan tombol Deploy, `padding-top: 2rem` |
| Tema | Palet paper/slate/brick | Teal + Warm Alabaster di `config.toml` |
| Kotak status | `st.success`/`st.warning` dengan border kiri tebal | Lencana shadcn + satu paragraf penjelasan |
| Panduan foto | Di sidebar dan di panel utama | Hanya di sidebar, berupa poin dengan pemimpin tebal |
| Input | Uploader dan contoh tampil bersamaan | Dua tab di dalam kartu |
| Pratinjau | Lebar penuh kolom | Maksimal 360 px + caption nama file |
| Angka utama | Tiga `st.metric` kecil | Dua `ui.metric_card`: prediksi dan risiko vs ambang 11% |
| Grafik | Bar CSS buatan sendiri | Plotly horizontal, 7 batang, teal/rose |
| Disclaimer | Tiga kotak terpisah | Satu `st.caption` di paling bawah |

Berkas cadangan sebelum perubahan: `backup_ui_2026-11-18/app_backup.py`
dan `backup_ui_2026-11-18/config_backup.toml`.

---

## 4. Delivery Gate (antislop, wajib sebelum deliver)

### Blok 1: Hard Gate

| Item | Status | Bukti |
|---|---|---|
| R-02 tidak ada em dash | **PASS** | `grep` pada `app/app.py`: 0 hasil (panduan memakai titik dua) |
| R-03 tidak ada overflow / layout rusak | **PASS** | Headless Edge: `scrollWidth == innerWidth` pada 390, 768, 1024, dan 1440 px; grafik 326 px di layar 390 px |
| R-17 angka bersumber nyata | **PASS** | Angka di UI berasal dari `result` predictor dan `threshold_config.json`; tidak ada angka hard-code selain ambang yang dibaca dari config |
| R-18 tidak ada testimoni palsu | **PASS** | Tidak ada bagian testimoni |
| R-23 aset tanpa izin | **PASS** | 4 contoh gambar dari HAM10000/PAD-UFES-20; tidak ada logo atau avatar buatan |
| R-24 nav ke halaman kosong | **PASS** | Aplikasi satu halaman, tanpa navbar |
| R-25 kontras WCAG AA | **PASS** | `scripts/audit_kontras.py`: 9 pasangan, 0 gagal. Terendah teks normal 4,51:1 (caption); grafik 3,74:1 dengan minimum 3,0 untuk elemen grafis |
| R-26 kontrol berfungsi | **PASS** | AppTest: unggah dan sampel sama-sama menghasilkan 7 batang; file rusak menghasilkan pesan ramah |
| R-27 empty/loading/error | **PASS** | Empty state dua baris, spinner "Menganalisis foto...", tiga `st.error` |
| R-28 FAQ generik | **PASS** | Tidak ada FAQ di aplikasi |
| R-32 keyboard + fokus | **PASS** | Tab traversal 14 langkah: tab, tombol bantuan, tombol upload, dan expander semuanya `:focus-visible` dengan cincin teal `rgba(13,148,136,0.5)` 3,2 px |
| R-33 patch CSS via skrip | **PASS** | Perubahan langsung di `app.py`; tidak ada skrip yang menulis ulang CSS |
| R-34 tema ganda | **PASS** | Satu tema, tanpa toggle |
| R-35 dijalankan dan diklik | **PASS** | Edge 1440 px: klik tab, centang sampel, unggah file, buka sidebar, buka expander. 0 error konsol, 0 page error |
| R-36 klaim palsu | **PASS** | Tidak ada klaim keamanan atau akurasi yang tidak terukur |
| R-37 arah desain | **PASS** | Arah "klinis tenang" tertulis di komentar `app.py` dan Bagian 2 dokumen ini |
| R-38 konten nyata | **PASS** | Nama dataset, lisensi, dan sitasi berasal dari file proyek |
| R-39 semantik dan affordance | **PASS** | `alt=` pada dua `st.image`; label pada semua input; kontrol native |
| R-40 performa media | **PASS** | Tidak ada `<img>` atau `<video>` mentah; gambar contoh 4 berkas kecil |

### Blok 2: Purpose-Gate

| Item | Status | Alasan tertulis |
|---|---|---|
| R-01 warna/gradien | **PASS** | Tanpa gradien; palet teal + alabaster dijelaskan di Bagian 2 |
| R-04 ikon generik | **PASS** | `page_icon` memakai `:material/biotech:`, relevan dengan konteks medis |
| R-06 tipografi | **PASS** | Font bawaan tema; hierarki dari ukuran dan warna, bukan font hias |
| R-07 grid latar | **PASS** | Tidak ada |
| R-08 panah dekoratif | **PASS** | Tidak ada pada tombol |
| R-09 badge | **PASS** | Lencana menyatakan versi dan status risiko, bukan klaim "AI Powered" |
| R-10 glassmorphism | **PASS** | Tidak ada |
| R-12 shadow | **PASS** | Kartu memakai border 1 px, bukan bayangan |
| R-13 glow | **PASS** | Tidak ada |
| R-14 kartu identik | **PASS** | Kartu input dan hasil punya isi berbeda; kartu metrik dibedakan oleh angka dan lencana |
| R-19 animasi | **PASS** | Hanya spinner bawaan Streamlit |
| R-22 ilustrasi generik | **PASS** | Tidak ada ilustrasi stok |

### Blok 3: Liveliness

Dial yang dipakai: **ENERGY 1 / RHYTHM 2 / MOTION 1**. Halaman terasa
tenang: satu judul, dua kartu, satu grafik. Satu-satunya warna panas
adalah rose untuk kelas ganas, dan hanya muncul setelah prediksi.

### Blok 4: Craftsmanship dan Quality Locks

| Item | Status | Bukti ringkas |
|---|---|---|
| C-1 intentionality | **PASS** | Setiap warna dan pemisahan kartu punya alasan di komentar atau Bagian 2 |
| C-2 functional completeness | **PASS** | Semua widget berfungsi (AppTest dan browser) |
| C-3 content-driven | **PASS** | Urutan halaman mengikuti alur skrining: foto, angka, status, rincian |
| C-4 resilience | **PASS** | Empty, loading, dan error state ada |
| C-5 evidence over claims | **PASS** | Angka dari log dan config, bukan karangan |
| R-05 layout | **PASS** | Dua kolom fungsional, bukan hero + tiga kartu generik |
| R-11 radius | **PASS** | 8 px konsisten pada kartu, input, dan komponen shadcn |
| R-15 CTA | **PASS** | Tidak ada "Get Started" atau CTA palsu |
| R-16 buzzwords | **PASS** | Tidak ada istilah pemasaran |
| R-20 identitas | **PASS** | Palet teal/alabaster dan tata letak khusus proyek ini |
| R-21 dark mode | **PASS** | Tidak ada toggle, tema terkunci |
| R-29 palet | **PASS** | Empat warna inti: `#0F766E`, `#0D9488`, `#E11D48`, `#1C1917`, di atas `#FAF9F6` |
| R-30 kloning produk | **PASS** | Tidak meniru produk komersial tertentu |
| R-31 alasan keputusan | **PASS** | Ditulis di komentar `app.py` dan Bagian 2 dokumen ini |

---

## 5. Bukti bahwa logika prediksi tidak berubah

Rombakan hanya menyentuh tampilan. Yang dijaga:

1. `PR.predict(predictor, image)` dipanggil dengan objek PIL yang sama,
   sekali per interaksi, dan hasilnya tidak diubah.
2. `PR.is_uncertain(result)` tetap satu-satunya penentu status "area
   abu-abu".
3. Ambang tetap dibaca dari `result["threshold"]` (0,11), bukan ditulis
   ulang di UI.
4. `st.cache_resource` untuk model tidak berubah.
5. `src/predictor.py`, `src/config.py`, dan berkas model tidak disentuh.

Pembanding berkas cadangan:

```powershell
git diff --no-index backup_ui_2026-11-18/app_backup.py app/app.py
```

Perbedaan hanya ada pada bagian tampilan dan impor `plotly` serta
`streamlit_shadcn_ui`.

---

## 6. Status akhir

**Delivery Gate: PASS.** Tidak ada item FAIL yang dikirim.

Verifikasi terakhir:

| Pemeriksaan | Hasil |
|---|---|
| `python scripts/verify_final.py` | 12 pemeriksaan lulus, exit 0 |
| `python scripts/test_app_content.py` | Lulus: tab, 7 batang, 4 komponen shadcn, alur sampel |
| `python scripts/audit_kontras.py` | Lulus: 9 pasangan, 0 gagal |
| `python scripts/simulasi_deploy.py` | LULUS: 147 berkas, 97,5 MB, aplikasi boot dari checkout bersih |
| Browser Edge 1440 px | 0 console error, 7 batang, fill `rgb(13,148,136)` dan `rgb(225,29,72)` |
| Browser Edge 390/768/1024 px | Tidak ada overflow horizontal |
| Deploy Streamlit Community Cloud (Python 3.13) | Live di `projectforestryml2026mendekteksilesikulit-vvearxur6nuwga44fmaf.streamlit.app`. Build pertama gagal karena `tqdm` belum ada di requirements; sudah diperbaiki, dan sekarang ada pemeriksa `scripts/cek_import_runtime.py` |

### Batasan yang jujur

- Uji browser memakai Edge headless (Chromium 157). Safari dan Firefox
  belum diuji; tata letak hanya memakai komponen native Streamlit dan
  Plotly, jadi risikonya kecil tetapi tidak nol.
- Kontras dihitung dari nilai tema dan campuran 60% yang dipakai
  Streamlit untuk caption, bukan dari eyedropper. Nilai caption 4,51:1
  hanya 0,01 di atas ambang, jadi perubahan `textColor` di masa depan
  harus diikuti audit ulang.
- Pembacaan layar (screen reader) tidak diuji dengan NVDA/JAWS. Yang
  diverifikasi adalah struktur semantik dan fokus keyboard.
