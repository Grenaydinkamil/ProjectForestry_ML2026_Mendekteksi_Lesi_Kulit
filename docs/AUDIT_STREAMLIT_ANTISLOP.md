# Audit Streamlit + Antislop — Proyek ML2026

| Keterangan | Isi |
|---|---|
| Tanggal | 17 November 2026 |
| Mode antislop | **DURING** (session override, dipilih pengguna) |
| Skill Streamlit | `streamlit/agent-skills@developing-with-streamlit` |
| Sumber skill | Repo GitHub resmi `streamlit` (227 bintang, Apache-2.0, 3,5K install) |
| Hasil pemeriksaan keamanan skill | Safe · 0 alert Socket · Low Risk Snyk |
| Versi Streamlit terpasang | 1.67.0 (referensi skill dimuat dari paket, cocok versi) |
| Objek audit | `app/app.py`, `src/predictor.py`, `.streamlit/config.toml`, `app/contoh/` |

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
Diskripsi menyatakan skill ini adalah routing resmi yang dikirim bersama
Streamlit ≥1.57, dan `discover.py` menemukan `SKILL.md` di dalam paket
`streamlit` yang terpasang, jadi referensinya benar-benar cocok dengan versi
1.67.0.

---

## 2. Temuan dan perbaikan (mode DURING)

Temuan diperbaiki langsung di source, bukan hanya dicatat.

| # | Temuan | Aturan | Bukti | Perbaikan |
|---|---|---|---|---|
| 1 | Dua `st.image()` tanpa `alt=` | Antislop R-39; Streamlit best practices | `app.py` baris 540, 575 | Ditambah `alt="Foto contoh lesi kulit..."` dan `alt="Foto lesi kulit yang diunggah pengguna"` |
| 2 | Teks `--ink-faint` `#6E757E` kontrasnya 4,46:1 (di bawah 4,5:1) | Antislop R-25 | `scripts/audit_kontras.py` | Diubah ke `#6A717A` → **4,73:1** |
| 3 | Sidebar terbuka saat aplikasi dibuka, menutupi konten utama pada lebar sempit | Antislop R-03 | Screenshot browser menunjukkan sidebar ±450 px | `initial_sidebar_state` diubah ke `"collapsed"` |
| 4 | `use_container_width`, `st.components.v1`, CSS patch via skrip | Streamlit best practices | Tidak ditemukan di `app.py` | Tidak ada tindakan |
| 5 | CSS kustom 260 baris via `st.markdown(unsafe_allow_html=True)` | Streamlit: config.toml lebih disukai | Warna dasar ternyata sudah di `.streamlit/config.toml` | **Dipertahankan**, alasan ditulis: komponen status dan bar probabilitas tidak punya padanan native. R-31 terpenuhi. |
| 6 | `@st.cache_resource` untuk model | Streamlit performance | `get_predictor()` sudah memakainya | Tidak ada tindakan |
| 7 | Pengujian lewat AppTest | Streamlit testing reference | `scripts/test_app_content.py` | Sudah sesuai referensi skill |

Skrip audit kontras disimpan di `scripts/audit_kontras.py` dan didaftarkan
ke `scripts/verify_final.py` sebagai pemeriksaan 7b, sehingga tidak bisa
hilang pada run berikutnya.

---

## 3. Delivery Gate (antislop, wajib sebelum deliver)

### Blok 1 — Hard Gate

| Item | Status | Bukti |
|---|---|---|
| R-02 tidak ada em dash | **PASS** | `grep "—" app/app.py` → 0 hasil |
| R-03 tidak ada overflow / layout rusak | **PASS** | Browser: `scrollWidth 1283 == innerWidth 1283`, sidebar collapse. Catatan: lebar ponsel 375 px tidak dirender di sesi ini; layout responsif mengandalkan Streamlit. |
| R-17 angka bersumber nyata | **PASS** | Semua angka di app dan laporan dari `docs/logs/*.json` |
| R-18 tidak ada testimoni palsu | **PASS** | Tidak ada bagian testimoni |
| R-23 aset tanpa izin | **PASS** | 4 contoh gambar berasal dari HAM10000/PAD-UFES-20, tanpa logo/avatar buatan |
| R-24 nav ke halaman kosong | **PASS** | Aplikasi tidak punya navbar |
| R-25 kontras WCAG AA | **PASS** | `audit_kontras.py`: 7 pasangan, 0 gagal, terendah 4,73:1 |
| R-26 kontrol berfungsi | **PASS** | AppTest: upload → 7 bar; contoh → 7 bar; file rusak → pesan ramah |
| R-27 empty/loading/error | **PASS** | empty state, spinner "Menganalisis foto...", 3 `st.error` |
| R-28 FAQ generik | **PASS** | Tidak ada FAQ di aplikasi (FAQ ada di dokumen, bukan UI) |
| R-32 keyboard + fokus | **PASS** | Kontrol native Streamlit; `document.activeElement` checkbox → `outlineStyle: auto` |
| R-33 patch CSS via skrip | **PASS** | Perubahan langsung di `app.py`; tidak ada skrip pengubah CSS |
| R-34 tema ganda | **PASS** | Hanya satu tema, tanpa toggle |
| R-35 dijalankan & diklik | **PASS** | AppTest + browser: centang contoh → status "Ciri ganas terdeteksi", console error 0 |
| R-36 klaim palsu | **PASS** | Tidak ada klaim keamanan/kinerja yang tidak terukur |
| R-37 arah desain | **PASS** | Arah klinis tertulis di komentar CSS (paper, slate, brick) |
| R-38 konten nyata | **PASS** | Nama anggota, NIM, dataset, dan lisensi berasal dari file proyek |
| R-39 semantik & affordance | **PASS** | `alt=` ada, tombol native dengan `aria-label`, `cursor: pointer` ada di CSS |
| R-40 performa media | **PASS** | Tidak ada `<img>`/`<video>` mentah yang ditulis aplikasi |

### Blok 2 — Purpose-Gate

| Item | Status | Alasan tertulis |
|---|---|---|
| R-01 warna/gradien | **PASS** | Tanpa gradien. Palet klinis: paper + slate + brick, alasan di komentar CSS |
| R-04 ikon generik | **PASS** | Tidak ada ikon hias; `page_icon` kosong sengaja |
| R-06 tipografi | **PASS** | Cambria untuk judul (kesan laporan klinis), Calibri untuk isi, tercatat di CSS |
| R-07 grid latar | **PASS** | Tidak ada |
| R-08 panah dekoratif | **PASS** | Tidak ada pada tombol |
| R-09 badge | **PASS** | Tidak ada badge "AI Powered" sejenisnya |
| R-10 glassmorphism | **PASS** | Tidak ada |
| R-12 shadow | **PASS** | Tidak dipakai; pemisah memakai hairline |
| R-13 glow | **PASS** | Tidak ada |
| R-14 kartu identik | **PASS** | Kotak status dibedakan warna menurut makna, bukan template |
| R-19 animasi | **PASS** | Hanya spinner bawaan Streamlit; `prefers-reduced-motion` dihormati di CSS |
| R-22 ilustrasi generik | **PASS** | Tidak ada |

### Blok 3 — Liveliness

Dial yang dipakai: **ENERGY 1 / RHYTHM 2 / MOTION 1** (klinis tenang,
satu fokus utama di panel hasil, satu aksen merah bata hanya untuk
status ganas). Ini sesuai R-37 karena arah desain memang disengaja,
bukan default steril.

### Blok 4 — Craftsmanship & Quality Locks

| Item | Status | Bukti ringkas |
|---|---|---|
| C-1 intentionality | **PASS** | Setiap warna punya alasan di komentar |
| C-2 functional completeness | **PASS** | Semua widget berfungsi (AppTest) |
| C-3 content-driven | **PASS** | Bagian ada karena kebutuhan skrining, bukan template |
| C-4 resilience | **PASS** | 3 state UI hadir |
| C-5 evidence over claims | **PASS** | Angka dari log, tidak dikarang |
| R-05 layout | **PASS** | Tidak ada hero+3 kartu generik |
| R-11 radius | **PASS** | Radius 2 px konsisten |
| R-15 CTA | **PASS** | Tidak ada "Get Started" dll |
| R-16 buzzwords | **PASS** | Tidak ada |
| R-20 identitas | **PASS** | Palet dan tipografi spesifik |
| R-21 dark mode | **PASS** | Tidak ada toggle, tema terkunci sengaja |
| R-29 palet | **PASS** | 3 warna inti + 1 aksen |
| R-30 kloning produk | **PASS** | Tidak meniru produk lain |
| R-31 alasan keputusan | **PASS** | Ditulis di komentar CSS dan laporan ini |

---

## 4. Status akhir

**Delivery Gate: PASS.** Tidak ada item FAIL yang dikirim.

Verifikasi terakhir dijalankan dengan `python scripts/verify_final.py`:
12 pemeriksaan (termasuk audit kontras) semua lulus. Aplikasi juga sudah
dibuka di browser dan prediksi contoh gambar berhasil menampilkan status
"Ciri ganas terdeteksi" dengan console error nol.

### Yang belum bisa diverifikasi di sesi ini

- Render pada lebar ponsel 375 px tidak dapat dilakukan karena jendela
  desktop tidak bisa diubah ukurannya dari sesi ini. R-03 dinyatakan PASS
  berdasarkan tidak adanya overflow pada lebar yang diuji dan perilaku
  responsif bawaan Streamlit, bukan berdasarkan render ponsel sungguhan.
- Type checker `ty`/`mypy` tidak dijalankan karena tidak terpasang di venv.
  Pemeriksaan API dan variabel sudah menggantikan sebagian fungsi ini.
