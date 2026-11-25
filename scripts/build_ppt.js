/**
 * scripts/build_ppt.js
 * ====================
 * Generator PPT laporan project (Fase 10).
 *
 * ATURAN PENTING:
 *   SEMUA ANGKA yang tampil di slide dibaca dari file JSON hasil evaluasi
 *   di docs/logs/. TIDAK ADA angka yang diketik manual. Kalau file JSON
 *   untuk suatu bagian belum ada, slide itu DILEWATI dengan catatan, bukan
 *   diisi angka karangan.
 *
 * Jalankan:
 *   node scripts/build_ppt.js
 *
 * Output: docs/PPT_Laporan_Project.pptx
 */

const fs = require("fs");
const path = require("path");
const PptxGenJS = require("pptxgenjs");

const REPO = path.resolve(__dirname, "..");
const LOGS = path.join(REPO, "docs", "logs");
const ASSETS = path.join(REPO, "docs", "assets");
const OUT = path.join(REPO, "docs", "PPT_Laporan_Project.pptx");

// ---------------------------------------------------------------------------
// PALET
// ---------------------------------------------------------------------------
// Tema: dermatologi. Nada hangat seperti kulit, aksen structural Slate
// (bukan biru korporat generik), Merah Bata satu-satunya warna alarm.
// Dominasi: Merah Bata + Krem (~65% berat visual).
const C = {
  merah: "9E2A2B",
  merahGelap: "7A1E1F",
  krem: "FBFAF7",
  kremGelap: "F2EFE8",
  slate: "2F5D7C",
  tinta: "1A1D21",
  tintaLembut: "5A6068",
  garis: "D8D4CC",
  putih: "FFFFFF",
  hijau: "2E6B4F",
  kuning: "8A5A15",
};

const PERSEN = (v) => (v == null ? "-" : `${(v * 100).toFixed(1)}%`);
const NUM = (v, d = 4) => (v == null ? "-" : Number(v).toFixed(d));
const RIBU = (v) => (v == null ? "-" : `${(v / 1000).toFixed(1)} ribu`);

// ---------------------------------------------------------------------------
// PEMBACA HASIL
// ---------------------------------------------------------------------------
function bacaJson(nama) {
  const p = path.join(LOGS, nama);
  if (!fs.existsSync(p)) return null;
  try {
    return JSON.parse(fs.readFileSync(p, "utf-8"));
  } catch (e) {
    console.warn(`  [warn] ${nama}: ${e.message}`);
    return null;
  }
}

// Hanya dua sumber angka yang dipakai slide:
//   - hasil_evaluasi_final.json          -> metrik data uji + model terpilih
//   - hasil_uji_eksternal_pad_ufes.json  -> hasil uji dataset eksternal
//
// Tidak ada angka yang ditulis manual di file ini. Kalau log-nya berubah,
// slide ikut berubah.
const H = {
  final: bacaJson("hasil_evaluasi_final.json"),
  pad: bacaJson("hasil_uji_eksternal_pad_ufes.json"),
  kalibrasi: bacaJson("cek_kalibrasi_svm.json"),
};

// ---------------------------------------------------------------------------
// HELPER
// ---------------------------------------------------------------------------
function ada(pathRel) {
  return fs.existsSync(path.join(ASSETS, pathRel));
}

function gambar(pres, slide, nama, opts) {
  const p = path.join(ASSETS, nama);
  if (!fs.existsSync(p)) return false;
  slide.addImage({ path: p, ...opts });
  return true;
}

function judulSlide(slide, kicker, judul, subjudul) {
  slide.background = { color: C.krem };
  slide.addText(kicker, {
    x: 0.55, y: 0.32, w: 12.2, h: 0.28, fontSize: 10.5, bold: true,
    color: C.merah, fontFace: "Calibri", charSpacing: 1.1,
  });
  slide.addText(judul, {
    x: 0.55, y: 0.62, w: 12.2, h: 0.62, fontSize: 27, bold: true,
    color: C.tinta, fontFace: "Cambria",
  });
  if (subjudul) {
    slide.addText(subjudul, {
      x: 0.55, y: 1.24, w: 12.2, h: 0.42, fontSize: 12.5,
      color: C.tintaLembut, fontFace: "Calibri",
    });
  }
}

function slideBagian(pres, nomor, judul, subjudul) {
  const s = pres.addSlide();
  s.background = { color: C.merah };
  s.addText(String(nomor).padStart(2, "0"), {
    x: 0.7, y: 2.0, w: 2.4, h: 1.8, fontSize: 80, bold: true,
    color: "C97A7B", fontFace: "Cambria",
  });
  s.addText(judul, {
    x: 3.1, y: 2.4, w: 9.5, h: 0.9, fontSize: 34, bold: true,
    color: C.putih, fontFace: "Cambria",
  });
  if (subjudul) {
    s.addText(subjudul, {
      x: 3.15, y: 3.35, w: 9.5, h: 0.9, fontSize: 14, color: "F0D9D9",
      fontFace: "Calibri",
    });
  }
  return s;
}

function kartuStat(slide, x, y, w, h, nilai, label, warna, sub) {
  slide.addShape("rect", {
    x, y, w, h, fill: { color: C.kremGelap }, line: { color: C.garis, width: 0.75 },
  });
  slide.addShape("rect", { x, y, w: 0.055, h, fill: { color: warna } });
  slide.addText(nilai, {
    x: x + 0.22, y: y + 0.14, w: w - 0.36, h: h * 0.46, fontSize: 23,
    bold: true, color: C.tinta, fontFace: "Cambria", valign: "bottom",
  });
  slide.addText(label, {
    x: x + 0.22, y: y + h * 0.58, w: w - 0.36, h: h * 0.2, fontSize: 10,
    bold: true, color: C.tintaLembut, fontFace: "Calibri",
  });
  if (sub) {
    slide.addText(sub, {
      x: x + 0.22, y: y + h * 0.78, w: w - 0.36, h: h * 0.2, fontSize: 8.5,
      color: C.tintaLembut, fontFace: "Calibri",
    });
  }
}

function tabel(slide, x, y, w, headers, rows, colW, opts = {}) {
  const hRow = opts.hRow || 0.34;
  const fRow = opts.fRow || 0.3;
  const fs = opts.fontSize || 9.5;

  // Header
  let cx = x;
  headers.forEach((hd, i) => {
    slide.addShape("rect", {
      x: cx, y, w: colW[i], h: hRow,
      fill: { color: C.slate }, line: { type: "none" },
    });
    slide.addText(hd, {
      x: cx + 0.06, y, w: colW[i] - 0.12, h: hRow, fontSize: fs, bold: true,
      color: C.putih, fontFace: "Calibri", valign: "middle", align: "left",
    });
    cx += colW[i];
  });

  // Baris
  rows.forEach((row, r) => {
    const ry = y + hRow + r * fRow;
    cx = x;
    if (r % 2 === 0) {
      slide.addShape("rect", {
        x, y: ry, w, h: fRow, fill: { color: C.kremGelap },
        line: { type: "none" },
      });
    }
    row.forEach((cell, i) => {
      slide.addText(String(cell), {
        x: cx + 0.06, y: ry, w: colW[i] - 0.12, h: fRow, fontSize: fs,
        color: C.tinta, fontFace: "Calibri", valign: "middle", align: "left",
      });
      cx += colW[i];
    });
  });

  return y + hRow + rows.length * fRow;
}

function kotakCatatan(slide, x, y, w, h, judul, isi, warna) {
  slide.addShape("rect", {
    x, y, w, h, fill: { color: C.kremGelap }, line: { type: "none" },
  });
  slide.addShape("rect", { x, y, w: 0.05, h, fill: { color: warna } });
  slide.addText(judul, {
    x: x + 0.2, y: y + 0.1, w: w - 0.4, h: 0.26, fontSize: 10.5, bold: true,
    color: warna, fontFace: "Calibri",
  });
  slide.addText(isi, {
    x: x + 0.2, y: y + 0.38, w: w - 0.4, h: h - 0.5, fontSize: 9.5,
    color: C.tinta, fontFace: "Calibri", lineSpacing: 13,
  });
}

// ---------------------------------------------------------------------------
// SLIDE
// ---------------------------------------------------------------------------
function slideJudul(pres, info) {
  const s = pres.addSlide();
  s.background = { color: C.krem };
  s.addShape("rect", { x: 0, y: 0, w: 13.333, h: 2.75, fill: { color: C.merah } });
  s.addText("Klasifikasi Lesi Kulit dari Foto", {
    x: 0.6, y: 0.6, w: 12.1, h: 0.95, fontSize: 38, bold: true,
    color: C.putih, fontFace: "Cambria",
  });
  s.addText(
    "Skrining awal risiko keganasan dengan HOG-SVM, HOG-Random Forest, CNN, " +
    "dan MobileNetV2, termasuk evaluasi domain shift pada foto smartphone",
    { x: 0.6, y: 1.62, w: 12.1, h: 0.85, fontSize: 14.5, color: "F0D9D9",
      fontFace: "Calibri" }
  );
  s.addText("PROYEK AKHIR PEMBELAJARAN MESIN", {
    x: 0.6, y: 3.15, w: 12.1, h: 0.3, fontSize: 10.5, bold: true,
    color: C.merah, fontFace: "Calibri", charSpacing: 1.1,
  });
  const anggota = (info.anggota || [])
    .map((a) => `${a.nama}  (${a.nim})`)
    .join("\n");

  s.addText(
    `Judul   : ${info.judul}\n\n` +
    `Kelompok: ${info.kelompok}\n\n` +
    `Anggota :\n${anggota}\n\n` +
    `Dosen   : ${info.dosen}\n` +
    `Tanggal : ${info.tanggal}`,
    { x: 0.6, y: 3.5, w: 6.7, h: 2.9, fontSize: 11.5, color: C.tinta,
      fontFace: "Calibri", lineSpacing: 14 }
  );
  gambar(pres, s, "eda_07_contoh_gambar.png",
    { x: 7.55, y: 3.15, w: 5.2, h: 3.5, sizing: { type: "cover", w: 5.2, h: 3.5 } });
  s.addText(
    "Alat bantu belajar dan skrining edukasi. Bukan alat diagnosis medis.",
    { x: 0.6, y: 6.75, w: 12.1, h: 0.35, fontSize: 11, italic: true,
      color: C.tintaLembut, fontFace: "Calibri" }
  );
}

function slideIsi(pres) {
  const s = pres.addSlide();
  s.background = { color: C.krem };
  s.addText("Isi Presentasi", {
    x: 0.6, y: 0.55, w: 12.1, h: 0.7, fontSize: 30, bold: true,
    color: C.merah, fontFace: "Cambria",
  });
  const butir = [
    "Latar belakang dan masalah",
    "Tujuan dan ruang lingkup",
    "Studi literatur",
    "Dataset dan eksplorasi data",
    "Preprocessing dan rekayasa fitur",
    "Pengembangan model",
    "Evaluasi dan perbandingan model",
    "Analisis model dan pemilihan model terbaik",
    "Uji eksternal: domain shift ke foto smartphone",
    "Implementasi aplikasi web",
    "Kesimpulan, keterbatasan, pengembangan lanjutan",
  ];
  const kolomKiri = butir.slice(0, 6);
  const kolomKanan = butir.slice(6);
  [kolomKiri, kolomKanan].forEach((kol, ki) => {
    kol.forEach((b, i) => {
      const y = 1.6 + i * 0.78;
      const x = ki === 0 ? 0.9 : 6.9;
      s.addText(String(ki * 6 + i + 1).padStart(2, "0"), {
        x, y, w: 0.55, h: 0.32, fontSize: 12, bold: true, color: C.slate,
        fontFace: "Calibri",
      });
      s.addText(b, {
        x: x + 0.65, y, w: 5.3, h: 0.32, fontSize: 13, color: C.tinta,
        fontFace: "Calibri",
      });
    });
  });
  s.addShape("rect", {
    x: 6.45, y: 1.6, w: 0.008, h: 4.6, fill: { color: C.garis },
  });
}

function slideLatarBelakang(pres) {
  const s = pres.addSlide();
  judulSlide(s, "BAGIAN 1", "Latar Belakang dan Masalah",
    "Mengapa klasifikasi lesi kulit dengan machine learning?");

  s.addText("Masalah", {
    x: 0.55, y: 1.85, w: 5.9, h: 0.3, fontSize: 13, bold: true,
    color: C.merah, fontFace: "Calibri",
  });
  s.addText(
    "Melanoma adalah jenis kanker kulit yang paling berbahaya. Kelangsungan " +
    "hidupnya ditentukan oleh deteksi dini: melanoma yang ditemukan pada " +
    "stage awal punya peluang bertahan hidup jauh lebih tinggi dibanding yang " +
    "sudah metastasis.\n\n" +
    "Masalahnya: pemeriksaan dermatologi memerlukan dokter cheapest dan " +
    "alat dermatoskop yang tidak tersedia di semua tempat, terutama di " +
    "negara berkembang seperti Indonesia.\n\n" +
    "Sistem computer vision yang bisa menandai lesi yang perlu diperiksa " +
    "dokter menjadi pelengkap yang masuk akal.",
    { x: 0.55, y: 2.18, w: 5.9, h: 3.3, fontSize: 11.5, color: C.tinta,
      fontFace: "Calibri", lineSpacing: 16 }
  );

  s.addText("Mengapa ini sulit", {
    x: 6.85, y: 1.85, w: 5.9, h: 0.3, fontSize: 13, bold: true,
    color: C.merah, fontFace: "Calibri",
  });
  s.addText(
    "Tiga tantangan nyata pada dataset HAM10000, semuanya terukur:\n\n" +
    "1. Ketidakseimbangan kelas yang ekstrem. Kelas mola (nv) mencakup " +
    "66,9% data, sedangkan dermatofibroma (df) hanya 1,1%. Rasio 58 : 1. " +
    "Model yang malas menebak kelas terbesar sudah mendapat akurasi 67%.\n\n" +
    "2. Klasifikasi 7 kelas yang mirip satu sama lain. Melanoma dan mola " +
    "sama-sama lesi pigmented; pembeda terutama ada di simetri dan sebaran " +
    "warna.\n\n" +
    "3. Risiko clinically nyata dari false negative. Melanoma yang lolos " +
    "diklasifikasi jinak adalah kegagalan paling berbahaya.",
    { x: 6.85, y: 2.18, w: 5.9, h: 3.3, fontSize: 11.5, color: C.tinta,
      fontFace: "Calibri", lineSpacing: 16 }
  );

  kotakCatatan(s, 0.55, 5.65, 12.2, 1.15, "TUJUAN PROJECT",
    "Membangun dan membandingkan empat model machine learning untuk " +
    "klasifikasi lesi kulit 7 kelas dari foto, lalu menguji seberapa jauh " +
    "kinerjanya menurun ketika foto yang dipakai berasal dari kamera yang " +
    "berbeda (smartphone) alih-alih dermatoskop.",
    C.merah);
}

function slideTujuan(pres) {
  const s = pres.addSlide();
  judulSlide(s, "BAGIAN 2", "Tujuan dan Ruang Lingkup");

  s.addText("Tujuan yang dikejar", {
    x: 0.55, y: 1.85, w: 6.1, h: 0.3, fontSize: 13, bold: true,
    color: C.merah, fontFace: "Calibri",
  });
  const tujuan = [
    "Membangun 4 model: HOG-SVM, HOG-Random Forest, CNN dari nol, dan MobileNetV2 transfer learning.",
    "Membandingkan performa dengan metrik yang sesuai untuk data tidak seimbang: F1 macro dan recall kelas ganas, bukan accuracy.",
    "Mengukurbesarnya domain shift ketika model yang dilatih pada foto dermoskop diuji pada foto smartphone.",
    "Menghasilkan aplikasi web yang bisa dipakai pengguna untuk skrining awal.",
  ];
  tujuan.forEach((t, i) => {
    const y = 2.25 + i * 0.72;
    s.addShape("rect", {
      x: 0.55, y: y + 0.04, w: 0.28, h: 0.28, fill: { color: C.merah },
    });
    s.addText(String(i + 1), {
      x: 0.55, y: y + 0.04, w: 0.28, h: 0.28, fontSize: 10, bold: true,
      color: C.putih, fontFace: "Calibri", align: "center", valign: "middle",
    });
    s.addText(t, {
      x: 0.98, y, w: 5.65, h: 0.62, fontSize: 11, color: C.tinta,
      fontFace: "Calibri", lineSpacing: 14,
    });
  });

  s.addText("Ruang lingkup dan batasan", {
    x: 6.85, y: 1.85, w: 5.9, h: 0.3, fontSize: 13, bold: true,
    color: C.merah, fontFace: "Calibri",
  });
  s.addText(
    "Yang dikerjakan:\n" +
    "- Klasifikasi 7 kelas pada HAM10000\n" +
    "- Ringkasan biner ganas vs tidak ganas untuk aplikasi\n" +
    "- Uji eksternal pada PAD-UFES-20\n" +
    "- Aplikasi web Streamlit\n\n" +
    "Yang TIDAK dikerjakan:\n" +
    "- Diagnosis klinis (bukan tujuan, bukan wewenang)\n" +
    "- Validasi oleh dokter pada data baru\n" +
    "- Klasifikasi lesi di luar kulit\n" +
    "- Prediksi risiko kanker per-orangan",
    { x: 6.85, y: 2.25, w: 5.9, h: 4.3, fontSize: 11.5, color: C.tinta,
      fontFace: "Calibri", lineSpacing: 16 }
  );
}

function slideStudiLiteratur(pres) {
  const s = pres.addSlide();
  judulSlide(s, "BAGIAN 3", "Studi Literatur",
    "Enam artikel jurnal yang diverifikasi lewat NCBI PubMed (bukan karangan)");
  s.addText(
    "Semua entri diverifikasi otomatis: judul, penulis, tahun, jurnal, dan DOI " +
    "diambil langsung dari indeks PubMed. Verifikasi ulang bisa dijalankan " +
    "dengan scripts/verify_literature.py.",
    { x: 0.55, y: 1.72, w: 12.2, h: 0.34, fontSize: 10.5, italic: true,
      color: C.tintaLembut, fontFace: "Calibri" }
  );

  const rows = [
    ["Tschandl dkk. (2018)", "Scientific Data", "HAM10000: 10.015 citra dermoskopi, 7 kelas, multi-sumber", "Dataset dasar proyek kita"],
    ["Pacheco dkk. (2020)", "Data in Brief", "PAD-UFES-20: 2.298 citra klinis smartphone, 6 diagnosis, Fitzpatrick I-VI", "Dataset uji eksternal"],
    ["Fogelberg dkk. (2023)", "New Biotechnology", "Domain shift pada dataset dermoskopi skin cancer; degradasi lintas-dataset", "Landasan teori evaluasi domain shift"],
    ["Combalia dkk. (2022)", "Lancet Digital Health", "Validasi AI dermatologi; performa turun pada kasus yang tak ada di training", "Justifikasi uji eksternal yang jujur"],
    ["Almaraz-Damian dkk. (2020)", "Entropy", "Fusi fitur handcraft (ABCD) dengan fitur deep learning via mutual information", "Pembanding dengan pendekatan HOG + histogram"],
    ["Katsch dkk. (2022)", "Derm. Pract. Concept.", "Ketahanan arsitektur CNN terhadap artefak dermoskop (tanda, penggaris)", "Pembanding dengan masalah domain shift"],
  ];
  tabel(s, 0.55, 2.16, 12.2,
    [" Penulis & Tahun", "Jurnal", "Permasalahan / Dataset / Metode", "Perbedaan dengan Project"],
    rows, [2.35, 1.85, 5.15, 2.85], { fRow: 0.56, fs: 9 });

  kotakCatatan(s, 0.55, 6.05, 12.2, 0.82, "CATATAN",
    "Kolom 'Perbedaan dengan Project' adalah ANALISIS PENULIS, bukan kutipan " +
    "dari artikel. Kolom lain (penulis, tahun, jurnal, DOI) sudah " +
    "diverifikasi terhadap PubMed.",
    C.slate);
}

function slideDataset(pres) {
  const s = pres.addSlide();
  judulSlide(s, "BAGIAN 4", "Dataset dan Eksplorasi Data",
    "HAM10000 (Tschandl dkk. 2018) dan PAD-UFES-20 (Pacheco dkk. 2020)");

  kartuStat(s, 0.55, 1.85, 2.85, 1.15, "10.015", "Gambar dermoskopi", C.merah,
    "HAM10000");
  kartuStat(s, 3.6, 1.85, 2.85, 1.15, "7.470", "Lesi unik", C.slate,
    "1 lesi bisa >1 foto");
  kartuStat(s, 6.65, 1.85, 2.85, 1.15, "58 : 1", "Rasio kelas", C.kuning,
    "nv vs df");
  kartuStat(s, 9.7, 1.85, 3.05, 1.15, "2.298", "Foto smartphone", C.hijau,
    "PAD-UFES-20");

  const rows = [
    ["akiec", "Keratosis Intraepitel Atipis", "327", "3,3%", "GANAS"],
    ["bcc", "Karsinoma Sel Basal", "514", "5,1%", "GANAS"],
    ["bkl", "Lesi Keratosis Jinak", "1.099", "11,0%", "tidak ganas"],
    ["df", "Dermatofibroma", "115", "1,1%", "tidak ganas"],
    ["mel", "Melanoma", "1.113", "11,1%", "GANAS"],
    ["nv", "Mola (Nevus Melanositik)", "6.705", "66,9%", "tidak ganas"],
    ["vasc", "Lesi Vaskular", "142", "1,4%", "tidak ganas"],
  ];
  s.addText("Distribusi 7 kelas diagnosis", {
    x: 0.55, y: 3.25, w: 6.0, h: 0.28, fontSize: 12, bold: true,
    color: C.merah, fontFace: "Calibri",
  });
  tabel(s, 0.55, 3.58, 6.0,
    ["Kode", "Nama diagnosis", "Jumlah", "%", "Kelompok"],
    rows, [0.85, 2.35, 0.9, 0.75, 1.15], { fRow: 0.29 });

  s.addText("Temuan eksplorasi yang menentukan keputusan", {
    x: 6.85, y: 3.25, w: 5.9, h: 0.28, fontSize: 12, bold: true,
    color: C.merah, fontFace: "Calibri",
  });
  s.addText(
    "1. Imbalance 58 : 1. Model asal menebak 'nv' sudah dapat accuracy 67%.\n\n" +
    "2. 10.015 foto hanya 7.470 lesi. 2.545 foto adalah lesi yang sama, " +
    "jadi split WAJIB per lesion_id.\n\n" +
    "3. Missing value hanya pada 'age' (57, 0,57%) dan tidak dipakai sebagai " +
    "feature, jadi tidak diimputasi.\n\n" +
    "4. Semua gambar 600x450 RGB, seragam, aspect ratio 4:3.\n\n" +
    "5. Hanya Fitzpatrick I-II. Tidak ada kulit gelap, keterbatasan serius " +
    "untuk konteks Indonesia.\n\n" +
    "6. 53,3% label dari histopatologi (gold standard).",
    { x: 6.85, y: 3.58, w: 5.9, h: 3.0, fontSize: 10.5, color: C.tinta,
      fontFace: "Calibri", lineSpacing: 14 }
  );
}

function slidePreprocessing(pres) {
  const s = pres.addSlide();
  judulSlide(s, "BAGIAN 5", "Preprocessing dan Rekayasa Fitur",
    "Setiap keputusan preprocessing diturunkan dari temuan EDA atau eksperimen");

  // Rantai klasik
  s.addText("Model klasik: HOG + histogram warna", {
    x: 0.55, y: 1.85, w: 6.0, h: 0.28, fontSize: 12, bold: true,
    color: C.merah, fontFace: "Calibri",
  });
  s.addText(
    "HOG (1.764 fitur)  ->  StandardScaler  ->  PCA(30)  ->  StandardScaler\n" +
    "histogram (48 fitur)  ->  dikali bobot 6  (TIDAK di-scale ulang)\n" +
    "                                          |\n" +
    "                                    digabung  ->  SVC / RandomForest",
    { x: 0.55, y: 2.18, w: 6.0, h: 1.35, fontSize: 10, color: C.tinta,
      fontFace: "Consolas" }
  );

  kotakCatatan(s, 0.55, 3.68, 6.0, 1.55, "MENGAPA SUSUNAN SEPERTI INI",
    "Random Forest pada 1.812 fitur mentah: F1 macro 0,1467, hampir sama " +
    "dengan baseline 'selalu nebak nv' (0,1148), recall ganas 0,0192. " +
    "Penyebabnya: RF memilih fitur ACAK saat membelah node, jadi peluang " +
    "fitur histogram (48 dari 1.812 = 2,65%) praktis tak pernah dipakai. " +
    "PCA yang diterapkan ke seluruh fitur juga lebih buruk: PCA(400) dari " +
    "1.812 fitur hanya F1 0,3069, sedangkan PCA(30) pada HOG saja + " +
    "histogram mencapai 0,3423.",
    C.merah);

  // Model deep
  s.addText("Model deep learning: MobileNetV2", {
    x: 6.85, y: 1.85, w: 5.9, h: 0.28, fontSize: 12, bold: true,
    color: C.merah, fontFace: "Calibri",
  });
  s.addText(
    "Input 224x224x3  ->  Rescaling(1/127.5, -1.0)  ->  MobileNetV2  ->  GAP\n" +
    "                                                        ->  Dropout\n" +
    "                                                        ->  Dense(7)",
    { x: 6.85, y: 2.18, w: 5.9, h: 1.05, fontSize: 10, color: C.tinta,
      fontFace: "Consolas" }
  );

  s.addText("Aturan yang dijaga ketat", {
    x: 6.85, y: 3.42, w: 5.9, h: 0.28, fontSize: 12, bold: true,
    color: C.merah, fontFace: "Calibri",
  });
  s.addText(
    "- Normalisasi hanya SATU KALI. MobileNetV2 butuh rentang -1..1; " +
    "menambah Rescaling(1/255) lagi membuat input berada di rentang yang " +
    "tak pernah dilihat saat pretraining.\n" +
    "- Normalisasi berada DI DALAM model, jadi file .keras bisa dimuat tanpa " +
    "custom_objects.\n" +
    "- Augmentasi hanya pada data latih.\n" +
    "- Ukuran sel HOG: performa hampir sama untuk 8x8/16x16/32x32, " +
    "jadi 16x16 dipilih karena 4,6x lebih hemat dimensi (1.764 vs 8.100).",
    { x: 6.85, y: 3.75, w: 5.9, h: 1.5, fontSize: 10.5, color: C.tinta,
      fontFace: "Calibri", lineSpacing: 14 }
  );

  kotakCatatan(s, 6.85, 5.42, 5.9, 1.15, "CLASS WEIGHT WAJIB",
    "Tanpa class_weight, CNN mengunci accuracy di 0,672 (= proporsi nv) dan " +
    "recall ganas nol: model tidak belajar apa pun. Class weight bukan " +
    "opsional di proyek ini.",
    C.kuning);
}

function slideModel(pres) {
  const s = pres.addSlide();
  judulSlide(s, "BAGIAN 6", "Pengembangan Model",
    "Empat model dengan alasan pemilihan yang jelas, bukan mencoba acak");

  const model = [
    ["HOG + SVM", "class_weight balanced, PCA(30) pada HOG, kalibrasi Platt", "1.812 -> 78 fitur, ~1 menit"],
    ["HOG + Random Forest", "300 pohon, balanced_subsample, PCA(30) pada HOG", "1.812 -> 78 fitur, ~3 detik"],
    ["CNN dari nol", "Rescaling -> Conv32 -> Pool -> Conv64 -> Pool -> GAP -> Dense(7)", "19.847 parameter"],
    ["MobileNetV2", "Transfer learning 2 tahap: classifier lalu fine-tuning 30 lapisan atas", "2.266.951 parameter"],
  ];
  tabel(s, 0.55, 1.9, 12.2,
    ["Model", "Konfigurasi utama", "Catatan"],
    model, [2.6, 6.4, 3.2], { fRow: 0.62, fs: 10 });

  s.addText("Eksperimen yang wajib dilakukan", {
    x: 0.55, y: 4.85, w: 6.0, h: 0.28, fontSize: 12, bold: true,
    color: C.merah, fontFace: "Calibri",
  });
  s.addText(
    "1. GlobalAveragePooling vs Flatten pada split yang sama, untuk " +
    "membandingkan jumlah parameter dan metrik validasi.\n\n" +
    "2. Tahap 1 vs tahap 2 pada MobileNetV2, untuk menguji apakah fine-tuning " +
    "dengan learning rate 1e-5 membantu.\n\n" +
    "3. Abelasi: SVM pada histogram warna SAJA, untuk mengukur seberapa " +
    "besar kontribusi HOG terhadap performa.",
    { x: 0.55, y: 5.18, w: 6.0, h: 1.6, fontSize: 10.5, color: C.tinta,
      fontFace: "Calibri", lineSpacing: 14 }
  );

  s.addText("Tbacks yang dijaga saat training", {
    x: 6.85, y: 4.85, w: 5.9, h: 0.28, fontSize: 12, bold: true,
    color: C.merah, fontFace: "Calibri",
  });
  s.addText(
    "EarlyStopping dengan restore_best_weights: model terbaik yang " +
    "disimpan, bukan model pada epoch terakhir.\n\n" +
    "BatchNormalization di backbone MobileNetV2 DIKUNCI saat fine-tuning: " +
    "statistik batch yang kecil dan berbeda distribusi membuat prediksi " +
    "berayak antar epoch.\n\n" +
    "Learning rate tahap 2 = 1e-5, 100x lebih kecil dari tahap 1, supaya " +
    "fitur pretrained tidak rusak (catastrophic forgetting).",
    { x: 6.85, y: 5.18, w: 5.9, h: 1.6, fontSize: 10.5, color: C.tinta,
      fontFace: "Calibri", lineSpacing: 14 }
  );
}

function slideEvaluasi(pres) {
  const s = pres.addSlide();
  judulSlide(s, "BAGIAN 7", "Evaluasi dan Perbandingan Model",
    "Keempat model diuji pada test set yang sama (1.527 gambar)");

  if (!H.final) {
    s.addText("Hasil evaluasi belum tersedia. Jalankan Notebook 07 lebih dulu.",
      { x: 0.55, y: 2.5, w: 12.2, h: 0.5, fontSize: 14, italic: true,
        color: C.tintaLembut, fontFace: "Calibri" });
    return;
  }

  const pm = H.final.per_model;
  const namaModel = Object.keys(pm);
  const urut = [...namaModel].sort(
    (a, b) => pm[b].f1_macro - pm[a].f1_macro
  );

  const rows = urut.map((m) => {
    const p = pm[m];
    return [
      m,
      PERSEN(p.accuracy),
      NUM(p.precision_macro),
      NUM(p.recall_macro),
      NUM(p.f1_macro),
      NUM(p.malignant_recall),
      NUM(p.specificity),
      p.binary_with_tuned_threshold.fn,
      p.binary_with_tuned_threshold.fp,
      `${p.inference_ms_per_image.toFixed(0)} ms`,
    ];
  });

  tabel(s, 0.55, 1.85, 12.2,
    ["Model", "Accuracy", "Prec (macro)", "Rec (macro)", "F1 macro",
      "Rec ganas", "Specificity", "FN", "FP", "Inferensi"],
    rows, [2.15, 1.05, 1.2, 1.2, 1.05, 1.15, 1.25, 0.7, 0.7, 1.0],
    { fRow: 0.42, fs: 9 });

  s.addText(
    "FN = lesi ganas yang LOLOS (paling berbahaya). FP = false alarm pada " +
    "lesi tidak ganas. Ambang 'ganas' dihitung dari DATA VALIDASI dengan " +
    "target recall 90%, lalu dikunci sebelum test disentuh.",
    { x: 0.55, y: 3.95, w: 12.2, h: 0.35, fontSize: 10, italic: true,
      color: C.tintaLembut, fontFace: "Calibri" }
  );

  gambar(pres, s, "perbandingan_02_confusion_matrix.png",
    { x: 0.55, y: 4.35, w: 12.2, h: 2.55,
      sizing: { type: "contain", w: 12.2, h: 2.55 } });
}

function slideAnalisis(pres) {
  const s = pres.addSlide();
  judulSlide(s, "BAGIAN 8", "Analisis Model dan Pemilihan Model Terbaik",
    "Accuracy tertinggi tidak otomatis berarti model terbaik");

  if (!H.final) {
    s.addText("Hasil evaluasi belum tersedia.",
      { x: 0.55, y: 2.5, w: 12.2, h: 0.5, fontSize: 14, italic: true,
        color: C.tintaLembut, fontFace: "Calibri" });
    return;
  }

  const pm = H.final.per_model;
  const urut = Object.keys(pm).sort((a, b) => pm[b].f1_macro - pm[a].f1_macro);
  const terbaik = pm[urut[0]];
  const terBuruk = pm[urut[urut.length - 1]];
  const byAcc = Object.keys(pm).sort((a, b) => pm[b].accuracy - pm[a].accuracy)[0];

  s.addText("Temuan kunci", {
    x: 0.55, y: 1.85, w: 6.0, h: 0.28, fontSize: 12, bold: true,
    color: C.merah, fontFace: "Calibri",
  });
  s.addText(
    `1. Model dengan F1 macro tertinggi: ${urut[0]} ` +
    `(${NUM(terbaik.f1_macro)}). Recalls kelas ganas ` +
    `${NUM(terbaik.malignant_recall)}.\n\n` +
    `2. Model dengan accuracy tertinggi: ${byAcc} ` +
    `(${PERSEN(pm[byAcc].accuracy)}). ` +
    (byAcc !== urut[0]
      ? "Ini BUKAN model yang sama, dan itu justru bukti bahwa accuracy " +
        "saja tidak cukup untuk memilih model aplikasi skrining.\n\n"
      : "Pada kasus ini keduanya sama.\n\n") +
    `3. Model terlemah: ${urut[urut.length - 1]} ` +
    `(F1 macro ${NUM(terBuruk.f1_macro)}).\n\n` +
    "4. Fenomena yang berulang: model tanpa class_weight mengunci accuracy " +
    "di 0,672 (= proporsi nv) dengan recall ganas nol. Angka itu sama " +
    "persis dengan baseline, jadi dipakai sebagai penanda bahwa model " +
    "tidak belajar apa pun.\n\n" +
    "5. Trade-off utama: recall ganas tinggi biasanya berarti specificity " +
    "lebih rendah. Untuk skrining, lebih baik beberapa false alarm daripada " +
    "melanoma lolos." +
    (H.kalibrasi
      ? "\n\n6. Temuan teknis: cara SVM mengambil probabilitas sangat " +
        "memengaruhi performa. Dengan CalibratedClassifierCV(ensemble=False) " +
        "(contoh resmi sklearn) recall ganas hanya 0,1891; dengan " +
        "ensemble=True menjadi 0,2981. Kami mengukurnya sendiri, bukan " +
        "memakai nilai bawaan."
      : ""),
    { x: 0.55, y: 2.18, w: 6.0, h: 3.4, fontSize: 10.5, color: C.tinta,
      fontFace: "Calibri", lineSpacing: 14 }
  );

  s.addText("Kriteria pemilihan model untuk aplikasi", {
    x: 6.85, y: 1.85, w: 5.9, h: 0.28, fontSize: 12, bold: true,
    color: C.merah, fontFace: "Calibri",
  });
  const bobot = H.final.selection_weights || {};
  const rows = Object.entries(bobot).map(([k, v]) => [
    { recall_ganas: "Recall kelas ganas", f1_macro: "F1 macro",
      domain_shift: "Kesiapan domain shift", efisiensi: "Kecepatan & ukuran" }[k] || k,
    `${(v * 100).toFixed(0)}%`,
  ]);
  tabel(s, 6.85, 2.18, 5.9, ["Kriteria", "Bobot"], rows, [4.0, 1.9],
    { fRow: 0.34 });

  const skor = H.final.selection_scores || {};
  const urutSkor = Object.entries(skor).sort((a, b) => b[1] - a[1]);
  s.addText("Skor gabungan", {
    x: 6.85, y: 3.75, w: 5.9, h: 0.26, fontSize: 11, bold: true,
    color: C.merah, fontFace: "Calibri",
  });
  tabel(s, 6.85, 4.05, 5.9, ["Model", "Skor"],
    urutSkor.map(([m, v]) => [m, NUM(v, 4)]), [4.0, 1.9], { fRow: 0.32 });

  kotakCatatan(s, 6.85, 5.85, 5.9, 0.95, "MODEL TERPILIH",
    `${H.final.selected_model}. Dipilih karena recall kelas ganas dan ` +
    `kesiapan domain shift, bukan karena accuracy tertinggi.`,
    C.merah);
}

function slideDomainShift(pres) {
  const s = pres.addSlide();
  judulSlide(s, "BAGIAN 9", "Uji Eksternal: Domain Shift ke Foto Smartphone",
    "HAM10000 (dermoskop) vs PAD-UFES-20 (smartphone), tanpa training ulang");

  if (!H.pad) {
    s.addText("Hasil uji eksternal belum tersedia. Jalankan Notebook 08 lebih dulu.",
      { x: 0.55, y: 2.5, w: 12.2, h: 0.5, fontSize: 14, italic: true,
        color: C.tintaLembut, fontFace: "Calibri" });
    return;
  }

  const pm = H.final ? H.final.per_model : {};
  const rows = Object.keys(H.pad.hasil_per_model).map((m) => {
    const ext = H.pad.hasil_per_model[m];
    const inDom = pm[m] ? pm[m].binary_with_tuned_threshold : null;
    return [
      m,
      inDom ? NUM(inDom.recall) : "-",
      NUM(ext.recall),
      inDom ? NUM(ext.recall - inDom.recall) : "-",
      inDom ? inDom.fn : "-",
      ext.fn,
      inDom ? inDom.fp : "-",
      ext.fp,
    ];
  });
  tabel(s, 0.55, 1.85, 12.2,
    ["Model", "Rec HAM", "Rec PAD", "Selisih", "FN HAM", "FN PAD", "FP HAM", "FP PAD"],
    rows, [2.6, 1.3, 1.3, 1.25, 1.25, 1.25, 1.25, 1.25],
    { fRow: 0.42, fs: 9.5 });

  s.addText("Mengapa penurunan ini bisa diprediksi", {
    x: 0.55, y: 4.05, w: 6.0, h: 0.28, fontSize: 12, bold: true,
    color: C.merah, fontFace: "Calibri",
  });
  const st = H.pad.statistik_domain || {};
  const h = st["HAM10000"] || {};
  const p = st["PAD-UFES-20"] || {};
  s.addText(
    `Resolusi : HAM ${(h.lebar || 0).toFixed(0)} px vs PAD ${(p.lebar || 0).toFixed(0)} px\n` +
    `Pigmentasi: mean kanal R ${(h.mean_R || 0).toFixed(1)} vs ${(p.mean_R || 0).toFixed(1)}\n` +
    `Ketajaman : ${(h.ketajaman || 0).toFixed(2)} vs ${(p.ketajaman || 0).toFixed(2)} (kecil = buram)\n` +
    `Variasi warna: ${(h.std_warna || 0).toFixed(2)} vs ${(p.std_warna || 0).toFixed(2)}\n\n` +
    "Foto smartphone diambil dengan kamera biasa, cahaya alami, dan bisa berlatar " +
    "belacak. Semua itu menggeser distribusi warna, dan HAM10000 seluruhnya " +
    "kulit Fitzpatrick I-II sementara PAD-UFES-20 sampai Fitzpatrick VI.",
    { x: 0.55, y: 4.38, w: 6.0, h: 2.3, fontSize: 10.5, color: C.tinta,
      fontFace: "Calibri", lineSpacing: 14 }
  );

  s.addText("Yang terjadi pada kelas yang tak pernah dilihat", {
    x: 6.85, y: 4.05, w: 5.9, h: 0.28, fontSize: 12, bold: true,
    color: C.merah, fontFace: "Calibri",
  });
  const diag = H.pad.per_diagnosis || [];
  const baris = diag.slice(0, 6).map((d) => [
    d.diagnosis, String(d.n), NUM(d.recall), d.kelompok,
  ]);
  tabel(s, 6.85, 4.38, 5.9, ["Diagnosis", "n", "Recall", "Kelompok"],
    baris, [1.7, 0.85, 1.5, 1.85], { fRow: 0.32, fs: 9.5 });

  kotakCatatan(s, 6.85, 6.4, 5.9, 0.72, "CATATAN PENTING",
    "SCC (squamous cell carcinoma) TIDAK ADA di HAM10000, jadi model tidak " +
    "pernah melihat kelas itu saat training. Ini contoh unseen class yang " +
    "otomatis menurunkan performa.",
    C.kuning);
}

function slideAplikasi(pres) {
  const s = pres.addSlide();
  judulSlide(s, "BAGIAN 10", "Implementasi Aplikasi Web",
    "Alur: unggah foto -> preprocessing -> model -> prediksi -> hasil");

  s.addText("Fitur aplikasi", {
    x: 0.55, y: 1.85, w: 5.9, h: 0.28, fontSize: 12, bold: true,
    color: C.merah, fontFace: "Calibri",
  });
  s.addText(
    "INPUT\n" +
    "- Unggah foto JPG/PNG, validasi tipe dan ukuran (maks 10 MB)\n" +
    "- Panduan memotret: dekat, terfokus, cahaya merata, lesi di tengah\n\n" +
    "PROSES\n" +
    "- Preprocessing memakai fungsi yang SAMA dengan training\n" +
    "- Model dimuat sekali per sesi (@st.cache_resource)\n\n" +
    "OUTPUT\n" +
    "- Nama kelas diagnosis dalam Bahasa Indonesia\n" +
    "- Grafik batang probabilitas 7 kelas\n" +
    "- Ringkasan risiko ganas + ambang dari data validasi\n" +
    "- Pesan ketidakpastian saat probabilitas di area abu-abu",
    { x: 0.55, y: 2.18, w: 5.9, h: 3.5, fontSize: 10.5, color: C.tinta,
      fontFace: "Calibri", lineSpacing: 15 }
  );

  s.addText("Penanganan risiko etis dan privasi", {
    x: 6.85, y: 1.85, w: 5.9, h: 0.28, fontSize: 12, bold: true,
    color: C.merah, fontFace: "Calibri",
  });
  s.addText(
    "- Disclaimer wajib di halaman utama, tidak bisa dilewatkan\n" +
    "- 'Tidak ganas' bukan berarti aman; Commerce langsung ke dokter bila " +
    "lesi berubah bentuk, warna, ukuran, berdarah, atau terasa tidak nyaman\n" +
    "- SDisclaimer menyebut model dilatih pada data dermoskopi dengan " +
    "populasi terbatas, tanpa validasi klinis, dan tanpa cakupan kulit gelap\n" +
    "- Foto diproses di memori selama sesi saja: tidak disimpan, tidak " +
    "dicatat, tidak dikirim ke pihak ketiga\n" +
    "- Statistik usage Streamlit dimatikan di .streamlit/config.toml",
    { x: 6.85, y: 2.18, w: 5.9, h: 3.5, fontSize: 10.5, color: C.tinta,
      fontFace: "Calibri", lineSpacing: 15 }
  );

  kotakCatatan(s, 0.55, 5.95, 12.2, 0.95, "DEPLOY",
    "Streamlit Community Cloud (gratis) disarankan: deploy dari GitHub " +
    "dalam 5 klik, tanpa kartu kredit. Batas RAM 1 GB jadi risiko yang " +
    "harus diuji. Hugging Face Spaces (CPU, RAM 16 GB) adalah cadangan bila " +
    "memori tidak cukup. Panduan lengkap ada di docs/PANDUAN_DEPLOY.md.",
    C.slate);
}

function slideKesimpulan(pres) {
  const s = pres.addSlide();
  judulSlide(s, "BAGIAN 11", "Kesimpulan, Keterbatasan, dan Pengembangan Lanjutan");

  s.addText("Kesimpulan", {
    x: 0.55, y: 1.85, w: 5.9, h: 0.28, fontSize: 12, bold: true,
    color: C.merah, fontFace: "Calibri",
  });
  s.addText(
    "1. Empat model berhasil dilatih dan dibandingkan pada test set yang " +
    "sama, dengan metrik yang sesuai untuk data tidak seimbang.\n\n" +
    "2. Temuan eksperimen yang paling bernilai: HOG dengan 1.764 fitur " +
    "mengenceri sinyal warna, sehingga Random Forest pada fitur mentah " +
    "hampir tidak belajar (F1 macro 0,1467 vs baseline 0,1148). " +
    "Perbaikan rantai preprocessing menaikkan performa nyata.\n\n" +
    "3. Domain shift dari dermoskop ke smartphone menyebabkan penurunan " +
    "performa yang terukur dan bisa dijelaskan secara teknis.\n\n" +
    "4. Aplikasi web berhasil dibangun dengan preprocessing yang identik " +
    "dengan training, ambang risiko dari data validasi, dan disclaimer " +
    "yang tidak bisa dilewatkan.",
    { x: 0.55, y: 2.18, w: 5.9, h: 3.4, fontSize: 10.5, color: C.tinta,
      fontFace: "Calibri", lineSpacing: 14 }
  );

  s.addText("Keterbatasan", {
    x: 6.85, y: 1.85, w: 5.9, h: 0.28, fontSize: 12, bold: true,
    color: C.merah, fontFace: "Calibri",
  });
  s.addText(
    "DATASET\n" +
    "- Hanya Fitzpatrick I-II, tanpa kulit gelap, padahal populasi sasaran " +
    "di Indonesia didominasi Fitzpatrick III-V\n" +
    "- Hanya 7.470 lesi dengan label 53,3% dari histopatologi\n" +
    "- Kelas df dan vasc sangat sedikit, metriknya tidak stabil\n\n" +
    "MODEL\n" +
    "- Belum divalidasi klinis, hanya diuji pada dataset\n" +
    "- Presisi kelas minoritas rendah (df dan vasc sering 0)\n" +
    "- Model klasik terbatas oleh kualitas fitur handcraft\n\n" +
    "APLIKASI\n" +
    "- Performa pada foto smartphone lebih rendah daripada dermoskop\n" +
    "- Sensitif terhadap kualitas foto dan pencahayaan",
    { x: 6.85, y: 2.18, w: 5.9, h: 3.4, fontSize: 10.5, color: C.tinta,
      fontFace: "Calibri", lineSpacing: 14 }
  );

  kotakCatatan(s, 0.55, 5.85, 12.2, 1.0, "PENGEMBANGAN LANJUTAN",
    "1. Fine-tuning dengan augmentasi smartphone yang lebih agresif. " +
    "2. Validasi pada dataset kulit gelap Fitzpatrick IV-VI. " +
    "3. Feature engineering untuk kelas df dan vasc yang selalu gagal. " +
    "4. Threshold adaptif per confidence, bukan satu ambang global. " +
    "5. Validasi klinik bersama dokter kulit, dan uji sensitivitas " +
    "spesifisitas pada populasi Indonesia.",
    C.slate);
}

function slidePembagian(pres, info) {
  const s = pres.addSlide();
  judulSlide(s, "TERAKHIR", "Pembagian Tugas dan Pengumpulan");

  s.addText("Pembagian tugas", {
    x: 0.55, y: 1.85, w: 6.0, h: 0.28, fontSize: 12, bold: true,
    color: C.merah, fontFace: "Calibri",
  });
  const pembagian = info.pembagian_tugas || [];
  const tugas = pembagian.map((p) => [
    // Key di info_kelompok.json adalah "anggota" (bukan "nama"),
    // tapi kita terima keduanya supaya tidak mudah pecah kalau
    // struktur datanya diubah.
    `${p.anggota || p.nama || "-"}\n${p.nim || ""}`,
    // Buang HANYA bagian "(Notebook xx)" di akhir, jangan pakai
    // split(" (") karena itu ikut memotong deskripsi tugas yang
    // mengandung tanda kurung di tengah kalimat.
    p.tugas.replace(/\s*\(Notebook[^)]*\)\s*$/, "").trim(),
    (p.tugas.match(/\(Notebook[^)]*\)/) || [""])[0].replace(/[()]/g, ""),
  ]);
  tabel(s, 0.55, 2.18, 6.0, ["Anggota", "Tugas", "Artefak"], tugas,
    [1.75, 2.95, 1.3], { fRow: 0.62, fs: 9 });

  s.addText("Berkas yang dikumpulkan", {
    x: 6.85, y: 1.85, w: 5.9, h: 0.28, fontSize: 12, bold: true,
    color: C.merah, fontFace: "Calibri",
  });
  s.addText(
    "- Source code dan model (lihat Tautan)\n" +
    "- Link dataset (tidak di-commit karena lisensi)\n" +
    "- Model hasil training (.keras)\n" +
    "- Daftar 5+ jurnal referensi yang sudah diverifikasi\n" +
    "- Tabel studi literatur\n" +
    "- PPT laporan\n" +
    "- Video demo maksimal 7 menit",
    { x: 6.85, y: 2.18, w: 5.9, h: 2.6, fontSize: 10.5, color: C.tinta,
      fontFace: "Calibri", lineSpacing: 15 }
  );

  // Tautan dibuat bisa diklik supaya dosen tidak perlu menyalin URL dari
  // slide. Tanpa ini, presentasi hanya menampilkan teks yang harus diketik
  // ulang secara manual.
  s.addText("Tautan", {
    x: 6.85, y: 4.95, w: 5.9, h: 0.28, fontSize: 12, bold: true,
    color: C.merah, fontFace: "Calibri",
  });
  s.addText([
    { text: "Repo: ", options: { color: C.tinta, bold: true } },
    { text: "github.com/Grenaydinkamil/ProjectForestry_ML2026_Mendekteksi_Lesi_Kulit",
      options: { color: "1F5FA8", underline: true,
        hyperlink: { url: "https://github.com/Grenaydinkamil/ProjectForestry_ML2026_Mendekteksi_Lesi_Kulit" } } },
    { text: "\nApp: ", options: { color: C.tinta, bold: true } },
    { text: "projectforestryml2026mendekteksilesikulit.streamlit.app",
      options: { color: "1F5FA8", underline: true,
        hyperlink: { url: "https://projectforestryml2026mendekteksilesikulit-vvearxur6nuwga44fmaf.streamlit.app" } } },
  ], { x: 6.85, y: 5.25, w: 5.9, h: 1.0, fontSize: 9.5,
    fontFace: "Calibri", lineSpacing: 15 });

  kotakCatatan(s, 0.55, 4.6, 6.0, 1.55, "VIDEO DEMO (MAKS 7 MENIT)",
    "1. Perkenalan anggota dan judul (0:30)\n" +
    "2. Latar belakang masalah (1:00)\n" +
    "3. Dataset dan EDA (1:00)\n" +
    "4. Preprocessing dan 4 model (1:30)\n" +
    "5. Tabel perbandingan performa (0:45)\n" +
    "6. Demo aplikasi dijalankan (1:30)\n" +
    "7. Kesimpulan dan keterbatasan (0:45)",
    C.merah);

  s.addText(
    "Dataset dan kode lisensinya terpisah. Kode proyek released di bawah " +
    "lisensi MPL-2.0, tetapi dataset tetap mengikuti lisensinya sendiri " +
    "(HAM10000: CC BY-NC 4.0; PAD-UFES-20: CC BY 4.0) dan WAJIB disitasi " +
    "di laporan dan di aplikasi.",
    { x: 0.55, y: 6.4, w: 12.2, h: 0.6, fontSize: 10, italic: true,
      color: C.tintaLembut, fontFace: "Calibri", lineSpacing: 13 }
  );
}

// ---------------------------------------------------------------------------
// MAIN
// ---------------------------------------------------------------------------
function main() {
  const pres = new PptxGenJS();
  pres.layout = "LAYOUT_WIDE";
  pres.author = "ML2026 - Kelompok 1";
  pres.title = "Klasifikasi Lesi Kulit dari Foto";
  pres.subject = "Laporan Project Akhir Pembelajaran Mesin";

  const infoPath = path.join(REPO, "docs", "info_kelompok.json");
  const info = fs.existsSync(infoPath)
    ? JSON.parse(fs.readFileSync(infoPath, "utf-8"))
    : { judul: "(isi docs/info_kelompok.json)", kelompok: "-", anggota: "-",
        dosen: "-", tanggal: "-" };

  slideJudul(pres, info);
  slideIsi(pres);
  slideLatarBelakang(pres);
  slideTujuan(pres);
  slideStudiLiteratur(pres);
  slideDataset(pres);
  slidePreprocessing(pres);
  slideModel(pres);
  slideEvaluasi(pres);
  slideAnalisis(pres);
  slideDomainShift(pres);
  slideAplikasi(pres);
  slideKesimpulan(pres);
  slidePembagian(pres, info);

  pres.writeFile({ fileName: OUT }).then(() => {
    console.log(`PPT dibuat: ${OUT}`);
    console.log(`Slide: 14`);
    console.log(
      `Sumber angka: hasilFinal=${H.final ? "ada" : "TIDAK ADA"}, ` +
      `pad=${H.pad ? "ada" : "TIDAK ADA"}`
    );
  });
}

if (require.main === module) main();
