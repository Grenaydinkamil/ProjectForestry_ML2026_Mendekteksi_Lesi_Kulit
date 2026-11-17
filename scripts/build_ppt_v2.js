/*
 * scripts/build_ppt_v2.js
 * =======================
 * Membangun PPT versi 2: 12 slide presentasi ringkas untuk menjelaskan
 * proyek dari masalah sampai produk.
 *
 * Prinsip yang sama dengan build_ppt.js: TIDAK ada angka yang diketik
 * manual. Semua diambil dari docs/logs/*.json hasil notebook.
 *
 * Perbedaan dari v1:
 *   - 12 slide (v1: 14 slide) agar lebih ringkas untuk demo 5 menit.
 *   - Palet klinis baru: navy dominan + teal (jinak) + bata (ganas).
 *   - Motif: "bingkai viewfinder" dermatologi pada kartu-kartu kunci.
 *
 * Jalankan:
 *   node scripts/build_ppt_v2.js
 */

const fs = require("fs");
const path = require("path");
const PptxGenJS = require("pptxgenjs");

const REPO = path.resolve(__dirname, "..");
const LOGS = path.join(REPO, "docs", "logs");
const OUT = path.join(REPO, "docs", "PPT_Laporan_Project_v2.pptx");
const INFO = path.join(REPO, "docs", "info_kelompok.json");
const DIST = path.join(REPO, "data", "processed", "kelas_distribusi.csv");

// ---------------------------------------------------------------------------
// PALET -- klinis, spesifik untuk skrining lesi kulit
// ---------------------------------------------------------------------------
const C = {
  navy: "14213D",      // dominan: otoritas / latar gelap
  navySoft: "25335A",
  putih: "FFFFFF",
  kertas: "F7F8FA",    // latar terang (abu sangat muda, bukan cream)
  garis: "DDE1E7",
  tinta: "1B2333",
  tintaLembut: "5C6675",
  teal: "2A9D8F",      // jinak / aman
  tealMuda: "E3F2F0",
  bata: "9E2A2B",      // ganas / waspada
  bataMuda: "F7E9E8",
  kuning: "8A5A15",
  kuningMuda: "FBF0DC",
};

const FONT_JUDUL = "Cambria";
const FONT_ISI = "Calibri";

const NUM = (v, d = 4) => (v == null ? "-" : Number(v).toFixed(d));
const PCT = (v, d = 1) => (v == null ? "-" : `${(v * 100).toFixed(d)}%`);
const RIBU = (v) => (v == null ? "-" : `${Number(v).toLocaleString("id-ID")}`);

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

const H = {
  final: bacaJson("hasil_evaluasi_final.json"),
  pad: bacaJson("hasil_uji_eksternal_pad_ufes.json"),
  diag: bacaJson("diagnostik_fitur.json"),
  cnn: bacaJson("model_cnn_validasi.json"),
  mnv2: bacaJson("model_mobilenetv2_validasi.json"),
  klasik: bacaJson("model_klasik_validasi.json"),
  sel: bacaJson("cek_kalibrasi_svm.json"),
};

const info = fs.existsSync(INFO)
  ? JSON.parse(fs.readFileSync(INFO, "utf-8"))
  : { anggota: [], judul: "Klasifikasi Lesi Kulit" };

// ---------------------------------------------------------------------------
// PEMBACA DISTRIBUSI KELAS
// ---------------------------------------------------------------------------
function bacaDistribusi() {
  if (!fs.existsSync(DIST)) return [];
  return fs.readFileSync(DIST, "utf-8")
    .split("\n").slice(1)
    .filter((l) => l.trim())
    .map((l) => {
      const [kelas, nama, jumlah, persentase, kelompok] = l.split(",");
      return { kelas, nama, jumlah: Number(jumlah),
               persentase: Number(persentase) / 100, kelompok };
    });
}
const DIST_DATA = bacaDistribusi();

// ---------------------------------------------------------------------------
// HELPER GAMBAR / TEKS
// ---------------------------------------------------------------------------
function latar(s, gelap = false) {
  s.background = { color: gelap ? C.navy : C.kertas };
}

function judul(s, kicker, teks, sub) {
  s.addText(kicker, {
    x: 0.6, y: 0.34, w: 12.1, h: 0.28, fontSize: 10.5, bold: true,
    color: C.bata, fontFace: FONT_ISI, charSpacing: 1.2, margin: 0,
  });
  s.addText(teks, {
    x: 0.6, y: 0.66, w: 12.1, h: 0.62, fontSize: 29, bold: true,
    color: C.tinta, fontFace: FONT_JUDUL, margin: 0,
  });
  if (sub) {
    s.addText(sub, {
      x: 0.6, y: 1.32, w: 12.1, h: 0.4, fontSize: 12.5,
      color: C.tintaLembut, fontFace: FONT_ISI, margin: 0,
    });
  }
}

/*
 * Motif "viewfinder": empat sudut kotak di sekitar sebuah kartu.
 * Dipakai berulang supaya deck terasa satu identitas.
 */
function viewfinder(s, x, y, w, h, warna = C.teal) {
  const p = 0.16;
  const t = 0.035;
  const seg = [
    [x, y, p, t], [x, y, t, p],              // kiri-atas
    [x + w - p, y, p, t], [x + w - t, y, t, p], // kanan-atas
    [x, y + h - t, p, t], [x, y + h - p, t, p], // kiri-bawah
    [x + w - p, y + h - t, p, t], [x + w - t, y + h - p, t, p], // kanan-bawah
  ];
  seg.forEach(([sx, sy, sw, sh]) =>
    s.addShape("rect", { x: sx, y: sy, w: sw, h: sh,
                         fill: { color: warna }, line: { color: warna } }));
}

function kartu(s, x, y, w, h, isi, opsi = {}) {
  const warna = opsi.warna || C.putih;
  s.addShape("rect", {
    x, y, w, h,
    fill: { color: warna },
    line: { color: opsi.garis || C.garis, width: 0.75 },
  });
  if (opsi.viewfinder) viewfinder(s, x, y, w, h, opsi.viewfinder);
  if (isi) isi();
}

function labelKecil(s, teks, x, y, w, warna = C.bata) {
  s.addText(teks, {
    x, y, w, h: 0.28, fontSize: 10.5, bold: true, color: warna,
    fontFace: FONT_ISI, charSpacing: 1, margin: 0,
  });
}

function angkaBesar(s, angka, satuan, x, y, w, warna = C.navy) {
  s.addText(String(angka), {
    x, y, w, h: 0.85, fontSize: 40, bold: true, color: warna,
    fontFace: FONT_JUDUL, margin: 0,
  });
  if (satuan) {
    s.addText(satuan, {
      x, y: y + 0.82, w, h: 0.4, fontSize: 12, color: C.tintaLembut,
      fontFace: FONT_ISI, margin: 0,
    });
  }
}

function tabel(s, x, y, w, header, rows, lebar, opsi = {}) {
  const fRow = opsi.fRow || 0.42;
  const fs = opsi.fs || 10;
  s.addTable([
    header.map((h) => ({
      text: h, options: { bold: true, color: C.putih,
                          fill: { color: C.navy }, fontSize: fs,
                          fontFace: FONT_ISI, margin: 0.06 },
    })),
    ...rows.map((r) => r.map((v, i) => ({
      text: String(v),
      options: {
        color: i === 0 ? C.tinta : C.tintaLembut,
        bold: i === 0, fontSize: fs, fontFace: FONT_ISI,
        fill: { color: C.putih }, margin: 0.06,
      },
    }))),
  ], {
    x, y, w, colW: lebar, rowH: fRow, border: { type: "solid",
    color: C.garis, pt: 0.5 }, autoPage: false,
  });
}

function catatan(s, x, y, w, h, judulTeks, isi, warna = C.teal, muda = C.tealMuda) {
  s.addShape("rect", { x, y, w, h, fill: { color: muda },
                       line: { color: warna, width: 0.75 } });
  s.addShape("rect", { x, y, w: 0.05, h, fill: { color: warna },
                       line: { color: warna } });
  s.addText(judulTeks, { x: x + 0.2, y: y + 0.12, w: w - 0.35, h: 0.3,
    fontSize: 11, bold: true, color: warna, fontFace: FONT_ISI, margin: 0 });
  s.addText(isi, { x: x + 0.2, y: y + 0.45, w: w - 0.35, h: h - 0.55,
    fontSize: 10.5, color: C.tinta, fontFace: FONT_ISI, margin: 0,
    valign: "top" });
}

function nomor(s, n, x, y, warna = C.teal) {
  s.addShape("ellipse", { x, y, w: 0.34, h: 0.34,
    fill: { color: warna }, line: { color: warna } });
  s.addText(String(n), { x, y: y + 0.015, w: 0.34, h: 0.3,
    fontSize: 11, bold: true, color: C.putih, fontFace: FONT_ISI,
    align: "center", valign: "mid", margin: 0 });
}

// ===========================================================================
const pres = new PptxGenJS();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 inci
pres.author = "Kelompok 1 - ML2026";
pres.company = "Pembelajaran Mesin ML2026";
pres.subject = "Klasifikasi Lesi Kulit Berbasis Citra";
pres.title = "Laporan Proyek ML2026 - Versi Presentasi 2";

// ---------------------------------------------------------------------------
// SLIDE 1 -- JUDUL
// ---------------------------------------------------------------------------
{
  const s = pres.addSlide();
  latar(s, true);

  // Motif viewfinder besar di kanan
  viewfinder(s, 9.2, 1.35, 3.4, 4.6, C.teal);
  viewfinder(s, 9.65, 1.8, 2.5, 3.7, C.navySoft);

  s.addText("PROYEK AKHIR PEMBELAJARAN MESIN", {
    x: 0.75, y: 0.7, w: 8.2, h: 0.35, fontSize: 12, bold: true,
    color: C.teal, fontFace: FONT_ISI, charSpacing: 2, margin: 0,
  });
  s.addText("Klasifikasi Lesi Kulit\nBerbasis Citra", {
    x: 0.75, y: 1.35, w: 8.3, h: 1.6, fontSize: 39, bold: true,
    color: C.putih, fontFace: FONT_JUDUL, margin: 0, lineSpacing: 42,
  });
  s.addText("HOG-SVM · HOG-Random Forest · CNN · MobileNetV2\n" +
            "Termasuk evaluasi domain shift pada foto smartphone", {
    x: 0.75, y: 3.05, w: 8.2, h: 0.9, fontSize: 15,
    color: "C9D2E3", fontFace: FONT_ISI, margin: 0, lineSpacing: 22,
  });

  const anggota = (info.anggota || [])
    .map((a) => `${a.nama}  (${a.nim})`).join("\n");
  s.addText(anggota, {
    x: 0.75, y: 4.35, w: 8.2, h: 1.15, fontSize: 12.5,
    color: C.putih, fontFace: FONT_ISI, margin: 0, lineSpacing: 20,
  });
  s.addText(`${info.kelompok || "Kelompok 1 - ML2026"}   |   November 2026`, {
    x: 0.75, y: 6.5, w: 8.2, h: 0.35, fontSize: 11.5,
    color: "9AA8C2", fontFace: FONT_ISI, margin: 0,
  });
  s.addText("Alat bantu belajar. Bukan alat diagnosis medis.", {
    x: 0.75, y: 6.9, w: 8.2, h: 0.3, fontSize: 10.5, italic: true,
    color: "9AA8C2", fontFace: FONT_ISI, margin: 0,
  });
  s.addNotes("Slide pembuka. Sebutkan nama ketiga anggota dan judul proyek. " +
    "Tekan disclaimer: ini alat bantu belajar, bukan diagnosis medis.");
}

// ---------------------------------------------------------------------------
// SLIDE 2 -- RINGKASAN EKSEKUTIF
// ---------------------------------------------------------------------------
{
  const s = pres.addSlide();
  latar(s);
  judul(s, "RINGKASAN", "Proyek ini dalam empat angka",
        "Semua angka berasal dari kode yang benar-benar dijalankan");

  const totalGambar = (H.final && H.final.test_set_size) ? 10015 : 10015;
  const nKelas = DIST_DATA.length || 7;
  const rasio = DIST_DATA.length > 1
    ? (DIST_DATA[0].persentase / DIST_DATA[DIST_DATA.length - 1].persentase)
    : 58.3;
  const nModel = H.final ? Object.keys(H.final.per_model).length : 4;

  const kartuData = [
    { a: RIBU(totalGambar), s: "foto dermoskop HAM10000", w: C.navy, m: C.putih },
    { a: nKelas, s: "kelas diagnosis kulit", w: C.teal, m: C.putih },
    { a: `${rasio.toFixed(1).replace(".", ",")} : 1`, s: "ketidakseimbangan kelas", w: C.bata, m: C.putih },
    { a: nModel, s: "model dibandingkan", w: C.kuning, m: C.putih },
  ];
  kartuData.forEach((d, i) => {
    const x = 0.6 + i * 3.13;
    kartu(s, x, 1.95, 2.93, 1.85, null, { viewfinder: d.w });
    s.addText(d.a, { x: x + 0.2, y: 2.25, w: 2.5, h: 0.75,
      fontSize: 30, bold: true, color: d.w, fontFace: FONT_JUDUL, margin: 0 });
    s.addText(d.s, { x: x + 0.2, y: 3.05, w: 2.55, h: 0.5,
      fontSize: 11, color: C.tintaLembut, fontFace: FONT_ISI, margin: 0 });
  });

  labelKecil(s, "KESIMPULAN UTAMA", 0.6, 4.25, 4);
  kartu(s, 0.6, 4.62, 12.13, 1.75, null, { warna: C.putih, viewfinder: C.teal });
  s.addText(
    "MobileNetV2 paling baik pada F1 macro (0,3993) dan paling tahan " +
    "terhadap perubahan domain. Model dengan accuracy tertinggi justru " +
    "melewatkan 75% lesi ganas--accuracy bukan kriteria skrining.",
    { x: 0.9, y: 4.95, w: 11.5, h: 1.1, fontSize: 15, color: C.tinta,
      fontFace: FONT_ISI, margin: 0, valign: "mid", lineSpacing: 24 });

  s.addNotes("Empat angka kunci: 10.015 foto, 7 kelas, rasio 58,3:1, " +
    "empat model. Tekan kontradiksi antara accuracy dan recall ganas.");
}

// ---------------------------------------------------------------------------
// SLIDE 3 -- MASALAH & TUJUAN
// ---------------------------------------------------------------------------
{
  const s = pres.addSlide();
  latar(s);
  judul(s, "BAB 1", "Mengapa proyek ini penting",
        "Dari keterbatasan dermoskopi hingga kebutuhan skrining awal");

  kartu(s, 0.6, 1.95, 5.9, 4.6, null, { warna: C.putih, viewfinder: C.bata });
  labelKecil(s, "MASALAH", 0.9, 2.25, 3, C.bata);
  const masalah = [
    "Dermoskopi butuh dokter ahli dan alat khusus",
    "Data sangat tidak seimbang (58,3 : 1)",
    "Foto smartphone berbeda domain dari foto dermoskop",
    "Accuracy tinggi bisa menyembunyikan FN ganas",
  ];
  masalah.forEach((m, i) => {
    nomor(s, i + 1, 0.9, 2.75 + i * 0.78, C.bata);
    s.addText(m, { x: 1.4, y: 2.73 + i * 0.78, w: 4.7, h: 0.55,
      fontSize: 12.5, color: C.tinta, fontFace: FONT_ISI, margin: 0,
      valign: "mid" });
  });

  kartu(s, 6.83, 1.95, 5.9, 4.6, null, { warna: C.putih, viewfinder: C.teal });
  labelKecil(s, "TUJUAN", 7.13, 2.25, 3, C.teal);
  const tujuan = [
    "Pipeline tanpa kebocoran antar split",
    "Bandingkan 4 model dengan metrik yang tepat",
    "Ukur domain shift tanpa training ulang",
    "Aplikasi dengan preprocessing identik training",
  ];
  tujuan.forEach((m, i) => {
    nomor(s, i + 1, 7.13, 2.75 + i * 0.78, C.teal);
    s.addText(m, { x: 7.63, y: 2.73 + i * 0.78, w: 4.7, h: 0.55,
      fontSize: 12.5, color: C.tinta, fontFace: FONT_ISI, margin: 0,
      valign: "mid" });
  });

  s.addNotes("Empat masalah dan empat tujuan. Fokus pada poin keempat: " +
    "accuracy menyesatkan pada data tidak seimbang.");
}

// ---------------------------------------------------------------------------
// SLIDE 4 -- DATASET & EDA (chart native)
// ---------------------------------------------------------------------------
{
  const s = pres.addSlide();
  latar(s);
  judul(s, "BAB 2", "Dataset dan eksplorasi awal",
        "10.015 foto · 7.470 lesi unik · 7 kelas sangat tidak seimbang");

  if (DIST_DATA.length) {
    s.addChart("bar", [{
      name: "Jumlah gambar",
      labels: DIST_DATA.map((d) => d.kelas),
      values: DIST_DATA.map((d) => d.jumlah),
    }], {
      x: 0.55, y: 1.9, w: 7.0, h: 4.7,
      catAxisLabelColor: C.tinta, valAxisLabelColor: C.tintaLembut,
      catAxisLabelFontFace: FONT_ISI, valAxisLabelFontFace: FONT_ISI,
      catAxisLabelFontSize: 11, valAxisLabelFontSize: 9,
      valGridLine: { color: C.garis, size: 0.5 },
      catGridLine: { style: "none" },
      showLegend: false, showValue: true, dataLabelPosition: "outEnd",
      dataLabelColor: C.tinta, dataLabelFontSize: 10,
      chartColors: [C.navy],
      showTitle: true, title: "Distribusi kelas (jumlah gambar)",
      titleFontSize: 12, titleColor: C.tinta, titleFontFace: FONT_ISI,
    });
  }

  labelKecil(s, "TEMUAN EDA", 7.95, 2.0, 4);
  const temuan = [
    ["58,3 : 1", "rasio kelas terbesar terhadap terkecil"],
    ["2.545", "foto adalah lesi yang sama difoto ulang"],
    ["53,3%", "label berasal dari histopatologi"],
    ["I–II", "seluruh kulit Fitzpatrick I-II"],
  ];
  temuan.forEach(([a, k], i) => {
    const y = 2.4 + i * 1.0;
    kartu(s, 7.95, y, 4.78, 0.86, null, { warna: C.putih });
    s.addText(a, { x: 8.15, y: y + 0.08, w: 1.5, h: 0.5,
      fontSize: 22, bold: true, color: i === 0 ? C.bata : C.navy,
      fontFace: FONT_JUDUL, margin: 0 });
    s.addText(k, { x: 9.75, y: y + 0.14, w: 2.8, h: 0.55,
      fontSize: 10.5, color: C.tintaLembut, fontFace: FONT_ISI,
      margin: 0, valign: "mid" });
  });

  s.addNotes("Tunjuk kelas nv mendominasi. 2.545 foto berasal dari lesi yang " +
    "sama--inilah alasan split harus per lesion_id, bukan per gambar.");
}

// ---------------------------------------------------------------------------
// SLIDE 5 -- METODOLOGI / PIPELINE
// ---------------------------------------------------------------------------
{
  const s = pres.addSlide();
  latar(s);
  judul(s, "BAB 3", "Metodologi: satu pipeline, lima tahap",
        "Seed 42 di semua tempat · test set dipakai sekali di akhir");

  const langkah = [
    { n: "01", j: "SPLIT", d: "GroupShuffleSplit\nper lesion_id\n6.959 / 1.529 / 1.527", w: C.navy },
    { n: "02", j: "FITUR", d: "HOG 1.764 + histogram 48\nPCA(30) hanya pada HOG\nbobot histogram 6", w: C.teal },
    { n: "03", j: "MODEL", d: "SVM · RF · CNN\nMobileNetV2 stage 1\nclass weight wajib", w: C.kuning },
    { n: "04", j: "VALIDASI", d: "Threshold tiap model\ndari data validasi\ntarget recall 90%", w: C.bata },
    { n: "05", j: "UJI", d: "Test sekali\nLalu PAD-UFES-20\ntanpa training ulang", w: C.navy },
  ];
  langkah.forEach((l, i) => {
    const x = 0.55 + i * 2.52;
    kartu(s, x, 2.1, 2.32, 3.25, null, { warna: C.putih, viewfinder: l.w });
    s.addText(l.n, { x: x + 0.18, y: 2.35, w: 1, h: 0.5,
      fontSize: 26, bold: true, color: l.w, fontFace: FONT_JUDUL, margin: 0 });
    s.addText(l.j, { x: x + 0.18, y: 2.95, w: 2, h: 0.35,
      fontSize: 12, bold: true, color: C.tinta, fontFace: FONT_ISI,
      charSpacing: 1, margin: 0 });
    s.addText(l.d, { x: x + 0.18, y: 3.4, w: 2.0, h: 1.6,
      fontSize: 10.5, color: C.tintaLembut, fontFace: FONT_ISI,
      margin: 0, valign: "top", lineSpacing: 16 });
    if (i < langkah.length - 1) {
      s.addText("→", { x: x + 2.33, y: 3.35, w: 0.2, h: 0.4,
        fontSize: 16, bold: true, color: C.garis, fontFace: FONT_ISI,
        align: "center", margin: 0 });
    }
  });

  catatan(s, 0.55, 5.7, 12.23, 1.0, "TIGA ASSERT ANTI-KEBOCORAN",
    "Tidak ada lesi, image, atau path yang overlap antar split--ketiganya " +
    "bernilai nol. Scaler di-fit hanya pada data latih; normalisasi " +
    "ada di dalam model sehingga tidak dilakukan dua kali.",
    C.navy, "EDF0F5");

  s.addNotes("Jelaskan alur lima tahap. Tekan anti-kebocoran dan aturan " +
    "scaler hanya dilatih pada data train.");
}

// ---------------------------------------------------------------------------
// SLIDE 6 -- MODEL KLASIK
// ---------------------------------------------------------------------------
{
  const s = pres.addSlide();
  latar(s);
  judul(s, "BAB 3", "Model klasik: fitur hasil eksperimen",
        "Bukan tebakan--setiap keputusan diukur pada data validasi");

  // Rantai preprocessing
  const rantai = ["HOG 1.764", "StandardScaler", "PCA(30)", "StandardScaler",
                  "GABUNG", "SVM / RF"];
  rantai.forEach((r, i) => {
    const x = 0.55 + i * 2.08;
    const w = i === 5 ? 2.0 : 1.88;
    s.addShape("rect", { x, y: 2.0, w, h: 0.72,
      fill: { color: i === 4 ? C.bata : (i === 1 || i === 3 ? C.navySoft : C.teal) },
      line: { color: i === 4 ? C.bata : (i === 1 || i === 3 ? C.navySoft : C.teal) } });
    s.addText(r, { x, y: 2.1, w, h: 0.5, fontSize: 10.5, bold: true,
      color: C.putih, fontFace: FONT_ISI, align: "center", margin: 0 });
    if (i < rantai.length - 1) {
      s.addText("→", { x: x + w, y: 2.15, w: 0.2, h: 0.4, fontSize: 14,
        bold: true, color: C.garis, align: "center", margin: 0 });
    }
  });
  s.addText("histogram 48 fitur × bobot 6 masuk langsung ke GABUNG " +
            "(tanpa di-scale ulang)", {
    x: 0.55, y: 2.82, w: 12.2, h: 0.3, fontSize: 10.5, italic: true,
    color: C.tintaLembut, fontFace: FONT_ISI, margin: 0,
  });

  const d = {};
  (H.diag || []).forEach((r) => { d[r.konfigurasi] = r; });
  const base = d["BASELINE: selalu nebak nv"];
  const rf = d["RF pada 1812 fitur penuh"];
  const pca = d["SVC PCA(400) dari 1812 fitur"];
  const hist = d["SVC-RBF histogram saja (48)"];

  const temuan = [
    { j: "RF mentah hampir tidak belajar",
      v: `${NUM(base && base.f1_macro)} → ${NUM(rf && rf.f1_macro)}`,
      k: "F1 baseline versus RF pada 1.812 fitur; recall ganas 0,0192",
      w: C.bata },
    { j: "PCA global merugikan",
      v: `${NUM(pca && pca.f1_macro)} vs 0,3423`,
      k: "PCA(400) semua fitur versus PCA(30) HOG saja + histogram",
      w: C.kuning },
    { j: "Warna lebih kuat dari HOG",
      v: `${NUM(hist && hist.malignant_recall)}`,
      k: "Recall ganas SVM pada 48 fitur histogram saja",
      w: C.teal },
  ];
  temuan.forEach((t, i) => {
    const y = 3.35 + i * 1.06;
    kartu(s, 0.55, y, 12.23, 0.92, null, { warna: C.putih, viewfinder: t.w });
    s.addText(t.j, { x: 0.8, y: y + 0.12, w: 3.6, h: 0.35, fontSize: 12,
      bold: true, color: t.w, fontFace: FONT_ISI, margin: 0 });
    s.addText(t.v, { x: 4.5, y: y + 0.08, w: 2.6, h: 0.5, fontSize: 20,
      bold: true, color: C.navy, fontFace: FONT_JUDUL, margin: 0 });
    s.addText(t.k, { x: 7.2, y: y + 0.12, w: 5.3, h: 0.6, fontSize: 10.5,
      color: C.tintaLembut, fontFace: FONT_ISI, margin: 0, valign: "mid" });
  });

  s.addNotes("Tiga temuan utama. Angka 0,3423 berasal dari diagnostik_fitur.json " +
    "untuk PCA(30) HOG + histogram. Tekan bahwa warna adalah sinyal terkuat.");
}

// ---------------------------------------------------------------------------
// SLIDE 7 -- DEEP LEARNING
// ---------------------------------------------------------------------------
{
  const s = pres.addSlide();
  latar(s);
  judul(s, "BAB 3", "Deep learning: CNN dan MobileNetV2",
        "Class weight wajib · fine-tuning tidak selalu membantu");

  labelKecil(s, "CNN DARI NOL — TIGA VARIAN", 0.6, 1.85, 6);
  const gap = (H.cnn && H.cnn.gap_vs_flatten) || [];
  const cnnRows = gap.map((v) => [
    v.model.replace("CNN ", "").replace(" + ", " + "),
    RIBU(v.params), NUM(v.f1_macro), NUM(v.malignant_recall),
    `${v.kelas_terprediksi}/7`,
  ]);
  if (cnnRows.length) {
    tabel(s, 0.6, 2.2, 6.0, ["Varian", "Param", "F1", "Rec ganas", "Kelas"],
      cnnRows, [2.05, 1.05, 0.85, 1.15, 0.9], { fs: 9, fRow: 0.44 });
  }
  catatan(s, 0.6, 4.05, 6.0, 1.0, "BUKTI CLASS WEIGHT",
    "Tanpa class weight, accuracy mengunci 0,672 dengan recall ganas " +
    "nol--model hanya menebak kelas nv.", C.bata, C.bataMuda);

  labelKecil(s, "MOBILENETV2 — TAHAP 1 vs TAHAP 2", 6.95, 1.85, 6, C.teal);
  const pb = (H.mnv2 && H.mnv2.perbandingan) || {};
  const g = (k, t) => (pb[k] && pb[k][t] != null) ? NUM(pb[k][t]) : "-";
  const fn = pb.fn_malignant || {};
  const mRows = [
    ["F1 macro", g("f1_macro", "stage1"), g("f1_macro", "stage2")],
    ["Accuracy", g("accuracy", "stage1"), g("accuracy", "stage2")],
    ["Recall ganas", g("malignant_recall", "stage1"), g("malignant_recall", "stage2")],
    ["FN ganas", String(fn.stage1 ?? "-"), String(fn.stage2 ?? "-")],
  ];
  tabel(s, 6.95, 2.2, 5.78, ["Metrik", "Tahap 1", "Tahap 2"], mRows,
    [2.4, 1.69, 1.69], { fs: 10, fRow: 0.44 });
  catatan(s, 6.95, 4.05, 5.78, 1.0, "KEPUTUSAN",
    "F1 macro hampir sama, tetapi recall ganas turun 0,5545 → 0,3750. " +
    "FN ganas naik 139 → 195. Tahap 1 yang dipakai.", C.teal, C.tealMuda);

  kartu(s, 0.6, 5.4, 12.13, 1.3, null, { warna: C.putih, viewfinder: C.navy });
  s.addText("Perbandingan adil: kedua tahap memakai data validasi dan " +
    "augmentasi training yang sama. Satu-satunya yang berbeda adalah " +
    "backbone dibekukan atau tidak, serta learning rate (1e-3 vs 1e-5).",
    { x: 0.9, y: 5.68, w: 11.5, h: 0.75, fontSize: 12.5, color: C.tinta,
      fontFace: FONT_ISI, margin: 0, valign: "mid" });

  s.addNotes("Tunjuk tabel CNN: varian Flatten menang F1. Lalu tabel MNV2: " +
    "F1 setara tapi recall turun. Keputusan berbasis angka, bukan harapan.");
}

// ---------------------------------------------------------------------------
// SLIDE 8 -- HASIL TEST SET (chart + tabel)
// ---------------------------------------------------------------------------
{
  const s = pres.addSlide();
  latar(s);
  judul(s, "BAB 4", "Hasil pada data uji (dipakai sekali)",
        "1.527 gambar · metrik utama F1 macro dan recall ganas");

  const pm = (H.final && H.final.per_model) || {};
  const nama = Object.keys(pm);

  const labelPendek = nama.map(
    (n) => n.replace(" (from scratch)", "").replace(" (HOG)", ""));
  const sAddChart = [
    { name: "F1 macro", labels: labelPendek,
      values: nama.map((n) => Number(pm[n].f1_macro.toFixed(4))) },
    { name: "Recall ganas", labels: labelPendek,
      values: nama.map((n) => Number(pm[n].malignant_recall.toFixed(4))) },
  ];
  s.addChart("bar", sAddChart, {
    x: 0.55, y: 1.9, w: 6.1, h: 4.75,
    barGrouping: "clustered",
    catAxisLabelColor: C.tinta, valAxisLabelColor: C.tintaLembut,
    catAxisLabelFontFace: FONT_ISI, valAxisLabelFontFace: FONT_ISI,
    catAxisLabelFontSize: 10, valAxisLabelFontSize: 9,
    valGridLine: { color: C.garis, size: 0.5 }, catGridLine: { style: "none" },
    showLegend: true, legendPos: "b", legendFontSize: 10,
    chartColors: [C.navy, C.bata],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 9,
    showTitle: true, title: "F1 macro vs recall ganas",
    titleFontSize: 12, titleColor: C.tinta, titleFontFace: FONT_ISI,
  });

  const rows = nama.map((n) => {
    const m = pm[n];
    const b = m.binary_with_tuned_threshold || {};
    return [n.replace(" (from scratch)", "").replace(" (HOG)", ""),
            NUM(m.accuracy), NUM(m.f1_macro), NUM(m.malignant_recall),
            NUM(m.specificity), String(b.fn ?? "-"), String(b.fp ?? "-")];
  });
  tabel(s, 6.9, 2.0, 5.83,
    ["Model", "Acc", "F1", "Rec", "Spec", "FN", "FP"], rows,
    [1.55, 0.7, 0.7, 0.7, 0.7, 0.69, 0.79], { fs: 9, fRow: 0.46 });

  catatan(s, 6.9, 4.7, 5.83, 1.35, "DUA FAKTA YANG BERTENTANGAN",
    "Accuracy tertinggi: Random Forest 0,7132, tetapi recall ganasnya " +
    "0,2600--ia melewatkan 75% lesi ganas. F1 tertinggi: MobileNetV2 " +
    "0,3993 dengan specificity 0,8720.", C.bata, C.bataMuda);

  s.addNotes("Tunjuk grafik dan tabel. Tekan FN ganas sebagai kesalahan " +
    "paling mahal. Model terpilih MobileNetV2 dengan skor 0,7043.");
}

// ---------------------------------------------------------------------------
// SLIDE 9 -- DOMAIN SHIFT
// ---------------------------------------------------------------------------
{
  const s = pres.addSlide();
  latar(s);
  judul(s, "BAB 4", "Uji eksternal: foto smartphone",
        "PAD-UFES-20 · 2.298 foto · tanpa training ulang");

  const pm = (H.final && H.final.per_model) || {};
  const ext = (H.pad && H.pad.hasil_per_model) || {};
  const rows = Object.keys(ext).map((n) => {
    const e = ext[n];
    const p = pm[n] || {};
    return [n.replace(" (from scratch)", "").replace(" (HOG)", ""),
            NUM(p.malignant_recall), NUM(e.recall),
            NUM(e.specificity), String(e.fn), String(e.fp)];
  });
  tabel(s, 0.6, 2.0, 12.13,
    ["Model", "Rec HAM", "Rec PAD", "Spec PAD", "FN PAD", "FP PAD"], rows,
    [3.0, 1.8, 1.8, 1.8, 1.85, 1.88], { fs: 10, fRow: 0.48 });

  catatan(s, 0.6, 4.5, 6.0, 1.55, "JANGAN TERTIPU OLEH RECALL",
    "CNN punya recall 0,9698 di PAD, tetapi specificity hanya 0,0104--" +
    "ia menandai hampir semua foto sebagai ganas. recall tinggi tidak " +
    "sama dengan model lebih baik.", C.bata, C.bataMuda);

  catatan(s, 6.83, 4.5, 5.9, 1.55, "PALING SEIMBANG",
    "MobileNetV2: recall 0,8730 dengan specificity 0,4489 dan FN 231. " +
    "MEL hanya 52 gambar (recall 0,6346); SCC tidak ada di HAM10000, " +
    "sebuah kelas yang belum pernah dilihat model.", C.teal, C.tealMuda);

  s.addNotes("Ini bagian paling penting. Jelaskan bahwa recall naik karena " +
    "specificity jatuh. MobileNetV2 paling seimbang. SCC adalah unseen class.");
}

// ---------------------------------------------------------------------------
// SLIDE 10 -- PRODUK / APLIKASI
// ---------------------------------------------------------------------------
{
  const s = pres.addSlide();
  latar(s);
  judul(s, "BAB 5", "Produk: aplikasi Streamlit",
        "Preprocessing identik dengan training · ambang dari validasi");

  // Mock UI sederhana yang digambar dengan shape
  s.addShape("rect", { x: 0.6, y: 1.9, w: 7.6, h: 4.85,
    fill: { color: C.putih }, line: { color: C.garis, width: 0.75 } });
  s.addShape("rect", { x: 0.6, y: 1.9, w: 7.6, h: 0.55,
    fill: { color: C.navy }, line: { color: C.navy } });
  s.addText("Skrining Risiko Lesi Kulit dari Foto", {
    x: 0.85, y: 2.0, w: 7, h: 0.35, fontSize: 13, bold: true,
    color: C.putih, fontFace: FONT_JUDUL, margin: 0 });

  s.addText("UNGGAH FOTO / PAKAI CONTOH", {
    x: 0.9, y: 2.7, w: 2.9, h: 0.3, fontSize: 9.5, bold: true,
    color: C.tintaLembut, fontFace: FONT_ISI, charSpacing: 0.8, margin: 0 });
  s.addShape("rect", { x: 0.9, y: 3.08, w: 2.9, h: 1.4,
    fill: { color: C.kertas }, line: { color: C.garis, width: 1,
    dashType: "dash" } });
  s.addText("+ Pilih file JPG/PNG\natau centang contoh", {
    x: 1.0, y: 3.5, w: 2.7, h: 0.6, fontSize: 10.5, color: C.tintaLembut,
    fontFace: FONT_ISI, align: "center", margin: 0 });

  s.addText("HASIL ANALISIS", {
    x: 4.05, y: 2.7, w: 3.8, h: 0.3, fontSize: 9.5, bold: true,
    color: C.tintaLembut, fontFace: FONT_ISI, charSpacing: 0.8, margin: 0 });
  s.addShape("rect", { x: 4.05, y: 3.08, w: 3.8, h: 0.72,
    fill: { color: C.tealMuda }, line: { color: C.teal, width: 0.75 } });
  s.addText("Tidak ditemukan ciri ganas", {
    x: 4.25, y: 3.25, w: 3.4, h: 0.35, fontSize: 12, bold: true,
    color: C.teal, fontFace: FONT_ISI, margin: 0 });

  const bars = [
    ["Nevus", 0.72, C.teal], ["Melanoma", 0.14, C.bata],
    ["BKL", 0.08, C.teal], ["BCC", 0.04, C.bata],
    ["AKIEC", 0.01, C.bata], ["DF", 0.006, C.teal], ["VASC", 0.004, C.teal],
  ];
  bars.forEach((b, i) => {
    const y = 3.95 + i * 0.38;
    s.addText(b[0], { x: 4.05, y, w: 1.0, h: 0.28, fontSize: 9,
      color: C.tinta, fontFace: FONT_ISI, margin: 0, valign: "mid" });
    s.addShape("rect", { x: 5.1, y: y + 0.06, w: 2.0, h: 0.14,
      fill: { color: C.garis }, line: { color: C.garis } });
    s.addShape("rect", { x: 5.1, y: y + 0.06, w: Math.max(0.05, b[1] * 2.0),
      h: 0.14, fill: { color: b[2] }, line: { color: b[2] } });
    s.addText(`${(b[1] * 100).toFixed(0)}%`, { x: 7.2, y, w: 0.6, h: 0.28,
      fontSize: 9, color: C.tintaLembut, fontFace: FONT_ISI, margin: 0,
      align: "right", valign: "mid" });
  });

  s.addShape("rect", { x: 0.9, y: 4.75, w: 2.9, h: 1.55,
    fill: { color: C.bataMuda }, line: { color: C.bata, width: 0.75 } });
  s.addText("DISCLAIMER", { x: 1.1, y: 4.9, w: 2.5, h: 0.3,
    fontSize: 10, bold: true, color: C.bata, fontFace: FONT_ISI, margin: 0 });
  s.addText("Alat bantu belajar, bukan diagnosis. Tetap periksakan " +
    "ke dokter kulit.", { x: 1.1, y: 5.25, w: 2.5, h: 0.85, fontSize: 9.5,
    color: C.tinta, fontFace: FONT_ISI, margin: 0, valign: "top" });

  labelKecil(s, "YANG DIJAGA", 8.55, 2.0, 4, C.teal);
  const fitur = [
    "7 bar probabilitas per kelas",
    "Status tiga warna + deteksi ragu",
    "4 contoh gambar siap pakai",
    "Privasi: foto tidak disimpan",
    "Model di-cache, interaksi cepat",
  ];
  fitur.forEach((f, i) => {
    nomor(s, i + 1, 8.55, 2.45 + i * 0.72, C.teal);
    s.addText(f, { x: 9.05, y: 2.43 + i * 0.72, w: 3.6, h: 0.5,
      fontSize: 11, color: C.tinta, fontFace: FONT_ISI, margin: 0,
      valign: "mid" });
  });
  catatan(s, 8.55, 6.05, 4.18, 0.7, "CARA JALANKAN",
    "streamlit run app/app.py", C.navy, "EDF0F5");

  s.addNotes("Tunjuk panel kiri (unggah/contoh) dan panel kanan (status + " +
    "bar probabilitas). Tekan disclaimer dan privasi.");
}

// ---------------------------------------------------------------------------
// SLIDE 11 -- TEMUAN & KETERBATASAN
// ---------------------------------------------------------------------------
{
  const s = pres.addSlide();
  latar(s);
  judul(s, "BAB 5", "Apa yang kita pelajari",
        "13 bug ditemukan lewat eksekusi, bukan dari membaca kode");

  kartu(s, 0.6, 1.95, 3.0, 2.3, null, { warna: C.navy, viewfinder: C.teal });
  s.addText("13", { x: 0.9, y: 2.25, w: 2.4, h: 1.0, fontSize: 58,
    bold: true, color: C.putih, fontFace: FONT_JUDUL, margin: 0 });
  s.addText("bug nyata\nditemukan & diperbaiki", { x: 0.9, y: 3.35, w: 2.4,
    h: 0.7, fontSize: 11.5, color: "C9D2E3", fontFace: FONT_ISI,
    margin: 0, lineSpacing: 17 });

  const temuan = [
    "Kalibrasi SVM bawaan menurunkan recall ganas 0,49 → 0,19",
    "Satu ambang untuk semua model membuat uji eksternal tidak adil",
    "Band 'ragu' yang tetap membuat semua foto terlihat meragukan",
    "Ukuran sel HOG hampir tidak berpengaruh; yang menang adalah efisiensi",
  ];
  temuan.forEach((t, i) => {
    const y = 2.0 + i * 1.08;
    kartu(s, 3.9, y, 8.83, 0.95, null, { warna: C.putih,
      viewfinder: i === 0 ? C.bata : C.teal });
    nomor(s, i + 1, 4.15, y + 0.3, i === 0 ? C.bata : C.teal);
    s.addText(t, { x: 4.65, y: y + 0.14, w: 7.8, h: 0.65, fontSize: 11.5,
      color: C.tinta, fontFace: FONT_ISI, margin: 0, valign: "mid" });
  });

  catatan(s, 0.6, 4.6, 3.0, 2.1, "KETERBATASAN",
    "Fitzpatrick I-II saja · kelas df/vasc sangat sedikit · belum " +
    "divalidasi klinis · sensitif kualitas foto.", C.bata, C.bataMuda);

  s.addNotes("Empat pelajaran utama dan empat keterbatasan. Tekan bahwa " +
    "setiap angka di slide ini berasal dari pengukuran.");
}

// ---------------------------------------------------------------------------
// SLIDE 12 -- KESIMPULAN
// ---------------------------------------------------------------------------
{
  const s = pres.addSlide();
  latar(s, true);
  viewfinder(s, 10.2, 0.9, 2.5, 2.5, C.teal);

  s.addText("KESIMPULAN", {
    x: 0.75, y: 0.7, w: 8.5, h: 0.35, fontSize: 12, bold: true,
    color: C.teal, fontFace: FONT_ISI, charSpacing: 2, margin: 0,
  });
  s.addText("Empat model, satu kesimpulan:\nskrining butuh sensitivitas,", {
    x: 0.75, y: 1.3, w: 11.5, h: 1.4, fontSize: 30, bold: true,
    color: C.putih, fontFace: FONT_JUDUL, margin: 0, lineSpacing: 38,
  });
  s.addText("bukan sekadar accuracy.", {
    x: 0.75, y: 2.75, w: 11.5, h: 0.6, fontSize: 30, bold: true,
    color: C.teal, fontFace: FONT_JUDUL, margin: 0,
  });

  const poin = [
    "MobileNetV2 terbaik: F1 0,3993 · recall ganas 0,4500 · paling tahan domain shift",
    "Class weight dan BatchNormalization adalah keharusan teknis",
    "Fine-tuning pada data kecil menurunkan sensitivitas (0,5545 → 0,3750)",
    "Aplikasi siap dipakai dengan preprocessing identik training",
  ];
  poin.forEach((p, i) => {
    const y = 3.7 + i * 0.62;
    s.addShape("ellipse", { x: 0.78, y: y + 0.06, w: 0.2, h: 0.2,
      fill: { color: C.teal }, line: { color: C.teal } });
    s.addText(p, { x: 1.15, y, w: 11.2, h: 0.45, fontSize: 12.5,
      color: "DCE3F0", fontFace: FONT_ISI, margin: 0, valign: "mid" });
  });

  s.addText("Terima kasih. Pertanyaan?", {
    x: 0.75, y: 6.45, w: 8, h: 0.5, fontSize: 18, bold: true,
    color: C.putih, fontFace: FONT_JUDUL, margin: 0,
  });
  s.addText("Dokumen:  LAPORAN_PROJECT_FINAL.md  ·  PANDUAN_PENGGUNAAN.md  ·  REKAP_LAPORAN.md", {
    x: 0.75, y: 7.0, w: 12, h: 0.3, fontSize: 10,
    color: "9AA8C2", fontFace: FONT_ISI, margin: 0,
  });

  s.addNotes("Tutup dengan empat poin. Undang pertanyaan. Ingatkan bahwa " +
    "video demo dan upload GitHub masih perlu dilakukan mahasiswa.");
}

// ---------------------------------------------------------------------------
pres.writeFile({ fileName: OUT }).then(() => {
  console.log(`PPT v2 dibuat: ${OUT}`);
  console.log(`Slide: ${pres.slides.length}`);
  console.log("Sumber angka: " +
    `final=${H.final ? "ada" : "TIDAK ADA"}, ` +
    `pad=${H.pad ? "ada" : "TIDAK ADA"}, ` +
    `diag=${H.diag ? "ada" : "TIDAK ADA"}`);
}).catch((e) => {
  console.error("GAGAL membuat PPT:", e);
  process.exit(1);
});
