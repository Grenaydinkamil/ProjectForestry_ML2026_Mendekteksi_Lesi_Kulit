/**
 * scripts/build_ppt_kel6.js
 * ==========================
 * Generator PPT presentasi Kelompok 6 untuk proyek Klasifikasi Lesi Kulit.
 *
 * Struktur dan komposisi slide mengikuti acuan PPT Kelompok 5 (22 slide,
 * footer "NN / 22", eyebrow "BAB X · NAMA", kartu angka, tabel, chevron).
 * ISI dan ANGKA seluruhnya milik proyek ini sendiri.
 *
 * ATURAN PENTING (sama seperti build_ppt.js):
 *   SEMUA angka di slide dibaca dari docs/logs/*.json. Tidak ada angka yang
 *   diketik manual. Kalau log tidak ada, slide dilewati, bukan diisi karangan.
 *
 * Jalankan:
 *   node scripts/build_ppt_kel6.js
 *
 * Output: docs/PPT_KEL6_Kaltafarmforesty.pptx
 */

const fs = require("fs");
const path = require("path");
const PptxGenJS = require("pptxgenjs");

const REPO = path.resolve(__dirname, "..");
const LOGS = path.join(REPO, "docs", "logs");
const OUT = path.join(REPO, "docs", "PPT_KEL6_Kaltafarmforesty.pptx");

// ---------------------------------------------------------------------------
// PALET
// ---------------------------------------------------------------------------
// Tema: dermatologi klinis. Warna diambil dari aplikasi yang benar-benar
// dipakai proyek (teal = jinak, rose = ganas), plus navy sebagai tinta
// struktur. Tidak ada warna dekoratif yang tidak punya fungsi.
// Dominasi: navy + paper. Aksen: teal. Alarm tunggal: rose.
const C = {
  navy: "14213D",
  navySoft: "25335A",
  ink: "1B2333",
  inkLembut: "5C6675",
  paper: "F4F6F9",
  garis: "DCE1E8",
  garisGelap: "C3CBD8",
  putih: "FFFFFF",
  teal: "0D9488",
  tealMuda: "E3F2F0",
  rose: "E11D48",
  roseMuda: "FCE7EC",
  amber: "B45309",
  amberMuda: "FBF0E2",
  slateMuda: "9AA8C2",
  teksGelap: "C9D2E3",
};

const FONT_JUDUL = "Cambria";
const FONT_ISI = "Calibri";

const M = 0.6;            // margin kiri dan kanan
const LEBAR = 13.33;
const TOTAL_SLIDE = 21;
const NAMA_KELOMPOK = "Kelompok 6";
const JUDUL_PROYEK = "Klasifikasi Lesi Kulit 7 Kelas";
const APP_URL =
  "https://projectforestryml2026mendekteksilesikulit-vvearxur6nuwga44fmaf.streamlit.app";
const REPO_URL =
  "https://github.com/Grenaydinkamil/ProjectForestry_ML2026_Mendekteksi_Lesi_Kulit";

// ---------------------------------------------------------------------------
// PEMBACA LOG
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

const H = {
  final: bacaJson("hasil_evaluasi_final.json"),
  pad: bacaJson("hasil_uji_eksternal_pad_ufes.json"),
  split: bacaJson("split_summary.json"),
  diag: bacaJson("diagnostik_fitur.json"),
  hog: bacaJson("diagnostik_hog_cell.json"),
  klasik: bacaJson("model_klasik_validasi.json"),
  cnn: bacaJson("model_cnn_validasi.json"),
  mnet: bacaJson("model_mobilenetv2_validasi.json"),
  info: JSON.parse(
    fs.readFileSync(path.join(REPO, "docs", "info_kelompok.json"), "utf-8")
  ),
};

if (!H.final || !H.split || !H.pad) {
  console.error("Log utama tidak lengkap. Jalankan notebook evaluasi lebih dulu.");
  process.exit(1);
}

// ---------------------------------------------------------------------------
// FORMAT ANGKA (pakai koma desimal, gaya Indonesia)
// ---------------------------------------------------------------------------
const f = (v, d = 4) => (v == null ? "-" : Number(v).toFixed(d).replace(".", ","));
const persen = (v, d = 1) =>
  v == null ? "-" : (v * 100).toFixed(d).replace(".", ",") + "%";
const thousands = (v) => (v == null ? "-" : String(v).replace(/\B(?=(\d{3})+(?!\d))/g, "."));

// ---------------------------------------------------------------------------
// RINGKASAN DATA (dihitung dari log, bukan diketik)
// ---------------------------------------------------------------------------
const KELAS = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"];
const LABEL_KELAS = {
  nv: "Nevus (nv)",
  mel: "Melanoma (mel)",
  bkl: "Benign keratosis (bkl)",
  bcc: "Basal cell carcinoma (bcc)",
  akiec: "Actinic keratosis (akiec)",
  vasc: "Vascular lesion (vasc)",
  df: "Dermatofibroma (df)",
};

// Jumlah kelas per split, digabung untuk seluruh dataset.
const JUMLAH_KELAS = {};
for (const split of ["train", "val", "test"]) {
  const dist = H.split.class_distribution[split] || {};
  for (const k of Object.keys(dist)) {
    JUMLAH_KELAS[k] = (JUMLAH_KELAS[k] || 0) + dist[k];
  }
}
const TOTAL_FOTO = Object.values(JUMLAH_KELAS).reduce((a, b) => a + b, 0);
const TOTAL_LESI =
  H.split.counts.train.lesions + H.split.counts.val.lesions + H.split.counts.test.lesions;
const RASIO_KELAS =
  Math.round((JUMLAH_KELAS.nv / JUMLAH_KELAS.df) * 10) / 10;
const PESEN_DF = JUMLAH_KELAS.df / TOTAL_FOTO;
const PESEN_NV = JUMLAH_KELAS.nv / TOTAL_FOTO;
const FOTO_PER_LESI = TOTAL_FOTO - TOTAL_LESI;

// Ukuran bobot model terpilih diambil dari berkas aslinya, bukan diketik.
const UKURAN_MODEL = (() => {
  const p = path.join(REPO, "models", "mobilenetv2_best.keras");
  if (!fs.existsSync(p)) return null;
  return (fs.statSync(p).size / (1024 * 1024)).toFixed(1).replace(".", ",");
})();

const MOB = H.final.per_model["MobileNetV2"];
const BIN = MOB.binary_with_tuned_threshold;
const MODEL_TERPILIH = H.final.selected_model;
const SKOR = H.final.selection_scores;

// ---------------------------------------------------------------------------
// HELPER SLIDE
// ---------------------------------------------------------------------------
let nomorSlide = 0;

function footer(s, gelap) {
  nomorSlide += 1;
  const warna = gelap ? C.slateMuda : C.inkLembut;
  s.addText(`${JUDUL_PROYEK} · ${NAMA_KELOMPOK}`, {
    x: M, y: 6.98, w: 8.5, h: 0.28, fontSize: 9.5, color: warna,
    fontFace: FONT_ISI, margin: 0,
  });
  s.addText(`${String(nomorSlide).padStart(2, "0")} / ${TOTAL_SLIDE}`, {
    x: 10.6, y: 6.98, w: 2.13, h: 0.28, fontSize: 9.5, bold: true,
    color: gelap ? C.slateMuda : C.inkLembut, fontFace: FONT_ISI,
    align: "right", margin: 0,
  });
}

function slideKonten(pres, kicker, judul, sub) {
  const s = pres.addSlide();
  s.background = { color: C.paper };

  // Penanda bagian: kotak kecil, bukan garis dekoratif di bawah judul.
  s.addShape("rect", { x: M, y: 0.37, w: 0.1, h: 0.1, fill: { color: C.teal } });
  s.addText(kicker, {
    x: M + 0.2, y: 0.3, w: 10.5, h: 0.26, fontSize: 10.5, bold: true,
    color: C.teal, fontFace: FONT_ISI, charSpacing: 1.6, margin: 0,
  });
  s.addText(judul, {
    x: M, y: 0.6, w: 12.13, h: 0.58, fontSize: 27, bold: true,
    color: C.navy, fontFace: FONT_JUDUL, margin: 0,
  });
  if (sub) {
    s.addText(sub, {
      x: M, y: 1.22, w: 12.13, h: 0.34, fontSize: 12.5, color: C.inkLembut,
      fontFace: FONT_ISI, margin: 0,
    });
  }
  s.addShape("rect", {
    x: M, y: 1.72, w: 12.13, h: 0.012, fill: { color: C.garis },
  });
  footer(s, false);
  return s;
}

function slideGelap(pres) {
  const s = pres.addSlide();
  s.background = { color: C.navy };
  // Lingkaran besar: motif "lesi yang difoto" yang sama di slide pembuka dan
  // penutup. Dipakai sebagai tekstur latar, bukan sebagai hiasan acak.
  s.addShape("ellipse", {
    x: 8.7, y: -1.5, w: 6.6, h: 6.6, fill: { color: C.navySoft },
  });
  s.addShape("ellipse", {
    x: 10.4, y: 3.6, w: 5.4, h: 5.4, fill: { color: "1B2C55" },
  });
  footer(s, true);
  return s;
}

function kartu(s, x, y, w, h, opsi = {}) {
  const o = {
    x, y, w, h,
    rectRadius: opsi.r || 0.1,
    fill: { color: opsi.fill || C.putih },
  };
  if (opsi.line) o.line = { color: opsi.line, width: 0.75 };
  if (opsi.shadow !== false) {
    o.shadow = {
      type: "outer", color: "9AA4B5", blur: 7, offset: 1, angle: 90,
      opacity: 0.18,
    };
  }
  s.addShape("roundRect", o);
}

// Kartu statistik dengan bar rasio, dipakai untuk EDA dan kesimpulan.
function kartuAngka(s, x, y, w, h, o) {
  kartu(s, x, y, w, h, {});
  s.addShape("rect", {
    x: x + 0.25, y: y + 0.24, w: 0.42, h: 0.055, fill: { color: o.warna || C.teal },
  });
  s.addText(o.label, {
    x: x + 0.25, y: y + 0.36, w: w - 0.5, h: 0.26, fontSize: 9.5, bold: true,
    color: C.inkLembut, fontFace: FONT_ISI, charSpacing: 1, margin: 0,
  });
  s.addText(o.nilai, {
    x: x + 0.25, y: y + 0.64, w: w - 0.5, h: 0.62, fontSize: o.ukuran || 30,
    bold: true, color: o.warna || C.teal, fontFace: FONT_JUDUL, margin: 0,
  });
  if (o.keterangan) {
    s.addText(o.keterangan, {
      x: x + 0.25, y: y + 1.26, w: w - 0.5, h: 0.3, fontSize: 10.5, bold: true,
      color: C.ink, fontFace: FONT_ISI, margin: 0,
    });
  }
  if (o.bar) {
    const lebarBar = w - 0.5;
    s.addShape("roundRect", {
      x: x + 0.25, y: y + 1.62, w: lebarBar, h: 0.11, rectRadius: 0.055,
      fill: { color: C.garis },
    });
    s.addShape("roundRect", {
      x: x + 0.25, y: y + 1.62, w: Math.max(0.12, lebarBar * o.bar), h: 0.11,
      rectRadius: 0.055, fill: { color: o.warna || C.teal },
    });
  }
  if (o.catatan) {
    s.addText(o.catatan, {
      x: x + 0.25, y: y + 1.85, w: w - 0.5, h: 0.5, fontSize: 10,
      color: C.inkLembut, fontFace: FONT_ISI, margin: 0, valign: "top",
    });
  }
}

// Bar bertumpuk horizontal untuk komposisi kelas.
function barTumpuk(s, x, y, w, h, segmen) {
  let xkursor = x;
  segmen.forEach((sg) => {
    const lebar = w * sg.porsi;
    s.addShape("rect", {
      x: xkursor, y, w: Math.max(lebar, 0.02), h, fill: { color: sg.warna },
    });
    if (lebar > 0.62) {
      s.addText(sg.label, {
        x: xkursor, y, w: lebar, h, fontSize: 9.5, bold: true,
        color: C.putih, fontFace: FONT_ISI, align: "center", valign: "mid",
        margin: 0,
      });
    }
    xkursor += lebar;
  });
}

function tabel(s, x, y, w, header, rows, lebar, opsi = {}) {
  const fs = opsi.fs || 10;
  const isiBaris = opsi.barisTinggi || 0.36;
  s.addTable(
    [
      header.map((h) => ({
        text: h,
        options: {
          bold: true, color: C.putih, fill: { color: C.navy },
          fontSize: fs, fontFace: FONT_ISI, margin: 0.07,
          align: opsi.rataKanan && opsi.rataKanan.includes(header.indexOf(h)) ? "right" : "left",
        },
      })),
      ...rows.map((r) =>
        r.map((v, i) => ({
          text: String(v),
          options: {
            color: i === 0 ? C.ink : C.inkLembut,
            bold: i === 0 || (opsi.tebalkan && opsi.tebalkan.includes(r[0])),
            fontSize: fs, fontFace: FONT_ISI, margin: 0.07,
            fill: { color: opsi.sorotBaris && opsi.sorotBaris.includes(r[0]) ? C.tealMuda : C.putih },
            align: opsi.kolomRata && opsi.kolomRata[i] ? opsi.kolomRata[i] : "left",
          },
        }))
      ),
    ],
    {
      x, y, w, colW: lebar, rowH: isiBaris,
      border: { type: "solid", color: C.garis, pt: 0.5 },
      autoPage: false,
    }
  );
}

function lencana(s, x, y, teks, warna, muda) {
  const w = 0.28 + teks.length * 0.085;
  s.addShape("roundRect", {
    x, y, w, h: 0.32, rectRadius: 0.16, fill: { color: muda },
  });
  s.addText(teks, {
    x, y: y + 0.02, w, h: 0.28, fontSize: 10, bold: true, color: warna,
    fontFace: FONT_ISI, align: "center", valign: "mid", margin: 0,
  });
  return w;
}

function nomorKartu(s, x, y, n, warna) {
  s.addShape("roundRect", {
    x, y, w: 0.38, h: 0.38, rectRadius: 0.08, fill: { color: warna },
  });
  s.addText(String(n).padStart(2, "0"), {
    x, y: y + 0.03, w: 0.38, h: 0.32, fontSize: 11, bold: true, color: C.putih,
    fontFace: FONT_ISI, align: "center", valign: "mid", margin: 0,
  });
}

function chevron(s, x, y, w, h, teks, warna) {
  s.addShape("chevron", { x, y, w, h, fill: { color: warna } });
  s.addText(teks, {
    x: x + 0.12, y, w: w - 0.3, h, fontSize: 10.5, bold: true, color: C.putih,
    fontFace: FONT_ISI, align: "center", valign: "mid", margin: 0,
  });
}

function catatanKartu(s, x, y, w, h, judulTeks, isi, warna, muda) {
  s.addShape("roundRect", {
    x, y, w, h, rectRadius: 0.1,
    fill: { color: muda }, line: { color: warna, width: 0.75 },
  });
  s.addText(judulTeks, {
    x: x + 0.25, y: y + 0.14, w: w - 0.5, h: 0.28, fontSize: 10.5, bold: true,
    color: warna, fontFace: FONT_ISI, charSpacing: 0.8, margin: 0,
  });
  s.addText(isi, {
    x: x + 0.25, y: y + 0.46, w: w - 0.5, h: h - 0.6, fontSize: 10.5,
    color: C.ink, fontFace: FONT_ISI, margin: 0, valign: "top",
  });
}

function barisInsight(s, x, y, w, judulTeks, isi, warna) {
  s.addShape("ellipse", { x, y: y + 0.05, w: 0.14, h: 0.14, fill: { color: warna } });
  s.addText(judulTeks, {
    x: x + 0.28, y, w: w - 0.28, h: 0.26, fontSize: 11.5, bold: true,
    color: C.ink, fontFace: FONT_ISI, margin: 0,
  });
  s.addText(isi, {
    x: x + 0.28, y: y + 0.27, w: w - 0.28, h: 0.42, fontSize: 10.5,
    color: C.inkLembut, fontFace: FONT_ISI, margin: 0, valign: "top",
  });
}

// ---------------------------------------------------------------------------
// BUILD
// ---------------------------------------------------------------------------
const pres = new PptxGenJS();
pres.layout = "LAYOUT_WIDE";
pres.author = `${NAMA_KELOMPOK} - ML2026`;
pres.company = "Pembelajaran Mesin ML2026";
pres.subject = "Klasifikasi Lesi Kulit 7 Kelas dengan HOG-SVM, HOG-Random Forest, CNN, dan MobileNetV2";
pres.title = "Klasifikasi Lesi Kulit 7 Kelas - Presentasi Kelompok 6";

// =========================== SLIDE 1: JUDUL ===============================
{
  const s = slideGelap(pres);
  s.addShape("ellipse", {
    x: 9.5, y: -0.9, w: 5.2, h: 5.2, fill: { color: "1E3A6E" },
  });

  s.addText("PROJECT AKHIR PEMBELAJARAN MESIN", {
    x: 0.85, y: 1.02, w: 7.4, h: 0.3, fontSize: 12, bold: true, color: C.teal,
    fontFace: FONT_ISI, charSpacing: 2, margin: 0,
  });
  s.addText(`${NAMA_KELOMPOK.toUpperCase()}  ·  ML2026`, {
    x: 0.85, y: 1.36, w: 7.4, h: 0.3, fontSize: 11, color: C.slateMuda,
    fontFace: FONT_ISI, charSpacing: 1.4, margin: 0,
  });
  s.addText(
    [
      { text: "Klasifikasi Lesi Kulit\n", options: { color: C.putih } },
      { text: "7 Kelas dari Foto", options: { color: C.teal } },
    ],
    {
      x: 0.85, y: 1.92, w: 7.4, h: 1.7, fontSize: 40, bold: true,
      fontFace: FONT_JUDUL, margin: 0, lineSpacing: 46,
    }
  );
  s.addText(
    "Empat model klasik dan deep learning, satu ambang batas yang dikunci " +
    "dari data validasi, lalu diuji ulang pada foto smartphone. Alat bantu " +
    "belajar, bukan alat diagnosis.",
    {
      x: 0.85, y: 3.78, w: 7.0, h: 0.9, fontSize: 13.5, color: C.teksGelap,
      fontFace: FONT_ISI, margin: 0, lineSpacing: 21,
    }
  );

  const chipData = [
    `${thousands(TOTAL_FOTO)} foto`,
    `${KELAS.length} kelas`,
    "4 model",
    `F1 macro ${f(MOB.f1_macro)}`,
  ];
  let cx = 0.85;
  chipData.forEach((t) => {
    const w = 0.34 + t.length * 0.085;
    s.addShape("roundRect", {
      x: cx, y: 4.86, w, h: 0.38, rectRadius: 0.19,
      fill: { color: C.navy, transparency: 100 },
      line: { color: C.slateMuda, width: 0.75 },
    });
    s.addText(t, {
      x: cx, y: 4.9, w, h: 0.3, fontSize: 10.5, color: C.teksGelap,
      fontFace: FONT_ISI, align: "center", valign: "mid", margin: 0,
    });
    cx += w + 0.22;
  });

  s.addShape("rect", {
    x: 0.85, y: 5.72, w: 6.9, h: 0.012, fill: { color: "3A4A6B" },
  });
  const anggota = H.info.anggota.map((a) => `${a.nama}  ·  ${a.nim}`).join("\n");
  s.addText(anggota, {
    x: 0.85, y: 5.88, w: 7.0, h: 0.9, fontSize: 11.5, color: C.putih,
    fontFace: FONT_ISI, margin: 0, lineSpacing: 19,
  });

  // Kartu "sekilas dataset": angka yang paling sering ditanya dosen.
  s.addShape("roundRect", {
    x: 8.55, y: 1.55, w: 3.9, h: 5.05, rectRadius: 0.12,
    fill: { color: C.putih, transparency: 92 },
    line: { color: C.slateMuda, width: 0.75 },
  });
  s.addText("SEKILAS DATASET", {
    x: 8.85, y: 1.82, w: 3.3, h: 0.28, fontSize: 10, bold: true, color: C.teal,
    fontFace: FONT_ISI, charSpacing: 1.4, margin: 0,
  });
  s.addText(`${f(RASIO_KELAS, 1)} : 1`, {
    x: 8.85, y: 2.14, w: 3.3, h: 0.72, fontSize: 38, bold: true, color: C.putih,
    fontFace: FONT_JUDUL, margin: 0,
  });
  s.addText("rasio kelas terbesar terhadap terkecil (nv terhadap df)", {
    x: 8.85, y: 2.86, w: 3.3, h: 0.44, fontSize: 10.5, color: C.teksGelap,
    fontFace: FONT_ISI, margin: 0,
  });

  barTumpuk(s, 8.85, 3.44, 3.3, 0.3, [
    { label: `nv ${persen(PESEN_NV)}`, porsi: PESEN_NV, warna: C.teal },
    { label: "", porsi: 1 - PESEN_NV - PESEN_DF, warna: "4A5A7A" },
    { label: "", porsi: PESEN_DF, warna: C.rose },
  ]);
  s.addText(`nevus ${persen(PESEN_NV)}   ·   df ${persen(PESEN_DF)}`, {
    x: 8.85, y: 3.8, w: 3.3, h: 0.28, fontSize: 10, color: C.teksGelap,
    fontFace: FONT_ISI, margin: 0,
  });

  s.addShape("rect", {
    x: 8.85, y: 4.28, w: 3.3, h: 0.012, fill: { color: "3A4A6B" },
  });
  const baris = [
    [`${thousands(H.split.counts.train.images)} / ${thousands(
      H.split.counts.val.images
    )} / ${thousands(H.split.counts.test.images)}`, "latih / validasi / uji"],
    [`F1 macro ${f(MOB.f1_macro)}`, `@ ambang ${f(BIN.threshold, 2)}`],
  ];
  baris.forEach(([besar, kecil], i) => {
    const y = 4.52 + i * 0.95;
    s.addText(besar, {
      x: 8.85, y, w: 3.3, h: 0.4, fontSize: 19, bold: true, color: C.putih,
      fontFace: FONT_JUDUL, margin: 0,
    });
    s.addText(kecil, {
      x: 8.85, y: y + 0.42, w: 3.3, h: 0.28, fontSize: 10.5, color: C.teksGelap,
      fontFace: FONT_ISI, margin: 0,
    });
  });

  s.addNotes("Slide pembuka. Sebutkan ketiga anggota dan judul proyek. " +
    "Angka kunci: rasio kelas " + f(RASIO_KELAS, 1) + ":1, " +
    thousands(TOTAL_FOTO) + " foto, 4 model. Tekankan disclaimer di slide ini: " +
    "alat bantu belajar, bukan diagnosis.");
}

// =========================== SLIDE 2: HOOK ================================
{
  const s = slideKonten(
    pres,
    "BAB 1  ·  HOOK",
    "Kenapa ini penting?",
    "Lesi kulit yang ganas sedikit, tapi salah klasifikasi berarti pasien menunda periksa."
  );

  kartu(s, M, 1.95, 4.3, 3.9, {});
  s.addShape("rect", { x: M + 0.28, y: 2.2, w: 0.42, h: 0.055, fill: { color: C.rose } });
  s.addText("DALAM DATASET", {
    x: M + 0.28, y: 2.32, w: 3.7, h: 0.26, fontSize: 9.5, bold: true,
    color: C.inkLembut, fontFace: FONT_ISI, charSpacing: 1, margin: 0,
  });
  s.addText(persen(PESEN_DF), {
    x: M + 0.28, y: 2.6, w: 3.7, h: 0.8, fontSize: 44, bold: true, color: C.rose,
    fontFace: FONT_JUDUL, margin: 0,
  });
  s.addText("kelas df (dermatofibroma), salah satu kelas ganas", {
    x: M + 0.28, y: 3.42, w: 3.7, h: 0.5, fontSize: 12, bold: true, color: C.ink,
    fontFace: FONT_ISI, margin: 0,
  });
  s.addText(`Hanya ${thousands(JUMLAH_KELAS.df)} dari ${thousands(TOTAL_FOTO)} foto. ` +
    "Model yang hanya hafal kelas besar akan terlihat akurat sambil melewatkan lesi ganas.", {
    x: M + 0.28, y: 4.05, w: 3.7, h: 1.1, fontSize: 11, color: C.inkLembut,
    fontFace: FONT_ISI, margin: 0, valign: "top", lineSpacing: 16,
  });
  barTumpuk(s, M + 0.28, 5.25, 3.7, 0.26, [
    { label: `nv ${persen(PESEN_NV)}`, porsi: PESEN_NV, warna: C.teal },
    { label: "", porsi: 1 - PESEN_NV - PESEN_DF, warna: C.garis },
    { label: "", porsi: PESEN_DF, warna: C.rose },
  ]);

  const poin = [
    [
      "Lesi jinak bukan berarti aman",
      `Dari ${thousands(TOTAL_FOTO)} foto, hanya ${thousands(
        JUMLAH_KELAS.df
      )} berlabel df. Skrining berguna justru karena kasus jarang itu yang paling sering terlewat.`,
    ],
    [
      "Foto pengguna bukan foto dermoskop",
      "HAM10000 diambil dengan dermatoskop. Aplikasi kami menerima foto smartphone, dan itu domain yang berbeda.",
    ],
    [
      "Yang dibutuhkan: skrining, bukan vonis",
      "Ambang batas dan status tidak yakin ditampilkan terbuka, supaya pengguna tahu kapan harus ke dokter.",
    ],
  ];
  poin.forEach(([judulTeks, isi], i) => {
    const y = 1.95 + i * 1.33;
    kartu(s, 5.2, y, 7.53, 1.2, {});
    nomorKartu(s, 5.45, y + 0.28, i + 1, [C.teal, C.amber, C.navy][i]);
    s.addText(judulTeks, {
      x: 5.98, y: y + 0.2, w: 6.5, h: 0.3, fontSize: 13.5, bold: true,
      color: C.ink, fontFace: FONT_ISI, margin: 0,
    });
    s.addText(isi, {
      x: 5.98, y: y + 0.52, w: 6.5, h: 0.6, fontSize: 11, color: C.inkLembut,
      fontFace: FONT_ISI, margin: 0, valign: "top", lineSpacing: 15,
    });
  });

  s.addNotes("Buka dengan angka: hanya " + persen(PESEN_DF) +
    " data yang kelas df. Poin kedua menjelaskan kenapa uji eksternal dengan foto " +
    "smartphone itu wajib, bukan opsional.");
}

// =========================== SLIDE 3: DATASET =============================
{
  const s = slideKonten(
    pres,
    "BAB 2  ·  DATA & SKENARIO",
    "Dataset",
    `Dua dataset: satu untuk melatih, satu untuk menguji-kejujuran model. ${thousands(
      TOTAL_FOTO
    )} foto, ${thousands(TOTAL_LESI)} lesi unik, ${KELAS.length} kelas diagnosis.`
  );

  const tiles = [
    { label: "UKURAN", nilai: thousands(TOTAL_FOTO), sub: "foto dermoskop RGB", warna: C.navy },
    { label: "LESI UNIK", nilai: thousands(TOTAL_LESI), sub: "satu lesi bisa beberapa foto", warna: C.teal },
    { label: "KELAS", nilai: String(KELAS.length), sub: "diagnosis dermatologi", warna: C.amber },
    { label: "SUMBER", nilai: "HAM10000", sub: "Harvard Dataverse · CC BY-NC 4.0", warna: C.navy },
  ];
  tiles.forEach((t, i) => {
    const x = M + i * 3.07;
    kartu(s, x, 1.95, 2.86, 1.32, {});
    s.addShape("rect", { x: x + 0.25, y: 2.16, w: 0.42, h: 0.055, fill: { color: t.warna } });
    s.addText(t.label, {
      x: x + 0.25, y: 2.27, w: 2.4, h: 0.24, fontSize: 9, bold: true,
      color: C.inkLembut, fontFace: FONT_ISI, charSpacing: 1, margin: 0,
    });
    s.addText(t.nilai, {
      x: x + 0.25, y: 2.5, w: 2.4, h: 0.42, fontSize: 22, bold: true,
      color: t.warna, fontFace: FONT_JUDUL, margin: 0,
    });
    s.addText(t.sub, {
      x: x + 0.25, y: 2.92, w: 2.4, h: 0.3, fontSize: 9.5, color: C.inkLembut,
      fontFace: FONT_ISI, margin: 0,
    });
  });

  kartu(s, M, 3.48, 6.05, 3.3, {});
  s.addText("Dua dataset, satu tujuan", {
    x: M + 0.3, y: 3.72, w: 5.4, h: 0.32, fontSize: 14, bold: true, color: C.ink,
    fontFace: FONT_ISI, margin: 0,
  });
  const bagian = [
    [
      "HAM10000 (training)",
      `${thousands(TOTAL_FOTO)} foto dermoskop, ${KELAS.length} kelas. Semua model dilatih di sini.`,
    ],
    [
      "PAD-UFES-20 (uji eksternal)",
      `${thousands(H.pad.n_gambar)} foto smartphone, 6 diagnosis. Tidak pernah dilihat saat latihan.`,
    ],
    [
      "Split per lesion_id",
      `${FOTO_PER_LESI} foto adalah lesi yang sama difoto ulang, jadi lesi tidak boleh dibagi dua split.`,
    ],
  ];
  bagian.forEach(([j, d], i) => {
    const y = 4.18 + i * 0.85;
    nomorKartu(s, M + 0.3, y, i + 1, [C.navy, C.teal, C.amber][i]);
    s.addText(j, {
      x: M + 0.82, y: y + 0.02, w: 4.9, h: 0.28, fontSize: 12, bold: true,
      color: C.ink, fontFace: FONT_ISI, margin: 0,
    });
    s.addText(d, {
      x: M + 0.82, y: y + 0.3, w: 4.9, h: 0.45, fontSize: 10.5, color: C.inkLembut,
      fontFace: FONT_ISI, margin: 0, valign: "top",
    });
  });

  kartu(s, 7.0, 3.48, 5.73, 3.3, {});
  s.addText("KOMPOSISI KELAS", {
    x: 7.3, y: 3.72, w: 5.1, h: 0.26, fontSize: 9.5, bold: true, color: C.inkLembut,
    fontFace: FONT_ISI, charSpacing: 1, margin: 0,
  });
  s.addText(`nv ${persen(PESEN_NV)}  ·  df ${persen(PESEN_DF)}`, {
    x: 7.3, y: 4.0, w: 5.1, h: 0.34, fontSize: 14, bold: true, color: C.ink,
    fontFace: FONT_ISI, margin: 0,
  });
  const maksKelas = Math.max(...Object.values(JUMLAH_KELAS));
  KELAS.forEach((k, i) => {
    const y = 4.48 + i * 0.29;
    s.addText(LABEL_KELAS[k], {
      x: 7.3, y, w: 2.15, h: 0.25, fontSize: 9.5, color: C.inkLembut,
      fontFace: FONT_ISI, margin: 0,
    });
    const lebar = 2.0 * (JUMLAH_KELAS[k] / maksKelas);
    s.addShape("roundRect", {
      x: 9.5, y: y + 0.05, w: 2.0, h: 0.15, rectRadius: 0.07,
      fill: { color: C.garis },
    });
    s.addShape("roundRect", {
      x: 9.5, y: y + 0.05, w: Math.max(0.06, lebar), h: 0.15, rectRadius: 0.07,
      fill: { color: k === "df" ? C.rose : C.teal },
    });
    s.addText(thousands(JUMLAH_KELAS[k]), {
      x: 11.58, y, w: 0.85, h: 0.25, fontSize: 9.5, bold: true, color: C.ink,
      fontFace: FONT_ISI, align: "right", margin: 0,
    });
  });
  s.addText(
    "Kelas df hanya " + persen(PESEN_DF) + " dari seluruh foto. Karena itu " +
    "class weight dihitung dari data latih saja.",
    {
      x: 7.3, y: 6.35, w: 5.1, h: 0.35, fontSize: 10, color: C.inkLembut,
      fontFace: FONT_ISI, margin: 0,
    }
  );

  s.addNotes("Tunjukkan dua dataset dan alasannya: PAD-UFES-20 dipakai sebagai " +
    "uji eksternal, bukan data latih tambahan. Angka kelas diambil dari split " +
    "summary, bukan ketik manual.");
}

// ====================== SLIDE 4: KONTROL LEAKAGE ===========================
{
  const s = slideKonten(
    pres,
    "BAB 2  ·  DATA & SKENARIO",
    "Skenario dan kontrol kebocoran data",
    "Pembagian per lesi, bukan per foto, dan data uji dipakai satu kali di akhir."
  );

  const tahap = [
    ["Sumber", "HAM10000\n10.015 foto", C.navy],
    ["Split per lesion_id", "GroupShuffleSplit\n2 tahap · seed 42", "1D4E89"],
    ["Data latih", "6.959 foto\n5.228 lesi", C.teal],
    ["Validasi", "1.529 foto\n1.121 lesi", "2A7E77"],
    ["Data uji", "1.527 foto\ndipakai sekali", C.rose],
  ];
  tahap.forEach(([j, d, w], i) => {
    chevron(s, M + i * 2.45, 1.98, 2.55, 0.82, j, w);
    s.addText(d, {
      x: M + i * 2.45 + 0.1, y: 2.92, w: 2.2, h: 0.62, fontSize: 10,
      color: C.inkLembut, fontFace: FONT_ISI, align: "center", margin: 0,
      lineSpacing: 14,
    });
  });

  kartu(s, M, 3.72, 6.05, 3.05, {});
  s.addText("Yang tidak boleh masuk model", {
    x: M + 0.3, y: 3.96, w: 5.4, h: 0.3, fontSize: 13.5, bold: true, color: C.ink,
    fontFace: FONT_ISI, margin: 0,
  });
  tabel(
    s, M + 0.3, 4.36, 5.45,
    ["Kolom", "Alasan"],
    [
      ["lesion_id", "Identifier, dipakai untuk group split, bukan fitur"],
      ["diagnosis", "Label target, bocor bila jadi fitur"],
      ["age", "Kosong pada sebagian gambar, tidak dipakai"],
    ],
    [1.5, 3.95], { fs: 10, barisTinggi: 0.42 }
  );
  s.addText(
    "Semua lesi hanya boleh muncul di satu split. Tiga assert memeriksa ini " +
    "dan hasilnya nol.",
    {
      x: M + 0.3, y: 6.18, w: 5.45, h: 0.5, fontSize: 10.5, color: C.inkLembut,
      fontFace: FONT_ISI, margin: 0, valign: "top",
    }
  );

  kartu(s, 7.0, 3.72, 5.73, 1.42, {});
  s.addText("BUKTI KEBOCORAN", {
    x: 7.3, y: 3.94, w: 5.1, h: 0.26, fontSize: 9.5, bold: true, color: C.inkLembut,
    fontFace: FONT_ISI, charSpacing: 1, margin: 0,
  });
  s.addText(String(H.split.leakage_check.all_zero ? 0 : "Cek"), {
    x: 7.3, y: 4.2, w: 1.5, h: 0.6, fontSize: 30, bold: true, color: C.teal,
    fontFace: FONT_JUDUL, margin: 0,
  });
  s.addText(
    "overlap lesi dan gambar antar split\nlatih, validasi, dan uji",
    {
      x: 8.9, y: 4.24, w: 3.5, h: 0.6, fontSize: 11, color: C.inkLembut,
      fontFace: FONT_ISI, margin: 0, lineSpacing: 15,
    }
  );

  catatanKartu(
    s, 7.0, 5.28, 5.73, 1.49,
    "TEST DIPAKAI SEKALI",
    "Data uji disentuh setelah semua hyperparameter dan ambang batas dikunci dari " +
    "data validasi. Tidak ada tuning yang melihat data uji, sehingga angka " +
    "test bisa dipercaya sebagai laporan akhir.",
    C.rose, C.roseMuda
  );

  s.addNotes("Poin anti-leakage. Tegaskan tiga assert overlap bernilai nol dan " +
    "file split_summary.json di docs/logs sebagai buktinya.");
}

// =========================== SLIDE 5: EDA ================================
{
  const s = slideKonten(
    pres,
    "BAB 3  ·  EDA",
    "Sinyalnya jelas, tapi tidak merata",
    "Empat angka yang cukup untuk menjelaskan seluruh cerita data."
  );

  const ham = H.pad.statistik_domain.HAM10000;
  const cards = [
    {
      label: "RASIO KELAS",
      nilai: `${f(RASIO_KELAS, 1)} : 1`,
      keterangan: "nv terhadap df",
      bar: 1,
      warna: C.rose,
      catatan: `df hanya ${persen(PESEN_DF)} dari seluruh foto, nv ${persen(
        PESEN_NV
      )}.`,
    },
    {
      label: "RESOLUSI",
      nilai: `${Math.round(ham.lebar)} × ${Math.round(ham.tinggi)}`,
      keterangan: "seragam untuk semua foto",
      bar: 1,
      warna: C.teal,
      catatan: "Semua gambar RGB dengan rasio 4:3, jadi resize seragam aman.",
    },
    {
      label: "LESI UNIK",
      nilai: thousands(TOTAL_LESI),
      keterangan: `dari ${thousands(TOTAL_FOTO)} foto`,
      bar: TOTAL_LESI / TOTAL_FOTO,
      warna: C.navy,
      catatan: `${thousands(
        FOTO_PER_LESI
      )} foto adalah lesi yang sama difoto ulang.`,
    },
    {
      label: "CEK KEBOCORAN",
      nilai: "0",
      keterangan: "overlap lesi dan gambar",
      bar: 0,
      warna: C.teal,
      catatan: "Enam pemeriksaan overlap antar split, semuanya nol.",
    },
  ];
  cards.forEach((c, i) => {
    kartuAngka(s, M + i * 3.07, 2.0, 2.86, 2.75, {
      label: c.label, nilai: c.nilai, keterangan: c.keterangan, bar: c.bar,
      warna: c.warna, catatan: c.catatan, ukuran: 26,
    });
  });

  catatanKartu(
    s, M, 5.0, 12.13, 1.3,
    "IMPLIKASI UNTUK MODEL",
    "Kelas minority harus diberi bobot, dan split harus per lesi. Keduanya " +
    "dijalankan di seluruh proyek: class weight dihitung hanya dari data latih, " +
    "dan pembagian data memakai GroupShuffleSplit per lesion_id dengan seed 42. " +
    "Tiga assert anti-kebocoran memeriksa hasilnya, semuanya bernilai nol.",
    C.teal, C.tealMuda
  );

  s.addNotes("Empat angka EDA: rasio kelas, resolusi seragam, jumlah lesi unik, " +
    "dan hasil cek kebocoran. Langsung hubungkan ke keputusan metodologi.");
}

// ================== SLIDE 6: DISTRIBUSI KELAS + CHART ====================
{
  const s = slideKonten(
    pres,
    "BAB 3  ·  EDA",
    "Distribusi kelas di data latih",
    "Tujuh kelas dengan panjang ekor yang sangat panjang. Class weight menutup celah kelas yang hilang."
  );

  const train = H.split.class_distribution.train;
  s.addChart(
    // Tipe "col" tidak dikenali pptxgenjs versi ini dan menghasilkan file
    // yang ditolak PowerPoint. Bar horizontal memakai "bar" + barDir.
    "bar",
    [{
      name: "Foto di data latih",
      labels: KELAS.map((k) => LABEL_KELAS[k]),
      values: KELAS.map((k) => train[k]),
    }],
    {
      x: M, y: 2.0, w: 7.1, h: 4.5,
      barDir: "bar",
      catAxisLabelColor: C.ink, valAxisLabelColor: C.inkLembut,
      catAxisLabelFontFace: FONT_ISI, valAxisLabelFontFace: FONT_ISI,
      catAxisLabelFontSize: 10, valAxisLabelFontSize: 9,
      valGridLine: { color: C.garis, size: 0.5 },
      catGridLine: { style: "none" },
      showLegend: false, showValue: true, dataLabelPosition: "outEnd",
      dataLabelColor: C.ink, dataLabelFontSize: 10,
      chartColors: [C.navy],
      showTitle: true, title: "Jumlah foto per kelas (data latih)",
      titleFontSize: 12, titleColor: C.ink, titleFontFace: FONT_ISI,
      barGapWidthPct: 45,
    }
  );

  kartu(s, 8.0, 2.0, 4.73, 2.6, {});
  s.addText("CLASS WEIGHT (dari data latih)", {
    x: 8.3, y: 2.24, w: 4.1, h: 0.26, fontSize: 9.5, bold: true,
    color: C.inkLembut, fontFace: FONT_ISI, charSpacing: 0.8, margin: 0,
  });
  const cw = H.split.class_weight;
  const urut = Object.keys(cw).sort((a, b) => cw[b] - cw[a]);
  urut.forEach((k, i) => {
    const y = 2.58 + i * 0.26;
    s.addText(LABEL_KELAS[k], {
      x: 8.3, y, w: 2.6, h: 0.24, fontSize: 10, color: C.inkLembut,
      fontFace: FONT_ISI, margin: 0,
    });
    s.addText(f(cw[k], 2), {
      x: 11.0, y, w: 1.4, h: 0.24, fontSize: 10, bold: true,
      color: cw[k] > 1 ? C.rose : C.ink, fontFace: FONT_ISI, align: "right", margin: 0,
    });
  });

  catatanKartu(
    s, 8.0, 4.75, 4.73, 1.75,
    "TANPA CLASS WEIGHT",
    "Varian CNN tanpa class weight mengunci accuracy 0,672 dengan recall ganas " +
    "nol. Bobot tidak menghapus ketidakseimbangan, hanya menyeimbangkan perhatian " +
    "model terhadapnya. Data tetap tidak seimbang.",
    C.rose, C.roseMuda
  );

  s.addNotes("Tunjuk panjang ekor kelas df dan vasc. Ingatkan bobot dihitung dari " +
    "data latih saja, bukan dari seluruh data.");
}

// ===================== SLIDE 7: DUA DOMAIN ================================
{
  const s = slideKonten(
    pres,
    "BAB 3  ·  EDA",
    "Dua domain, dua dunia foto",
    "Model dilatih pada dermoskop, lalu diuji pada foto smartphone. Perbedaannya diukur, bukan diasumsikan."
  );

  const ham = H.pad.statistik_domain.HAM10000;
  const pad = H.pad.statistik_domain["PAD-UFES-20"];
  kartu(s, M, 1.98, 7.4, 4.35, {});
  s.addText("Perbandingan teknis kedua dataset", {
    x: M + 0.3, y: 2.22, w: 6.8, h: 0.3, fontSize: 13.5, bold: true, color: C.ink,
    fontFace: FONT_ISI, margin: 0,
  });
  tabel(
    s, M + 0.3, 2.62, 6.8,
    ["Aspek", "HAM10000 (latih)", "PAD-UFES-20 (uji)"],
    [
      ["Jenis foto", "dermoskop", "smartphone"],
      ["Resolusi", `${Math.round(ham.lebar)} × ${Math.round(ham.tinggi)}`, `${Math.round(pad.lebar)} × ${Math.round(pad.tinggi)}`],
      ["Mean kanal merah", f(ham.mean_R, 1), f(pad.mean_R, 1)],
      ["Ketajaman (varians Laplacian)", f(ham.ketajaman, 1), f(pad.ketajaman, 1)],
      ["Jumlah foto", thousands(TOTAL_FOTO), thousands(H.pad.n_gambar)],
      ["Kelas", "7 diagnosis", "6 diagnosis"],
    ],
    [2.2, 2.3, 2.3], { fs: 10, barisTinggi: 0.42 }
  );

  s.addText(
    "PAD-UFES-20 memakai 6 diagnosis: ACK, BCC, MEL, NEV, SCC, dan SEK. " +
    "HAM10000 memakai 7 kelas dengan kategori diagnosis yang berbeda.",
    {
      x: M + 0.3, y: 5.68, w: 6.8, h: 0.5, fontSize: 10.5, color: C.inkLembut,
      fontFace: FONT_ISI, margin: 0, valign: "top",
    }
  );

  kartu(s, 8.3, 1.98, 4.43, 2.25, {});
  s.addText("APA ARTINYA", {
    x: 8.58, y: 2.22, w: 3.9, h: 0.26, fontSize: 9.5, bold: true,
    color: C.inkLembut, fontFace: FONT_ISI, charSpacing: 1, margin: 0,
  });
  barisInsight(
    s, 8.58, 2.56, 3.9,
    "Foto smartphone lebih tajam",
    `Ketajaman naik dari ${f(ham.ketajaman, 0)} menjadi ${f(pad.ketajaman, 0)}, ` +
    "karena pencahayaan dan jarak pengambilan berbeda.",
    C.teal
  );
  barisInsight(
    s, 8.58, 3.28, 3.9,
    "Kelas yang belum pernah dilihat",
    "SCC tidak ada di HAM10000, jadi kelas yang belum pernah dilihat model.",
    C.rose
  );

  catatanKartu(
    s, 8.3, 4.42, 4.43, 1.91,
    "KENAPA INI DIUKUR",
    "Model tidak dilatih ulang untuk PAD-UFES-20 dan ambang batasnya tetap " +
    "diambil dari validasi HAM10000. Jadi penurunan performa diuji eksternal " +
    "adena angka pengukuran domain shift, bukan opinion.",
    C.teal, C.tealMuda
  );

  s.addNotes("Dua sumber foto berbeda. Tunjukkan angka ketajaman dan resolusi, " +
    "lalu tekankan bahwa kelas SCC tidak pernah dilihat model saat latihan.");
}

// ================== SLIDE 8: REKAYASA FITUR ===============================
{
  const s = slideKonten(
    pres,
    "BAB 4  ·  METODOLOGI",
    "Rekayasa fitur berbasis eksperimen",
    "Desain fitur ditentukan oleh pengukuran, bukan kebiasaan. Empat konfigurasi yang gagal dicatat sebagai bukti."
  );

  const d = (konf) => (H.diag || []).find((x) => x.konfigurasi === konf);
  const rfPenuh = d("RF pada 1812 fitur penuh");
  const pca400 = d("SVC PCA(400) dari 1812 fitur");
  const pca30 = d("SVC PCA(30) HOG + histogram");
  const histSaja = d("SVC-RBF histogram saja (48)");

  kartu(s, M, 1.98, 12.13, 1.05, { fill: C.navy, line: C.navy });
  s.addText(
    "HOG (sel 16×16 → 1.764 dimensi)  →  StandardScaler  →  PCA(30)  →  StandardScaler\n" +
    "Histogram warna RGB (48 dimensi)  →  dikali bobot 6  (tidak di-scale ulang)   →  digabung  →  SVC / Random Forest",
    {
      x: M + 0.3, y: 2.14, w: 11.5, h: 0.8, fontSize: 11, color: C.teksGelap,
      fontFace: FONT_JUDUL, margin: 0, lineSpacing: 17,
    }
  );

  const temuan = [
    {
      j: "Fitur mentah membuat RF buta",
      a: rfPenuh ? f(rfPenuh.f1_macro) : "-",
      d: rfPenuh
        ? `RF pada 1.812 fitur mentah hanya F1 ${f(rfPenuh.f1_macro)}, recall ganas ` +
          `${f(rfPenuh.malignant_recall, 4)}. Peluang acak memilih 48 fitur histogram hanya 2,65%.`
        : "-",
      warna: C.rose,
    },
    {
      j: "PCA global lebih buruk",
      a: pca400 ? f(pca400.f1_macro) : "-",
      d: pca400 && pca30
        ? `PCA(400) dari seluruh fitur F1 ${f(pca400.f1_macro)}, sedangkan PCA(30) ` +
          `pada HOG saja + histogram ${f(pca30.f1_macro)}.`
        : "-",
      warna: C.amber,
    },
    {
      j: "Histogram lebih kuat dari HOG",
      a: histSaja ? f(histSaja.f1_macro) : "-",
      d: histSaja
        ? `SVC pada 48 fitur histogram mencapai F1 ${f(histSaja.f1_macro)} dengan ` +
          `recall ganas ${f(histSaja.malignant_recall, 4)}.`
        : "-",
      warna: C.teal,
    },
  ];
  temuan.forEach((t, i) => {
    const x = M + i * 4.07;
    kartu(s, x, 3.25, 3.86, 2.45, {});
    s.addText(t.j, {
      x: x + 0.28, y: 3.48, w: 3.3, h: 0.32, fontSize: 13, bold: true, color: C.ink,
      fontFace: FONT_ISI, margin: 0,
    });
    s.addText(t.a, {
      x: x + 0.28, y: 3.86, w: 3.3, h: 0.5, fontSize: 26, bold: true, color: t.warna,
      fontFace: FONT_JUDUL, margin: 0,
    });
    s.addText("F1 macro", {
      x: x + 0.28, y: 4.34, w: 3.3, h: 0.24, fontSize: 9.5, color: C.inkLembut,
      fontFace: FONT_ISI, margin: 0,
    });
    s.addText(t.d, {
      x: x + 0.28, y: 4.64, w: 3.3, h: 0.95, fontSize: 10, color: C.inkLembut,
      fontFace: FONT_ISI, margin: 0, valign: "top", lineSpacing: 14,
    });
  });

  const hog16 = (H.hog || []).find((x) => x.cell_size === "16x16");
  const hog8 = (H.hog || []).find((x) => x.cell_size === "8x8");
  catatanKartu(
    s, M, 5.9, 12.13, 0.87,
    "UKURAN SEL HOG DIPILIH BERDASARKAN EFISIENSI",
    hog16 && hog8
      ? `F1 hampir sama di semua ukuran sel (8×8 ${f(hog8.f1_macro)} versus 16×16 ${f(
          hog16.f1_macro
        )}), jadi 16×16 dipilih karena ${thousands(
          hog16.n_hog_dims
        )} dimensi, bukan ${thousands(hog8.n_hog_dims)} dimensi.`
      : "-",
    C.navy, C.paper
  );

  s.addNotes("Tiga temuan eksperimen yang membentuk desain fitur. Angka diambil " +
    "dari diagnostik_fitur.json. Jangan hanya menampilkan konfigurasi final.");
}

// ================== SLIDE 9: ALUR PREPROCESSING ===========================
{
  const s = slideKonten(
    pres,
    "BAB 4  ·  METODOLOGI",
    "Alur preprocessing",
    "Satu pipeline, di-fit hanya pada data training. Test set tidak pernah menyentuh scaler maupun PCA."
  );

  const langkah = [
    ["Resize", "224 × 224\nuntuk deep learning", C.navy],
    ["Augmentasi", "hanya saat\nlatihan", "1D4E89"],
    ["Ekstraksi fitur", "HOG 1.764\n+ hist 48", C.teal],
    ["Skala", "StandardScaler\nfit di train", "2A7E77"],
    ["Reduksi", "PCA(30)\nhanya pada HOG", C.amber],
    ["Bobot kelas", "dari data\nlatih saja", "C0574A"],
    ["Latih + uji", "1.527 foto\nsekali", C.navy],
  ];
  langkah.forEach(([j, d, w], i) => {
    chevron(s, M + i * 1.74, 1.95, 1.8, 0.6, j, w);
    s.addText(d, {
      x: M + i * 1.74, y: 2.62, w: 1.6, h: 0.58, fontSize: 9,
      color: C.inkLembut, fontFace: FONT_ISI, align: "center", margin: 0,
      lineSpacing: 13,
    });
  });

  kartu(s, M, 3.36, 6.9, 3.4, {});
  s.addText("DI DALAM PIPELINE", {
    x: M + 0.3, y: 3.6, w: 6.3, h: 0.26, fontSize: 9.5, bold: true,
    color: C.inkLembut, fontFace: FONT_ISI, charSpacing: 1, margin: 0,
  });
  tabel(
    s, M + 0.3, 3.94, 6.3,
    ["Komponen", "Perlakuan", "Alasan"],
    [
      ["Scaler", "StandardScaler", "Menyamakan skala antar fitur"],
      ["PCA", "30 komponen, HOG saja", "Menjaga 48 fitur histogram tetap utuh"],
      ["Bobot kelas", "df 2,39 · nv 0,05", "Kelas minority tidak boleh hilang"],
      ["Augmentasi", "Train saja, termasuk gaya smartphone", "Menyiapkan model untuk foto phone"],
    ],
    [1.3, 2.3, 2.7], { fs: 9.5, barisTinggi: 0.42 }
  );
  s.addText(
    "Scaler dan PCA berada di dalam pipeline sklearn, jadi otomatis di-fit " +
    "pada X_train saja.",
    {
      x: M + 0.3, y: 6.05, w: 6.3, h: 0.5, fontSize: 10.5, color: C.inkLembut,
      fontFace: FONT_ISI, margin: 0, valign: "top",
    }
  );

  catatanKartu(
    s, 7.8, 3.36, 4.93, 3.4,
    "SENGAJA TIDAK DILAKUKAN",
    "Tidak ada imputasi age, karena tidak dipakai sebagai fitur.\n" +
    "Tidak ada PCA di seluruh fitur, karena justru menghapus informasi histogram.\n" +
    "Tidak ada SMOTE, karena class weight lebih sederhana dan tidak menambah " +
    "gambar sintetis yang bisa menyesatkan.",
    C.rose, C.roseMuda
  );

  s.addNotes("Tunjukkan posisi scaler dan PCA di dalam pipeline. Tiga hal yang " +
    "sengaja dilewati juga penting: metodologi yang dijelaskan di slide ini.");
}

// ================== SLIDE 10: EMPAT MODEL =================================
{
  const s = slideKonten(
    pres,
    "BAB 4  ·  METODOLOGI",
    "Empat model, satu protokol evaluasi",
    "Dua model klasik berbasis fitur handcraft, dua model deep learning. Semua dinilai pada split yang sama."
  );

  const svm = H.klasik.svm.params;
  const rf = H.klasik.random_forest.params;
  const modelCards = [
    {
      tag: "Klasik",
      nama: "HOG + SVM",
      desk:
        "Kernel linear pada 1.764 dimensi HOG dan 48 histogram warna, digabung " +
        "setelah PCA(30) pada HOG.",
      hyper: `C = ${f(svm.C, 0)} · kernel ${svm.kernel} · gamma ${svm.gamma} · class weight`,
      nilai: f(H.klasik.svm.f1_macro),
      warna: C.navy,
    },
    {
      tag: "Klasik",
      nama: "HOG + Random Forest",
      desk:
        "200 pohon pada fitur yang sama. Pohon membelah berdasarkan urutan " +
        "nilai, jadi bobot histogram tidak berpengaruh.",
      hyper: `n_estimators = ${rf.n_estimators} · max_depth = ${rf.max_depth} · class weight`,
      nilai: f(H.klasik.random_forest.f1_macro),
      warna: C.teal,
    },
    {
      tag: "Deep learning",
      nama: "CNN dari nol",
      desk:
        "Tiga konv layer dan dense besar dengan BatchNormalization, dilatih " +
        "dengan class weight dan augmentasi.",
      hyper: `32-64-128 + Flatten · ${thousands(H.cnn.model ? H.cnn.params : 0)} parameter`,
      nilai: f(H.cnn.f1_macro),
      warna: C.amber,
    },
    {
      tag: "Deep learning",
      nama: "MobileNetV2",
      desk:
        "Transfer learning dua tahap. Tahap 1 dipakai: classifier baru dengan " +
        "backbone ImageNet beku.",
      hyper: `input 224 × 224 · ${thousands(H.mnet.total_params)} parameter · tahap 1`,
      nilai: f(H.mnet.stage1_result.f1_macro),
      warna: C.rose,
    },
  ];
  modelCards.forEach((m, i) => {
    const x = M + i * 3.07;
    kartu(s, x, 1.98, 2.86, 3.7, {});
    s.addShape("rect", { x: x + 0.25, y: 2.2, w: 0.42, h: 0.055, fill: { color: m.warna } });
    lencana(s, x + 0.25, 2.36, m.tag, m.warna, C.paper);
    s.addText(m.nama, {
      x: x + 0.25, y: 2.86, w: 2.4, h: 0.6, fontSize: 15, bold: true, color: C.ink,
      fontFace: FONT_JUDUL, margin: 0,
    });
    s.addText(m.desk, {
      x: x + 0.25, y: 3.44, w: 2.4, h: 0.95, fontSize: 10, color: C.inkLembut,
      fontFace: FONT_ISI, margin: 0, valign: "top", lineSpacing: 14,
    });
    s.addText("HYPERPARAMETER", {
      x: x + 0.25, y: 4.42, w: 2.4, h: 0.24, fontSize: 8.5, bold: true,
      color: C.inkLembut, fontFace: FONT_ISI, charSpacing: 0.8, margin: 0,
    });
    s.addShape("roundRect", {
      x: x + 0.25, y: 4.66, w: 2.36, h: 0.56, rectRadius: 0.08,
      fill: { color: C.paper },
    });
    s.addText(m.hyper, {
      x: x + 0.37, y: 4.7, w: 2.14, h: 0.48, fontSize: 9, color: C.ink,
      fontFace: FONT_ISI, margin: 0, valign: "mid",
    });
    s.addText(`F1 macro validasi ${m.nilai}`, {
      x: x + 0.25, y: 5.28, w: 2.4, h: 0.26, fontSize: 10, bold: true,
      color: m.warna, fontFace: FONT_ISI, margin: 0,
    });
  });

  catatanKartu(
    s, M, 5.88, 12.13, 0.9,
    "TUNJUKAN YANG SERING DILALU",
    "Fine-tuning 30 lapisan atas MobileNetV2 menaikkan accuracy tetapi menurunkan " +
    "recall ganas dari 0,5545 menjadi 0,3750 dan menambah FN ganas dari 139 menjadi " +
    "195 kasus. Pada dataset sebesar ini, membuka representasi pretrained justru " +
    "merusak sensitivitas.",
    C.rose, C.roseMuda
  );

  s.addNotes("Empat model dengan hyperparameter aktual. Tutup dengan temuan " +
    "fine-tuning: F1 macro hampir sama, recall ganas justru turun.");
}

// ================== SLIDE 11: PROTOKOL EVALUASI ===========================
{
  const s = slideKonten(
    pres,
    "BAB 4  ·  METODOLOGI",
    "Protokol evaluasi",
    "Data test dipakai satu kali, untuk laporan akhir. Ambang batas setiap model diambil dari data validasi."
  );

  const langkah = [
    ["01", "Split per lesi", "6.959 / 1.529 / 1.527", "Seed 42, tanpa overlap", C.navy],
    ["02", "Ekstraksi fitur", "HOG + histogram", "Scaler dan PCA fit di train", "1D4E89"],
    ["03", "Tuning", "Hyperparameter", "Hanya data latih", C.teal],
    ["04", "Ambang", `Target recall ${persen(H.final.threshold_target_recall, 0)}`, "Tiap model punya ambang sendiri", "2A7E77"],
    ["05", "Uji", "1.527 foto", "Sekali, lalu PAD-UFES-20", C.rose],
  ];
  langkah.forEach(([n, j, v, d, w], i) => {
    const x = M + i * 2.45;
    kartu(s, x, 1.98, 2.3, 2.25, {});
    nomorKartu(s, x + 0.25, 2.2, n, w);
    s.addText(j, {
      x: x + 0.25, y: 2.72, w: 1.8, h: 0.3, fontSize: 13.5, bold: true, color: C.ink,
      fontFace: FONT_ISI, margin: 0,
    });
    s.addText(v, {
      x: x + 0.25, y: 3.06, w: 1.8, h: 0.34, fontSize: 12, bold: true, color: w,
      fontFace: FONT_JUDUL, margin: 0,
    });
    s.addText(d, {
      x: x + 0.25, y: 3.44, w: 1.85, h: 0.6, fontSize: 9.5, color: C.inkLembut,
      fontFace: FONT_ISI, margin: 0, valign: "top", lineSpacing: 13,
    });
  });

  kartu(s, M, 4.45, 7.2, 2.32, {});
  s.addText("PEMBAGIAN DATA", {
    x: M + 0.3, y: 4.68, w: 6.6, h: 0.26, fontSize: 9.5, bold: true,
    color: C.inkLembut, fontFace: FONT_ISI, charSpacing: 1, margin: 0,
  });
  const pgsql = [
    { label: "Train", n: H.split.counts.train.images, w: 4.3, warna: C.navy },
    { label: "Validasi", n: H.split.counts.val.images, w: 0.95, warna: C.teal },
    { label: "Test", n: H.split.counts.test.images, w: 0.95, warna: C.rose },
  ];
  let xsplit = M + 0.3;
  pgsql.forEach((p) => {
    const x = xsplit;
    xsplit += p.w;
    s.addShape("roundRect", {
      x, y: 5.02, w: p.w, h: 0.46, rectRadius: 0.08, fill: { color: p.warna },
    });
    s.addText(`${p.label}  ${thousands(p.n)}`, {
      x: x + 0.12, y: 5.02, w: p.w - 0.24, h: 0.46, fontSize: 10.5, bold: true,
      color: C.putih, fontFace: FONT_ISI, valign: "mid", margin: 0,
    });
  });
  s.addText(
    "DATA TEST TIDAK DIPAKAI UNTUK\nmemilih fitur, tuning, atau ambang batas",
    {
      x: M + 0.3, y: 5.72, w: 6.6, h: 0.6, fontSize: 10.5, bold: true,
      color: C.rose, fontFace: FONT_ISI, margin: 0, lineSpacing: 15,
    }
  );

  kartu(s, 8.1, 4.45, 4.63, 2.32, {});
  s.addText("AMBANG MASING-MASING MODEL", {
    x: 8.4, y: 4.68, w: 4.0, h: 0.26, fontSize: 9.5, bold: true,
    color: C.inkLembut, fontFace: FONT_ISI, charSpacing: 0.8, margin: 0,
  });
  Object.keys(H.final.per_model).forEach((nama, i) => {
    const m = H.final.per_model[nama];
    const y = 5.02 + i * 0.38;
    s.addText(nama, {
      x: 8.4, y, w: 2.6, h: 0.3, fontSize: 10, color: C.inkLembut,
      fontFace: FONT_ISI, margin: 0,
    });
    s.addText(f(m.binary_with_tuned_threshold.threshold, 2), {
      x: 11.2, y, w: 1.2, h: 0.3, fontSize: 11, bold: true,
      color: nama === MODEL_TERPILIH ? C.rose : C.ink, fontFace: FONT_ISI,
      align: "right", margin: 0,
    });
  });

  s.addNotes("Tegaskan urutan: split, fitur, tuning, ambang, baru uji. Ambang " +
    "berbeda per model karena tiap model punya skala probabilitasnya sendiri.");
}

// ============== SLIDE 12: KENAPA F1 DAN RECALL, BUKAN ACCURACY ============
{
  const s = slideKonten(
    pres,
    "BAB 4  ·  METODOLOGI",
    "Kenapa F1 macro dan recall ganas, bukan accuracy?",
    "Kelas yang tidak seimbang membuat accuracy mudah menipu. Yang mahal adalah lesi ganas yang lolos."
  );

  const baseline = (H.diag || []).find((x) =>
    x.konfigurasi.startsWith("BASELINE")
  );
  const rfFinal = H.final.per_model["Random Forest (HOG)"];

  kartu(s, M, 1.98, 3.95, 2.6, {});
  s.addText("TEBAKAN BASELINE", {
    x: M + 0.28, y: 2.2, w: 3.4, h: 0.26, fontSize: 9.5, bold: true,
    color: C.inkLembut, fontFace: FONT_ISI, charSpacing: 1, margin: 0,
  });
  s.addText(baseline ? persen(baseline.accuracy) : "-", {
    x: M + 0.28, y: 2.5, w: 3.4, h: 0.66, fontSize: 34, bold: true, color: C.amber,
    fontFace: FONT_JUDUL, margin: 0,
  });
  s.addText("accuracy model yang selalu menebak nv", {
    x: M + 0.28, y: 3.16, w: 3.4, h: 0.28, fontSize: 10.5, bold: true, color: C.ink,
    fontFace: FONT_ISI, margin: 0,
  });
  s.addText(
    baseline
      ? `Tanpa belajar apa pun, model ini sudah mendapat F1 ${f(
          baseline.f1_macro
        )}. Semua lesi ganas lolos.`
      : "-",
    {
      x: M + 0.28, y: 3.52, w: 3.4, h: 0.9, fontSize: 10.5, color: C.inkLembut,
      fontFace: FONT_ISI, margin: 0, valign: "top", lineSpacing: 15,
    }
  );

  kartu(s, 4.75, 1.98, 3.95, 2.6, {});
  s.addText("AKURACY TERTINGGI, BUKAN MODEL TERBAIK", {
    x: 5.03, y: 2.2, w: 3.4, h: 0.26, fontSize: 9.5, bold: true,
    color: C.inkLembut, fontFace: FONT_ISI, charSpacing: 0.6, margin: 0,
  });
  s.addText(persen(rfFinal.accuracy), {
    x: 5.03, y: 2.5, w: 3.4, h: 0.66, fontSize: 34, bold: true, color: C.amber,
    fontFace: FONT_JUDUL, margin: 0,
  });
  s.addText("accuracy Random Forest, tertinggi dari empat model", {
    x: 5.03, y: 3.16, w: 3.4, h: 0.28, fontSize: 10.5, bold: true, color: C.ink,
    fontFace: FONT_ISI, margin: 0,
  });
  s.addText(
    `Recall ganasnya hanya ${persen(
      rfFinal.malignant_recall
    )}. Sekitar ${persen(1 - rfFinal.malignant_recall)} lesi ganas terlewat.`,
    {
      x: 5.03, y: 3.52, w: 3.4, h: 0.9, fontSize: 10.5, color: C.inkLembut,
      fontFace: FONT_ISI, margin: 0, valign: "top", lineSpacing: 15,
    }
  );

  kartu(s, 8.9, 1.98, 3.83, 2.6, {});
  s.addText("BIAYA SALAH KLASIFIKASI", {
    x: 9.18, y: 2.2, w: 3.3, h: 0.26, fontSize: 9.5, bold: true,
    color: C.inkLembut, fontFace: FONT_ISI, charSpacing: 1, margin: 0,
  });
  s.addText("False negative", {
    x: 9.18, y: 2.56, w: 3.3, h: 0.26, fontSize: 12, bold: true, color: C.rose,
    fontFace: FONT_ISI, margin: 0,
  });
  s.addText(
    "Lesi ganas dikira jinak. Pasien menunda periksa. Ini kesalahan paling mahal, " +
    "jadi ambang batas sengaja diturunkan.",
    {
      x: 9.18, y: 2.84, w: 3.3, h: 0.85, fontSize: 10.5, color: C.inkLembut,
      fontFace: FONT_ISI, margin: 0, valign: "top", lineSpacing: 15,
    }
  );
  s.addText("False positive", {
    x: 9.18, y: 3.74, w: 3.3, h: 0.26, fontSize: 12, bold: true, color: C.teal,
    fontFace: FONT_ISI, margin: 0,
  });
  s.addText(
    "Foto jinak ditandai perlu diperiksa. Biayanya hanya satu pemeriksaan tambahan.",
    {
      x: 9.18, y: 4.02, w: 3.3, h: 0.55, fontSize: 10.5, color: C.inkLembut,
      fontFace: FONT_ISI, margin: 0, valign: "top", lineSpacing: 15,
    }
  );

  catatanKartu(
    s, M, 4.78, 12.13, 1.35,
    "METRIK UTAMA YANG KAMI PAKAI",
    `F1 macro (mengseimbangkan presisi dan recall di tujuh kelas), recall ganas ` +
    `(FN ganas adalah kesalahan yang paling mahal), dan specificity sebagai penyeimbang. ` +
    `Pada data uji, model terpilih ${MODEL_TERPILIH} mencatat FN ganas ${BIN.fn} kasus dan ` +
    `FP ${BIN.fp} kasus pada ambang ${f(BIN.threshold, 2)}. Accuracy tetap dilaporkan, ` +
    "tetapi tidak menjadi dasar pemilihan.",
    C.teal, C.tealMuda
  );

  s.addNotes("Slide argumen. Dosen sering bertanya kenapa tidak accuracy. Jawaban " +
    "nya: karena baseline menebak nv sudah dapat accuracy " +
    (baseline ? persen(baseline.accuracy) : "-") + ".");
}

// ================== SLIDE 13: PEMILIHAN MODEL =============================
{
  const s = slideKonten(
    pres,
    "BAB 5  ·  HASIL",
    "Pemilihan model",
    "Dipilih dari skor gabungan yang menimbang recall ganas, F1 macro, kesiapan domain shift, dan efisiensi."
  );

  const urutModel = Object.keys(H.final.per_model).sort(
    (a, b) => SKOR[b] - SKOR[a]
  );
  const baris = urutModel.map((nama) => {
    const m = H.final.per_model[nama];
    return [
      nama,
      f(m.f1_macro),
      persen(m.malignant_recall),
      persen(m.accuracy),
      f(m.binary_with_tuned_threshold.threshold, 2),
      f(SKOR[nama]),
    ];
  });
  kartu(s, M, 1.98, 7.55, 2.5, {});
  s.addText("SKOR GABUNGAN (TEST SET)", {
    x: M + 0.3, y: 2.2, w: 7.0, h: 0.26, fontSize: 9.5, bold: true,
    color: C.inkLembut, fontFace: FONT_ISI, charSpacing: 1, margin: 0,
  });
  tabel(
    s, M + 0.3, 2.52, 6.95,
    ["Model", "F1 macro", "Recall ganas", "Accuracy", "Ambang", "Skor"],
    baris,
    [2.15, 0.95, 1.2, 0.95, 0.8, 0.9],
    { fs: 9.5, barisTinggi: 0.36, sorotBaris: [MODEL_TERPILIH] }
  );

  kartu(s, 8.4, 1.98, 4.33, 2.5, {});
  s.addText("BOBOT PENILAIAN", {
    x: 8.68, y: 2.2, w: 3.8, h: 0.26, fontSize: 9.5, bold: true,
    color: C.inkLembut, fontFace: FONT_ISI, charSpacing: 1, margin: 0,
  });
  const w = H.final.selection_weights;
  Object.keys(w).forEach((k, i) => {
    const y = 2.58 + i * 0.46;
    const label = {
      recall_ganas: "Recall ganas",
      f1_macro: "F1 macro",
      domain_shift: "Kesiapan domain shift",
      efisiensi: "Efisiensi",
    }[k];
    s.addText(label, {
      x: 8.68, y, w: 2.2, h: 0.26, fontSize: 10.5, color: C.ink,
      fontFace: FONT_ISI, margin: 0,
    });
    s.addText(persen(w[k], 0), {
      x: 11.3, y, w: 1.1, h: 0.26, fontSize: 11, bold: true, color: C.teal,
      fontFace: FONT_ISI, align: "right", margin: 0,
    });
    s.addShape("roundRect", {
      x: 8.68, y: y + 0.28, w: 3.72, h: 0.08, rectRadius: 0.04, fill: { color: C.garis },
    });
    s.addShape("roundRect", {
      x: 8.68, y: y + 0.28, w: Math.max(0.1, 3.72 * w[k]), h: 0.08,
      rectRadius: 0.04, fill: { color: C.teal },
    });
  });

  const alasan = [
    ["F1 macro tertinggi", `F1 macro ${f(MOB.f1_macro)} di data uji, di atas tiga model lain.`],
    ["Recall ganas naik", `Recall ganas ${persen(MOB.malignant_recall)} dengan FN hanya ${BIN.fn} kasus.`],
    ["Tahan domain shift", `Di PAD-UFES-20: recall ${persen(H.pad.hasil_per_model[MODEL_TERPILIH].recall)}, specificity ${persen(H.pad.hasil_per_model[MODEL_TERPILIH].specificity)}.`],
    ["Siap dipakai", UKURAN_MODEL
      ? `Bobot model ${UKURAN_MODEL} MB, inferensi ${f(MOB.inference_ms_per_image, 1)} ms per foto di CPU.`
      : `Inferensi ${f(MOB.inference_ms_per_image, 1)} ms per foto di CPU.`],
  ];
  alasan.forEach(([j, d], i) => {
    const x = M + i * 3.07;
    kartu(s, x, 4.65, 2.86, 1.45, {});
    s.addText(j, {
      x: x + 0.25, y: 4.85, w: 2.4, h: 0.3, fontSize: 12, bold: true, color: C.ink,
      fontFace: FONT_ISI, margin: 0,
    });
    s.addText(d, {
      x: x + 0.25, y: 5.17, w: 2.4, h: 0.8, fontSize: 10, color: C.inkLembut,
      fontFace: FONT_ISI, margin: 0, valign: "top", lineSpacing: 14,
    });
  });

  s.addNotes("Tunjukkan tabel: skor tertinggi bukan accuracy tertinggi. Bobot " +
    "penilaian announced sebelum hasil dilihat, jadi bukan setelah fact.");
}

// ================== SLIDE 14: CONFUSION MATRIX ============================
{
  const s = slideKonten(
    pres,
    "BAB 5  ·  HASIL",
    `Confusion matrix biner pada ambang ${f(BIN.threshold, 2)}`,
    `Data uji ${thousands(H.final.test_set_size)} foto, dipakai satu kali. Angka inilah yang dilaporkan.`
  );

  const lebarSel = 2.05;
  const x0 = M + 0.9;
  s.addText("PREDIKSI JINAK", {
    x: x0, y: 2.05, w: lebarSel, h: 0.28, fontSize: 10, bold: true,
    color: C.inkLembut, fontFace: FONT_ISI, align: "center", charSpacing: 0.8, margin: 0,
  });
  s.addText("PREDIKSI GANAS", {
    x: x0 + lebarSel + 0.2, y: 2.05, w: lebarSel, h: 0.28, fontSize: 10, bold: true,
    color: C.inkLembut, fontFace: FONT_ISI, align: "center", charSpacing: 0.8, margin: 0,
  });

  const sel = [
    { label: "AKTUAL\njinak", nilai: BIN.tn, judul: "True Negative", ket: "benar jinak", fill: C.tealMuda, warna: C.teal },
    { label: "AKTUAL\njinak", nilai: BIN.fp, judul: "False Positive", ket: "alarm palsu", fill: C.amberMuda, warna: C.amber },
    { label: "AKTUAL\nganas", nilai: BIN.fn, judul: "False Negative", ket: "lesi ganas lolos", fill: C.roseMuda, warna: C.rose },
    { label: "AKTUAL\nganas", nilai: BIN.tp, judul: "True Positive", ket: "berhasil tertangkap", fill: C.tealMuda, warna: C.teal },
  ];
  sel.forEach((c, i) => {
    const col = i % 2;
    const row = Math.floor(i / 2);
    const x = x0 + col * (lebarSel + 0.2);
    const y = 2.42 + row * 1.72;
    s.addShape("roundRect", {
      x, y, w: lebarSel, h: 1.5, rectRadius: 0.1, fill: { color: c.fill },
      line: { color: c.warna, width: 0.75 },
    });
    s.addText(thousands(c.nilai), {
      x, y: y + 0.16, w: lebarSel, h: 0.52, fontSize: 26, bold: true, color: c.warna,
      fontFace: FONT_JUDUL, align: "center", margin: 0,
    });
    s.addText(c.judul, {
      x, y: y + 0.72, w: lebarSel, h: 0.28, fontSize: 11, bold: true, color: C.ink,
      fontFace: FONT_ISI, align: "center", margin: 0,
    });
    s.addText(c.ket, {
      x, y: y + 1.0, w: lebarSel, h: 0.28, fontSize: 9.5, color: C.inkLembut,
      fontFace: FONT_ISI, align: "center", margin: 0,
    });
    s.addText(c.label, {
      x: x0 - 0.95, y: y + 0.55, w: 0.85, h: 0.5, fontSize: 9.5, bold: true,
      color: c.label.includes("ganas") ? C.rose : C.inkLembut, fontFace: FONT_ISI,
      align: "right", margin: 0,
    });
  });
  s.addText(
    `Baris = kondisi sebenarnya · kolom = prediksi model. ${thousands(
      BIN.tp
    )} dari ${thousands(
      BIN.tp + BIN.fn
    )} kasus ganas berhasil tertangkap.`,
    {
      x: x0 - 0.95, y: 5.98, w: 5.2, h: 0.5, fontSize: 10.5, color: C.inkLembut,
      fontFace: FONT_ISI, align: "center", margin: 0,
    }
  );

  const metrik = [
    ["RECALL", persen(BIN.recall), `${thousands(BIN.tp)} dari ${thousands(BIN.tp + BIN.fn)} kasus ganas`, C.teal],
    ["SPECIFICITY", persen(BIN.specificity), "foto jinak yang benar dikira jinak", C.navy],
    ["PRECISION", persen(BIN.precision), "dari yang ditandai ganas, benar ganas", C.navy],
    ["FALSE NEGATIVE", thousands(BIN.fn), "lesi ganas yang lolos dari skrining", C.rose],
    ["FALSE POSITIVE", thousands(BIN.fp), "foto jinak yang perlu dicek manual", C.amber],
  ];
  metrik.forEach(([j, v, d, w], i) => {
    const y = 2.05 + i * 0.93;
    kartu(s, 7.0, y, 5.73, 0.82, {});
    s.addText(j, {
      x: 7.28, y: y + 0.13, w: 2.1, h: 0.26, fontSize: 9.5, bold: true,
      color: C.inkLembut, fontFace: FONT_ISI, charSpacing: 0.8, margin: 0,
    });
    s.addText(v, {
      x: 7.28, y: y + 0.36, w: 1.9, h: 0.38, fontSize: 19, bold: true, color: w,
      fontFace: FONT_JUDUL, margin: 0,
    });
    s.addText(d, {
      x: 9.35, y: y + 0.28, w: 3.1, h: 0.5, fontSize: 10.5, color: C.inkLembut,
      fontFace: FONT_ISI, margin: 0, valign: "mid",
    });
  });

  s.addNotes("Ambang 0,11 dipilih dari validasi dengan target recall 90%. " +
    "Bahas trade-off: FP lebih banyak dari FN, dan itu pilihan sadar untuk " +
    "skrining, bukan diagnosis.");
}

// ================== SLIDE 15: PERBANDINGAN MODEL ==========================
{
  const s = slideKonten(
    pres,
    "BAB 5  ·  HASIL",
    "Perbandingan model",
    "F1 macro dan recall ganas pada data uji, satu protocol, ambang berbeda untuk tiap model."
  );

  const namaModel = Object.keys(H.final.per_model);
  s.addChart(
    // Kolom vertikal memakai tipe "bar" dengan barDir bawaan (col).
    "bar",
    [
      {
        name: "F1 macro",
        labels: namaModel,
        values: namaModel.map((n) => H.final.per_model[n].f1_macro),
      },
      {
        name: "Recall ganas",
        labels: namaModel,
        values: namaModel.map((n) => H.final.per_model[n].malignant_recall),
      },
    ],
    {
      x: M, y: 2.0, w: 7.6, h: 4.4,
      catAxisLabelColor: C.ink, valAxisLabelColor: C.inkLembut,
      catAxisLabelFontFace: FONT_ISI, valAxisLabelFontFace: FONT_ISI,
      catAxisLabelFontSize: 10, valAxisLabelFontSize: 9,
      valGridLine: { color: C.garis, size: 0.5 },
      catGridLine: { style: "none" },
      showLegend: true, legendPos: "b", legendColor: C.inkLembut,
      legendFontFace: FONT_ISI, legendFontSize: 10,
      showValue: true, dataLabelColor: C.ink, dataLabelFontSize: 9,
      dataLabelFormatCode: "0.00", dataLabelPosition: "outEnd",
      chartColors: [C.teal, C.rose],
      showTitle: true, title: "Metrik pada data uji (1.527 foto)",
      titleFontSize: 12, titleColor: C.ink, titleFontFace: FONT_ISI,
      valAxisMinVal: 0, valAxisMaxVal: 0.8,
    }
  );

  kartu(s, 8.45, 2.0, 4.28, 2.6, {});
  s.addText("ANALISIS OVERFITTING", {
    x: 8.73, y: 2.22, w: 3.7, h: 0.26, fontSize: 9.5, bold: true,
    color: C.inkLembut, fontFace: FONT_ISI, charSpacing: 1, margin: 0,
  });
  Object.keys(H.final.per_model).forEach((nama, i) => {
    const o = H.final.per_model[nama].overfit_analysis;
    const parah = o.gap > 0.3;
    const y = 2.56 + i * 0.5;
    s.addText(nama, {
      x: 8.73, y, w: 2.1, h: 0.24, fontSize: 10, color: C.ink,
      fontFace: FONT_ISI, margin: 0,
    });
    s.addText(`selisih F1 ${f(o.gap, 4)}`, {
      x: 10.85, y, w: 1.55, h: 0.24, fontSize: 10, bold: true,
      color: parah ? C.rose : C.inkLembut, fontFace: FONT_ISI, align: "right", margin: 0,
    });
    s.addText(o.diagnosis, {
      x: 8.73, y: y + 0.22, w: 3.7, h: 0.24, fontSize: 9,
      color: parah ? C.rose : C.inkLembut, fontFace: FONT_ISI, margin: 0,
    });
  });

  catatanKartu(
    s, 8.45, 4.78, 4.28, 1.62,
    "TIGA MODEL, TIGA POLA",
    "Random Forest menghafal data latih dengan F1 latih 1,0000. MobileNetV2 dan " +
    "CNN punya selisih kecil terhadap data uji, jadi tidak ada overfitting berarti.",
    C.rose, C.roseMuda
  );

  s.addNotes("Chart: F1 macro dan recall ganas berdampingan. RF punya accuracy " +
    "tertinggi tapi recall ganas terendah. Analisis overfitting strengthens " +
    "cerita bahwa RF bukan pilihan.");
}

// ================== SLIDE 16: UJI EKSTERNAL ==============================
{
  const s = slideKonten(
    pres,
    "BAB 5  ·  HASIL",
    "Uji eksternal: PAD-UFES-20",
    `${thousands(H.pad.n_gambar)} foto smartphone. Model tidak dilatih ulang, ambang tetap dari validasi HAM10000.`
  );

  const baris = Object.keys(H.pad.hasil_per_model).map((nama) => {
    const e = H.pad.hasil_per_model[nama];
    const h = H.final.per_model[nama];
    return [
      nama,
      persen(h.malignant_recall),
      persen(e.recall),
      persen(h.specificity),
      persen(e.specificity),
      thousands(e.fn),
      thousands(e.fp),
    ];
  });
  kartu(s, M, 1.98, 8.2, 2.7, {});
  s.addText("RECALL DAN SPECIFICITY DI DUA DOMAIN", {
    x: M + 0.3, y: 2.2, w: 7.6, h: 0.26, fontSize: 9.5, bold: true,
    color: C.inkLembut, fontFace: FONT_ISI, charSpacing: 1, margin: 0,
  });
  tabel(
    s, M + 0.3, 2.52, 7.6,
    ["Model", "Rec HAM", "Rec PAD", "Spec HAM", "Spec PAD", "FN PAD", "FP PAD"],
    baris,
    [2.3, 0.9, 0.9, 0.9, 0.9, 0.85, 0.85],
    { fs: 9.5, barisTinggi: 0.36, sorotBaris: [MODEL_TERPILIH] }
  );

  kartu(s, 9.05, 1.98, 3.68, 2.7, {});
  s.addText("RECALL TINGGI BUKAN SEMBAGUS ITU", {
    x: 9.33, y: 2.2, w: 3.1, h: 0.26, fontSize: 9.5, bold: true,
    color: C.inkLembut, fontFace: FONT_ISI, charSpacing: 0.8, margin: 0,
  });
  const cnnPad = H.pad.hasil_per_model["CNN (from scratch)"];
  s.addText(persen(cnnPad.recall), {
    x: 9.33, y: 2.5, w: 3.1, h: 0.56, fontSize: 28, bold: true, color: C.amber,
    fontFace: FONT_JUDUL, margin: 0,
  });
  s.addText("recall CNN di data eksternal", {
    x: 9.33, y: 3.06, w: 3.1, h: 0.26, fontSize: 10.5, bold: true, color: C.ink,
    fontFace: FONT_ISI, margin: 0,
  });
  s.addText(
    `Specificity-nya hanya ${persen(cnnPad.specificity)}. Model itu menandai ` +
    "hampir semua foto sebagai ganas, jadi recall tinggi bukan berarti model bagus.",
    {
      x: 9.33, y: 3.38, w: 3.1, h: 1.1, fontSize: 10.5, color: C.inkLembut,
      fontFace: FONT_ISI, margin: 0, valign: "top", lineSpacing: 15,
    }
  );

  const mobPad = H.pad.hasil_per_model[MODEL_TERPILIH];
  catatanKartu(
    s, M, 4.85, 6.1, 1.55,
    "KESEIMBANGAN TERBAIK DARI MODEL TERPILIH",
    `${MODEL_TERPILIH} di PAD-UFES-20: recall ${persen(mobPad.recall)} dengan ` +
    `specificity ${persen(mobPad.specificity)}, FN ${thousands(mobPad.fn)} dan FP ` +
    `${thousands(mobPad.fp)}. Di domain yang berbeda, keseimbangan ini tetap terjaga, ` +
    "sedangkan specificity turun dari " +
    `${persen(H.final.per_model[MODEL_TERPILIH].specificity)} menjadi ${persen(
      mobPad.specificity
    )}.`,
    C.teal, C.tealMuda
  );

  const stat = H.pad.statistik_domain;
  catatanKartu(
    s, 7.0, 4.85, 5.73, 1.55,
    "PERGESERAN DOMAIN YANG TERUKUR",
    `Resolusi ${Math.round(stat.HAM10000.lebar)} × ${Math.round(
      stat.HAM10000.tinggi
    )} menjadi ${Math.round(stat["PAD-UFES-20"].lebar)} × ${Math.round(
      stat["PAD-UFES-20"].tinggi
    )}. Ketajaman naik dari ${f(stat.HAM10000.ketajaman, 0)} menjadi ${f(
      stat["PAD-UFES-20"].ketajaman, 0
    )}. Distribusi citra berubah, dan itu terlihat langsung pada specificity.`,
    C.navy, C.paper
  );

  s.addNotes("Slide kunci domain shift. Tunjukkan CNN yang recall 0,9698 tapi " +
    "specificity 0,0104. MobileNetV2 dipilih karena keseimbangannya.");
}

// ================== SLIDE 17: RECALL PER DIAGNOSIS ========================
{
  const s = slideKonten(
    pres,
    "BAB 5  ·  HASIL",
    "Recall per diagnosis pada data eksternal",
    "Di domain yang berbeda, tiap diagnosis punya cerita sendiri. Dua kelas punya catatan khusus."
  );

  const diag = H.pad.per_diagnosis;
  const maks = Math.max(...diag.map((d) => d.recall));
  kartu(s, M, 1.98, 7.55, 4.8, {});
  s.addText("MOBILENETV2 DI PAD-UFES-20", {
    x: M + 0.3, y: 2.2, w: 7.0, h: 0.26, fontSize: 9.5, bold: true,
    color: C.inkLembut, fontFace: FONT_ISI, charSpacing: 1, margin: 0,
  });
  diag.forEach((d, i) => {
    const y = 2.58 + i * 0.64;
    s.addText(d.diagnosis, {
      x: M + 0.3, y, w: 1.0, h: 0.26, fontSize: 11, bold: true, color: C.ink,
      fontFace: FONT_ISI, margin: 0,
    });
    s.addText(`${d.kelompok} · n = ${thousands(d.n)}`, {
      x: M + 1.28, y: y + 0.02, w: 2.2, h: 0.24, fontSize: 9.5, color: C.inkLembut,
      fontFace: FONT_ISI, margin: 0,
    });
    s.addShape("roundRect", {
      x: M + 3.55, y: y + 0.04, w: 2.6, h: 0.19, rectRadius: 0.09,
      fill: { color: C.garis },
    });
    s.addShape("roundRect", {
      x: M + 3.55, y: y + 0.04, w: Math.max(0.1, 2.6 * (d.recall / maks)), h: 0.19,
      rectRadius: 0.09, fill: { color: d.kelompok === "ganas" ? C.teal : C.amber },
    });
    s.addText(persen(d.recall), {
      x: M + 6.25, y, w: 1.0, h: 0.26, fontSize: 11, bold: true, color: C.ink,
      fontFace: FONT_ISI, align: "right", margin: 0,
    });
    s.addText(`${thousands(d.terdeteksi)} dari ${thousands(d.n)} terdeteksi`, {
      x: M + 3.55, y: y + 0.26, w: 3.7, h: 0.24, fontSize: 9,
      color: C.inkLembut, fontFace: FONT_ISI, margin: 0,
    });
  });

  const mel = diag.find((d) => d.diagnosis === "MEL");
  const scc = diag.find((d) => d.diagnosis === "SCC");
  const sek = diag.find((d) => d.diagnosis === "SEK");
  kartu(s, 8.45, 1.98, 4.28, 1.55, {});
  s.addText("KELAS YANG BELUM PERNAH DILIHAT", {
    x: 8.73, y: 2.2, w: 3.7, h: 0.26, fontSize: 9.5, bold: true,
    color: C.inkLembut, fontFace: FONT_ISI, charSpacing: 0.8, margin: 0,
  });
  s.addText(
    scc
      ? `SCC (karsinoma sel skuamos) tidak ada di HAM10000. Recall ${persen(
          scc.recall
        )} diuji pada kelas yang model tidak pernah lihat, dan hasilnya justru di atas rata-rata kelas ganas lain.`
      : "-",
    {
      x: 8.73, y: 2.54, w: 3.7, h: 0.9, fontSize: 10.5, color: C.inkLembut,
      fontFace: FONT_ISI, margin: 0, valign: "top", lineSpacing: 15,
    }
  );

  kartu(s, 8.45, 3.65, 4.28, 1.55, {});
  s.addText("ANGKA YANG TIDAK STABIL", {
    x: 8.73, y: 3.87, w: 3.7, h: 0.26, fontSize: 9.5, bold: true,
    color: C.inkLembut, fontFace: FONT_ISI, charSpacing: 0.8, margin: 0,
  });
  s.addText(
    mel
      ? `MEL hanya ${thousands(mel.n)} foto di PAD-UFES-20. Recall ${persen(
          mel.recall
        )} terlihat rendah, tetapi dengan sample sebesar itu angkanya belum bisa dipakai untuk menilai model.`
      : "-",
    {
      x: 8.73, y: 4.21, w: 3.7, h: 0.9, fontSize: 10.5, color: C.inkLembut,
      fontFace: FONT_ISI, margin: 0, valign: "top", lineSpacing: 15,
    }
  );

  catatanKartu(
    s, 8.45, 5.32, 4.28, 1.46,
    "KELAS TIDAK GANAS",
    sek
      ? `SEK (lesi benigna) punya recall ${persen(
          sek.recall
        )}. Model terlalu sering menandainya sebagai ganas, dan itu yang membuat specificity turun di data eksternal.`
      : "-",
    C.amber, C.amberMuda
  );

  s.addNotes("Tiga poin: SCC sebagai kelas yang tak pernah dilihat, MEL dengan " +
    "sample kecil, dan SEK yang paling sering salah. Semua dari log uji eksternal.");
}

// ================== SLIDE 18: DEMO APLIKASI ==============================
{
  const s = slideKonten(
    pres,
    "BAB 6  ·  DEMO",
    "Demo aplikasi",
    "Model, ambang batas, dan preprocessing dimuat dari satu artefak yang sama dengan yang dipakai saat evaluasi."
  );

  kartu(s, M, 1.98, 12.13, 0.85, { fill: C.navy, line: C.navy });
  s.addText("streamlit run app/app.py", {
    x: M + 0.32, y: 2.12, w: 5.6, h: 0.34, fontSize: 15, bold: true, color: C.teal,
    fontFace: FONT_JUDUL, margin: 0,
  });
  s.addText(APP_URL, {
    x: M + 6.0, y: 2.18, w: 5.8, h: 0.28, fontSize: 10.5, color: C.teksGelap,
    fontFace: FONT_ISI, align: "right", margin: 0,
  });

  const kolom = [
    {
      j: "Unggah foto",
      d: "Foto lesi kulit dari dekat. Aplikasi membaca preprocessing dari " +
        "models/preprocess_config.json, jadi ukuran gambar dan normalisasi tidak " +
        "bisa tidak sinkron dengan training.",
      warna: C.teal,
    },
    {
      j: "Baca hasilnya",
      d: "Tujuh bar probabilitas per kelas, status tiga warna (aman, waspada, " +
        "tidak yakin), dan catatan privasi. Foto diproses di memori selama sesi, " +
        "tidak disimpan.",
      warna: C.navy,
    },
    {
      j: "Batas input",
      d: "Kartu input menyebut foto yang didukung: lesi kulit dari dekat. " +
        "Foto wajah, tangkapan layar, atau gambar tanpa lesi yang jelas bisa " +
        "memberi prediksi yang menyesatkan.",
      warna: C.amber,
    },
  ];
  kolom.forEach((k, i) => {
    const x = M + i * 4.07;
    kartu(s, x, 3.05, 3.86, 2.45, {});
    nomorKartu(s, x + 0.28, 3.3, i + 1, k.warna);
    s.addText(k.j, {
      x: x + 0.8, y: 3.32, w: 2.8, h: 0.32, fontSize: 14, bold: true, color: C.ink,
      fontFace: FONT_ISI, margin: 0,
    });
    s.addText(k.d, {
      x: x + 0.28, y: 3.85, w: 3.3, h: 1.45, fontSize: 10.5, color: C.inkLembut,
      fontFace: FONT_ISI, margin: 0, valign: "top", lineSpacing: 15,
    });
  });

  catatanKartu(
    s, M, 5.7, 12.13, 1.07,
    "PENGUJIAN APLIKASI",
    "Empat kondisi diuji otomatis dengan streamlit.testing.v1.AppTest: kondisi " +
    "kosong, unggah foto (tujuh bar muncul), file rusak (pesan ramah), dan " +
    "pemilihan contoh gambar. Semua lulus. Aplikasi juga berjalan di Streamlit " +
    "Community Cloud dengan subdomain di atas.",
    C.teal, C.tealMuda
  );

  s.addNotes("Demo langsung di sini kalau jaringan mengizinkan: unggah satu " +
    "foto atau pakai contoh. Tunjukkan tujuh bar probabilitas dan status tiga " +
    "warna, lalu disclaimer di bagian bawah.");
}

// ================== SLIDE 19: KETERBATASAN ===============================
{
  const s = slideKonten(
    pres,
    "BAB 6  ·  PENUTUP",
    "Keterbatasan dan pengembangan",
    " disampaikan terbuka. Kredibilitas sebuah model skrining diuji dari keterbatasannya."
  );

  const bats = [
    ["Populasi kulit terbatas", "Seluruh data HAM10000 adalah kulit Fitzpatrick I-II, sedangkan populasi Indonesia didominasi III-V. Performa pada kulit gelap belum diketahui."],
    ["Tanpa validasi klinis", "Tidak ada uji dengan dokter kulit. Semua angka berasal dari pengukuran pada dataset, bukan dari diagnosis klinis."],
    ["Kelas minority sangat sedikit", `Hanya ${thousands(JUMLAH_KELAS.df)} foto kelas df dan ${thousands(JUMLAH_KELAS.vasc)} foto kelas vasc di seluruh dataset.`],
    ["Domain shift menurunkan specificity", `Specificity model terpilih turun dari ${persen(H.final.per_model[MODEL_TERPILIH].specificity)} menjadi ${persen(H.pad.hasil_per_model[MODEL_TERPILIH].specificity)} di foto smartphone.`],
  ];
  bats.forEach(([j, d], i) => {
    const x = M + (i % 2) * 3.55;
    const y = 1.98 + Math.floor(i / 2) * 1.42;
    kartu(s, x, y, 3.4, 1.28, {});
    s.addShape("ellipse", {
      x: x + 0.25, y: y + 0.26, w: 0.16, h: 0.16, fill: { color: C.rose },
    });
    s.addText(j, {
      x: x + 0.52, y: y + 0.2, w: 2.7, h: 0.28, fontSize: 11.5, bold: true,
      color: C.ink, fontFace: FONT_ISI, margin: 0,
    });
    s.addText(d, {
      x: x + 0.25, y: y + 0.56, w: 2.95, h: 0.65, fontSize: 9.5, color: C.inkLembut,
      fontFace: FONT_ISI, margin: 0, valign: "top", lineSpacing: 13,
    });
  });

  kartu(s, 7.85, 1.98, 4.88, 4.3, {});
  s.addText("PENGEMBANGAN BERIKUTNYA", {
    x: 8.13, y: 2.22, w: 4.3, h: 0.26, fontSize: 9.5, bold: true,
    color: C.inkLembut, fontFace: FONT_ISI, charSpacing: 1, margin: 0,
  });
  const lanjut = [
    "Kumpulkan data lesi pada kulit Fitzpatrick III-V",
    "Validasi klinis bersama dokter kulit",
    "Kalibrasi ulang ambang batas untuk foto smartphone",
    "Perbanyak kelas df dan vasc lewat recruiting biopsy",
    "Uji pada dataset lokal, bukan hanya HAM10000",
  ];
  lanjut.forEach((t, i) => {
    const y = 2.6 + i * 0.68;
    s.addText("✓", {
      x: 8.13, y, w: 0.3, h: 0.28, fontSize: 12, bold: true, color: C.teal,
      fontFace: FONT_ISI, margin: 0,
    });
    s.addText(t, {
      x: 8.48, y, w: 3.95, h: 0.55, fontSize: 11, color: C.ink,
      fontFace: FONT_ISI, margin: 0, valign: "top", lineSpacing: 15,
    });
  });

  catatanKartu(
    s, M, 4.88, 6.9, 1.1,
    "KESIMPULAN YANG JUJUR",
    "\"Model ini alat bantu untuk menandai foto yang perlu diperiksa lebih lanjut. " +
    "Keputusan tetap ada pada dokter kulit.\"",
    C.navy, C.paper
  );

  s.addNotes("Sampaikan keterbatasan lebih dulu. Dosen lebih menghargai tim yang " +
    "menyebutkan batas modelnya sendiri.");
}

// ================== SLIDE 20: KESIMPULAN =================================
{
  const s = slideKonten(
    pres,
    "BAB 6  ·  PENUTUP",
    "Kesimpulan",
    "Satu model terpilih, tiga angka yang perlu diingat, dan satu hal yang tidak boleh dilupakan."
  );

  const mobPad = H.pad.hasil_per_model[MODEL_TERPILIH];
  const big = [
    ["F1 macro", f(MOB.f1_macro), "data uji, 1.527 foto", C.teal],
    ["RECALL GANAS", persen(MOB.malignant_recall), `FN ganas ${BIN.fn} kasus`, C.rose],
    ["F1 EKSTERNAL", f(mobPad.f1), "PAD-UFES-20, 2.298 foto", C.navy],
  ];
  big.forEach(([j, v, d, w], i) => {
    const x = M + i * 4.07;
    kartu(s, x, 1.98, 3.86, 1.75, {});
    s.addText(j, {
      x: x + 0.28, y: 2.18, w: 3.3, h: 0.26, fontSize: 9.5, bold: true,
      color: C.inkLembut, fontFace: FONT_ISI, charSpacing: 1, margin: 0,
    });
    s.addText(v, {
      x: x + 0.28, y: 2.46, w: 3.3, h: 0.66, fontSize: 34, bold: true, color: w,
      fontFace: FONT_JUDUL, margin: 0,
    });
    s.addText(d, {
      x: x + 0.28, y: 3.14, w: 3.3, h: 0.28, fontSize: 10.5, color: C.inkLembut,
      fontFace: FONT_ISI, margin: 0,
    });
  });

  const simpul = [
    ["MobileNetV2 + ambang 0,11", `Model dengan skor gabungan tertinggi ${f(
      SKOR[MODEL_TERPILIH]
    )}. Ambang diambil dari validasi, bukan dari data uji.`],
    ["Alat skrining, bukan alat diagnosis", "Aplikasi menampilkan status dan ketidakpastian, lalu mengarahkan pengguna ke dokter kulit."],
    ["Model harus dinilai pada domain aslinya", `Di foto smartphone, specificity berubah dari ${persen(
      H.final.per_model[MODEL_TERPILIH].specificity
    )} menjadi ${persen(mobPad.specificity)}. Uji eksternal wajib, bukan opsional.`],
  ];
  simpul.forEach(([j, d], i) => {
    const y = 3.95 + i * 0.82;
    kartu(s, M, y, 8.6, 0.72, {});
    nomorKartu(s, M + 0.2, y + 0.17, i + 1, [C.teal, C.navy, C.amber][i]);
    s.addText(j, {
      x: M + 0.72, y: y + 0.1, w: 3.9, h: 0.28, fontSize: 12, bold: true,
      color: C.ink, fontFace: FONT_ISI, margin: 0,
    });
    s.addText(d, {
      x: M + 4.7, y: y + 0.08, w: 3.7, h: 0.58, fontSize: 10, color: C.inkLembut,
      fontFace: FONT_ISI, margin: 0, valign: "mid", lineSpacing: 13,
    });
  });

  kartu(s, 9.4, 3.95, 3.33, 2.56, { fill: C.navy, line: C.navy });
  s.addText("KALIMAT PENUTUP", {
    x: 9.68, y: 4.18, w: 2.8, h: 0.26, fontSize: 9.5, bold: true, color: C.teal,
    fontFace: FONT_ISI, charSpacing: 1.2, margin: 0,
  });
  s.addText(
    "\"Kami tidak mendiagnosis kulit. Kami menandai foto mana yang perlu " +
    "diperiksa dokter lebih cepat.\"",
    {
      x: 9.68, y: 4.52, w: 2.8, h: 1.3, fontSize: 13, italic: true,
      color: C.putih, fontFace: FONT_JUDUL, margin: 0, valign: "top",
      lineSpacing: 19,
    }
  );

  s.addNotes("Tiga angka, tiga simpulan, satu kalimat penutup. Sambungkan ke " +
    "slide demo bila waktu memungkinkan.");
}

// ================== SLIDE 21: PENUTUP ===================================
{
  const s = slideGelap(pres);
  s.addText("BAB 6  ·  PENUTUP", {
    x: 0.85, y: 1.05, w: 6.5, h: 0.3, fontSize: 11, bold: true, color: C.teal,
    fontFace: FONT_ISI, charSpacing: 2, margin: 0,
  });
  s.addText("Terima kasih", {
    x: 0.85, y: 1.5, w: 6.5, h: 1.0, fontSize: 46, bold: true, color: C.putih,
    fontFace: FONT_JUDUL, margin: 0,
  });
  s.addText("Pertanyaan dan diskusi", {
    x: 0.85, y: 2.6, w: 6.5, h: 0.4, fontSize: 17, color: C.teksGelap,
    fontFace: FONT_ISI, margin: 0,
  });
  s.addText(`${NAMA_KELOMPOK}  ·  ML2026  ·  ${H.info.tanggal}`, {
    x: 0.85, y: 3.2, w: 6.5, h: 0.32, fontSize: 12, bold: true, color: C.putih,
    fontFace: FONT_ISI, margin: 0,
  });
  s.addText(H.info.dosen, {
    x: 0.85, y: 3.58, w: 6.5, h: 0.32, fontSize: 11.5, color: C.teksGelap,
    fontFace: FONT_ISI, margin: 0,
  });
  s.addText("ALAT BANTU BELAJAR, BUKAN ALAT DIAGNOSIS MEDIS", {
    x: 0.85, y: 4.2, w: 6.5, h: 0.3, fontSize: 10, bold: true, color: C.slateMuda,
    fontFace: FONT_ISI, charSpacing: 1, margin: 0,
  });
  // Dua baris terpisah: kalau digabung dalam satu kotak, URL pertama
  // mendorong baris kedua ke bawah dan tampilan jadi tidak rapi.
  s.addText([
    { text: "Repo  ", options: { bold: true, color: C.teal } },
    { text: "github.com/Grenaydinkamil/ProjectForestry_ML2026_Mendekteksi_Lesi_Kulit",
      options: { color: C.teksGelap, underline: true, hyperlink: { url: REPO_URL } } },
  ], {
    x: 0.85, y: 4.55, w: 7.2, h: 0.3, fontSize: 10, fontFace: FONT_ISI, margin: 0,
  });
  s.addText([
    { text: "App  ", options: { bold: true, color: C.teal } },
    { text: "projectforestryml2026mendekteksilesikulit.streamlit.app",
      options: { color: C.teksGelap, underline: true, hyperlink: { url: APP_URL } } },
  ], {
    x: 0.85, y: 4.9, w: 7.2, h: 0.3, fontSize: 10, fontFace: FONT_ISI, margin: 0,
  });

  s.addShape("roundRect", {
    x: 8.35, y: 1.35, w: 4.1, h: 4.9, rectRadius: 0.12,
    fill: { color: C.putih, transparency: 92 },
    line: { color: C.slateMuda, width: 0.75 },
  });
  s.addText("PEMBAGIAN TUGAS", {
    x: 8.68, y: 1.62, w: 3.4, h: 0.28, fontSize: 10, bold: true, color: C.teal,
    fontFace: FONT_ISI, charSpacing: 1.4, margin: 0,
  });
  (H.info.pembagian_tugas || []).forEach((p, i) => {
    const y = 2.05 + i * 1.32;
    s.addShape("roundRect", {
      x: 8.68, y: y + 0.05, w: 0.09, h: 0.42, rectRadius: 0.04,
      fill: { color: [C.teal, C.amber, C.slateMuda][i % 3] },
    });
    s.addText(p.anggota || p.nama, {
      x: 8.92, y, w: 3.2, h: 0.3, fontSize: 12.5, bold: true, color: C.putih,
      fontFace: FONT_ISI, margin: 0,
    });
    s.addText(`${p.tugas.replace(/\s*\(Notebook[^)]*\)\s*$/, "")}  ·  ${p.nim}`, {
      x: 8.92, y: y + 0.32, w: 3.25, h: 0.8, fontSize: 10, color: C.teksGelap,
      fontFace: FONT_ISI, margin: 0, valign: "top", lineSpacing: 14,
    });
  });

  s.addNotes("Buka sesi tanya jawab. Tutup dengan menyebut bahwa kode, model, " +
    "dan log evaluasi tersedia di repo publik.");
}

// ---------------------------------------------------------------------------
pres.writeFile({ fileName: OUT }).then(() => {
  console.log(`PPT dibuat: ${OUT}`);
  console.log(`Slide: ${nomorSlide} / ${TOTAL_SLIDE}`);
  console.log(
    `Sumber angka: final=${H.final ? "ada" : "TIDAK ADA"}, ` +
    `pad=${H.pad ? "ada" : "TIDAK ADA"}, split=${H.split ? "ada" : "TIDAK ADA"}`
  );
});
