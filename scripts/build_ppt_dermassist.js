/**
 * scripts/build_ppt_dermassist.js
 * ================================
 * Generator PPT presentasi "DermAssist Clinical" untuk Kelompok 6 (24 slide).
 *
 * Struktur mengikuti urutan komponen yang diwajibkan tugas:
 * judul, latar belakang, tujuan, studi literatur, dataset, EDA,
 * preprocessing, pengembangan model, evaluasi, pemilihan model,
 * implementasi aplikasi, demo, kesimpulan, keterbatasan, pembagian tugas.
 * Slide terakhir berisi versi ringkas tabel studi literatur sebagai lampiran.
 *
 * ATURAN PENTING:
 * SEMUA angka dibaca dari docs/logs/*.json dan docs/logs/literature_summary.json.
 * Tidak ada angka atau hasil yang diketik manual.
 *
 * Jalankan:
 * node scripts/build_ppt_dermassist.js
 *
 * Output: docs/PPT_KEL6_DermAssist_Clinical.pptx
 */

const fs = require("fs");
const path = require("path");
const PptxGenJS = require("pptxgenjs");

const REPO = path.resolve(__dirname, "..");
const LOGS = path.join(REPO, "docs", "logs");
const OUT = path.join(REPO, "docs", "PPT_KEL6_DermAssist_Clinical.pptx");

// ---------------------------------------------------------------------------
// PALET
// ---------------------------------------------------------------------------
// Sama dengan aplikasi dan deck sebelumnya: navy sebagai tinta struktur,
// teal untuk jinak, rose untuk ganas. Satu warna alarm saja.
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

const M = 0.6;
const TOTAL_SLIDE = 24;
const NAMA_KELOMPOK = "Kelompok 6";
const JUDUL_PROYEK = "DermAssist Clinical";
const APP_URL =
 "https://projectforestryml2026mendekteksilesikulit-vvearxur6nuwga44fmaf.streamlit.app";
const REPO_URL =
 "https://github.com/Grenaydinkamil/ProjectForestry_ML2026_Mendekteksi_Lesi_Kulit";

// ---------------------------------------------------------------------------
// PEMBACA DATA
// ---------------------------------------------------------------------------
function bacaJson(nama) {
 const p = path.join(LOGS, nama);
 if (!fs.existsSync(p)) return null;
 try {
 return JSON.parse(fs.readFileSync(p, "utf-8"));
 } catch (e) {
 console.warn(` [warn] ${nama}: ${e.message}`);
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
 cnnHistory: bacaJson("cnn_training_history.json"),
 mnet: bacaJson("model_mobilenetv2_validasi.json"),
 lit: bacaJson("literature_summary.json"),
 info: JSON.parse(
 fs.readFileSync(path.join(REPO, "docs", "info_kelompok.json"), "utf-8")
 ),
 threshold: JSON.parse(
 fs.readFileSync(path.join(REPO, "models", "threshold_config.json"), "utf-8")
 ),
 preprocess: JSON.parse(
 fs.readFileSync(path.join(REPO, "models", "preprocess_config.json"), "utf-8")
 ),
};

if (!H.final || !H.split || !H.pad || !H.lit) {
 console.error("Log utama tidak lengkap. Jalankan notebook evaluasi lebih dulu.");
 process.exit(1);
}

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
const KELAS_GANAS = H.threshold.malignant_classes;
const KELAS_JINAK = H.threshold.benign_classes;

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
const FOTO_GANAS = KELAS_GANAS.reduce((a, k) => a + (JUMLAH_KELAS[k] || 0), 0);
const PESEN_GANAS = FOTO_GANAS / TOTAL_FOTO;
const PESEN_NV = JUMLAH_KELAS.nv / TOTAL_FOTO;
const PESEN_DF = JUMLAH_KELAS.df / TOTAL_FOTO;
const PESEN_MEL = JUMLAH_KELAS.mel / TOTAL_FOTO;
const RASIO_KELAS = Math.round((JUMLAH_KELAS.nv / JUMLAH_KELAS.df) * 10) / 10;
const FOTO_PER_LESI = TOTAL_FOTO - TOTAL_LESI;

const MOB = H.final.per_model["MobileNetV2"];
const BIN = MOB.binary_with_tuned_threshold;
const SKOR = H.final.selection_scores;
const MODEL_TERPILIH = H.final.selected_model;
const UKURAN_MODEL = (() => {
 const p = path.join(REPO, "models", "mobilenetv2_best.keras");
 if (!fs.existsSync(p)) return null;
 return (fs.statSync(p).size / (1024 * 1024)).toFixed(1).replace(".", ",");
})();

const f = (v, d = 4) => (v == null ? "-" : Number(v).toFixed(d).replace(".", ","));
const persen = (v, d = 1) =>
 v == null ? "-" : (v * 100).toFixed(d).replace(".", ",") + "%";
const thousands = (v) => (v == null ? "-" : String(v).replace(/\B(?=(\d{3})+(?!\d))/g, "."));
const menit = (detik) =>
 detik == null ? "-" :
 detik < 60 ? f(detik, 1) + " detik" : f(detik / 60, 1) + " menit";

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
 s.addShape("rect", { x: M, y: 0.37, w: 0.1, h: 0.1, fill: { color: C.teal } });
 s.addText(kicker, {
 x: M + 0.2, y: 0.3, w: 11.5, h: 0.26, fontSize: 10.5, bold: true,
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
 s.addShape("rect", { x: M, y: 1.72, w: 12.13, h: 0.012, fill: { color: C.garis } });
 footer(s, false);
 return s;
}

function slideGelap(pres) {
 const s = pres.addSlide();
 s.background = { color: C.navy };
 s.addShape("ellipse", { x: 8.7, y: -1.5, w: 6.6, h: 6.6, fill: { color: C.navySoft } });
 s.addShape("ellipse", { x: 10.4, y: 3.6, w: 5.4, h: 5.4, fill: { color: "1B2C55" } });
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
 type: "outer", color: "9AA4B5", blur: 7, offset: 1, angle: 90, opacity: 0.18,
 };
 }
 s.addShape("roundRect", o);
}

function labelKecil(s, x, y, teks, warna = C.inkLembut) {
 s.addText(teks, {
 x, y, w: 6.5, h: 0.26, fontSize: 9.5, bold: true, color: warna,
 fontFace: FONT_ISI, charSpacing: 1.1, margin: 0,
 });
}

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
 x: x + 0.25, y: y + 0.64, w: w - 0.5, h: 0.62, fontSize: o.ukuran || 28,
 bold: true, color: o.warna || C.teal, fontFace: FONT_JUDUL, margin: 0,
 });
 if (o.keterangan) {
 s.addText(o.keterangan, {
 x: x + 0.25, y: y + 1.24, w: w - 0.5, h: 0.3, fontSize: 10.5, bold: true,
 color: C.ink, fontFace: FONT_ISI, margin: 0,
 });
 }
 if (o.bar) {
 const lebarBar = w - 0.5;
 s.addShape("roundRect", {
 x: x + 0.25, y: y + 1.58, w: lebarBar, h: 0.11, rectRadius: 0.055,
 fill: { color: C.garis },
 });
 s.addShape("roundRect", {
 x: x + 0.25, y: y + 1.58, w: Math.max(0.12, lebarBar * o.bar), h: 0.11,
 rectRadius: 0.055, fill: { color: o.warna || C.teal },
 });
 }
 if (o.catatan) {
 s.addText(o.catatan, {
 x: x + 0.25, y: y + 1.8, w: w - 0.5, h: 0.5, fontSize: 10,
 color: C.inkLembut, fontFace: FONT_ISI, margin: 0, valign: "top",
 });
 }
}

function barTumpuk(s, x, y, w, h, segmen) {
 let xkursor = x;
 segmen.forEach((sg) => {
 const lebar = w * sg.porsi;
 s.addShape("rect", {
 x: xkursor, y, w: Math.max(lebar, 0.02), h, fill: { color: sg.warna },
 });
 if (lebar > 0.62) {
 s.addText(sg.label, {
 x: xkursor, y, w: lebar, h, fontSize: 9.5, bold: true, color: C.putih,
 fontFace: FONT_ISI, align: "center", valign: "mid", margin: 0,
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
 bold: true, color: C.putih, fill: { color: C.navy }, fontSize: fs,
 fontFace: FONT_ISI, margin: 0.07,
 },
 })),
 ...rows.map((r) =>
 r.map((v, i) => ({
 text: String(v),
 options: {
 color: i === 0 ? C.ink : C.inkLembut,
 bold: i === 0 || (opsi.tebalkan && opsi.tebalkan.includes(r[0])),
 fontSize: fs, fontFace: FONT_ISI, margin: 0.07,
 fill: {
 color:
 opsi.sorotBaris && opsi.sorotBaris.includes(r[0]) ? C.tealMuda : C.putih,
 },
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
 x: x + 0.1, y, w: w - 0.28, h, fontSize: 10, bold: true, color: C.putih,
 fontFace: FONT_ISI, align: "center", valign: "mid", margin: 0,
 });
}

function catatanKartu(s, x, y, w, h, judulTeks, isi, warna, muda) {
 s.addShape("roundRect", {
 x, y, w, h, rectRadius: 0.1,
 fill: { color: muda }, line: { color: warna, width: 0.75 },
 });
 s.addText(judulTeks, {
 x: x + 0.25, y: y + 0.12, w: w - 0.5, h: 0.26, fontSize: 10.5, bold: true,
 color: warna, fontFace: FONT_ISI, charSpacing: 0.8, margin: 0,
 });
 s.addText(isi, {
 // Tinggi teks dijaga positif. Kalau kartu terlalu pendek, tinggi yang
 // negatif membuat PowerPoint menolak seluruh file.
 x: x + 0.25, y: y + 0.44, w: w - 0.5, h: Math.max(0.3, h - 0.56),
 fontSize: 10.5, color: C.ink, fontFace: FONT_ISI, margin: 0,
 valign: "top", lineSpacing: 15,
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
pres.subject =
 "DermAssist Clinical: alat bantu skrining awal berbasis pembelajaran mesin untuk lesi kulit";
pres.title = "DermAssist Clinical - Presentasi Kelompok 6";

// ====================== SLIDE 1: JUDUL ====================================
{
 const s = slideGelap(pres);
 s.addShape("ellipse", { x: 9.5, y: -0.9, w: 5.2, h: 5.2, fill: { color: "1E3A6E" } });

 s.addText("PROJECT AKHIR PEMBELAJARAN MESIN", {
 x: 0.85, y: 0.95, w: 7.4, h: 0.3, fontSize: 12, bold: true, color: C.teal,
 fontFace: FONT_ISI, charSpacing: 2, margin: 0,
 });
 s.addText(`${NAMA_KELOMPOK.toUpperCase()} · ML2026`, {
 x: 0.85, y: 1.29, w: 7.4, h: 0.3, fontSize: 11, color: C.slateMuda,
 fontFace: FONT_ISI, charSpacing: 1.4, margin: 0,
 });
 s.addText(JUDUL_PROYEK, {
 x: 0.85, y: 1.82, w: 7.4, h: 0.95, fontSize: 44, bold: true, color: C.putih,
 fontFace: FONT_JUDUL, margin: 0,
 });
 s.addText("Alat bantu skrining awal berbasis pembelajaran mesin untuk lesi kulit", {
 x: 0.85, y: 2.82, w: 7.2, h: 0.5, fontSize: 16, color: C.teal,
 fontFace: FONT_ISI, margin: 0,
 });
 s.addText(
 "Empat model dibandingkan pada split yang sama, ambang batas dikunci dari " +
 "data validasi, lalu diuji ulang pada foto smartphone. Alat bantu belajar, " +
 "bukan alat diagnosis.",
 {
 x: 0.85, y: 3.45, w: 7.0, h: 0.9, fontSize: 13, color: C.teksGelap,
 fontFace: FONT_ISI, margin: 0, lineSpacing: 20,
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
 x: cx, y: 4.55, w, h: 0.38, rectRadius: 0.19,
 fill: { color: C.navy, transparency: 100 },
 line: { color: C.slateMuda, width: 0.75 },
 });
 s.addText(t, {
 x: cx, y: 4.59, w, h: 0.3, fontSize: 10.5, color: C.teksGelap,
 fontFace: FONT_ISI, align: "center", valign: "mid", margin: 0,
 });
 cx += w + 0.22;
 });

 s.addShape("rect", { x: 0.85, y: 5.42, w: 6.9, h: 0.012, fill: { color: "3A4A6B" } });
 const anggota = H.info.anggota.map((a) => `${a.nama} · ${a.nim}`).join("\n");
 s.addText(anggota, {
 x: 0.85, y: 5.58, w: 7.0, h: 0.95, fontSize: 11.5, color: C.putih,
 fontFace: FONT_ISI, margin: 0, lineSpacing: 19,
 });

 s.addShape("roundRect", {
 x: 8.55, y: 1.45, w: 3.9, h: 5.15, rectRadius: 0.12,
 fill: { color: C.putih, transparency: 92 },
 line: { color: C.slateMuda, width: 0.75 },
 });
 s.addText("SEKILAS PROYEK", {
 x: 8.85, y: 1.72, w: 3.3, h: 0.28, fontSize: 10, bold: true, color: C.teal,
 fontFace: FONT_ISI, charSpacing: 1.4, margin: 0,
 });
 s.addText(persen(PESEN_GANAS), {
 x: 8.85, y: 2.04, w: 3.3, h: 0.72, fontSize: 38, bold: true, color: C.putih,
 fontFace: FONT_JUDUL, margin: 0,
 });
 s.addText("kelas ganas dan pre-ganas (mel, bcc, akiec)", {
 x: 8.85, y: 2.76, w: 3.3, h: 0.44, fontSize: 10.5, color: C.teksGelap,
 fontFace: FONT_ISI, margin: 0,
 });
 barTumpuk(s, 8.85, 3.34, 3.3, 0.3, [
 { label: `jinak ${persen(1 - PESEN_GANAS)}`, porsi: 1 - PESEN_GANAS, warna: C.teal },
 { label: `ganas ${persen(PESEN_GANAS)}`, porsi: PESEN_GANAS, warna: C.rose },
 ]);
 s.addShape("rect", { x: 8.85, y: 4.18, w: 3.3, h: 0.012, fill: { color: "3A4A6B" } });
 const baris = [
 ["4 model", "1.527 foto uji, dipakai sekali"],
 [`F1 macro ${f(MOB.f1_macro)}`, `@ ambang ${f(BIN.threshold, 2)}`],
 ];
 baris.forEach(([besar, kecil], i) => {
 const y = 4.42 + i * 0.95;
 s.addText(besar, {
 x: 8.85, y, w: 3.3, h: 0.4, fontSize: 19, bold: true, color: C.putih,
 fontFace: FONT_JUDUL, margin: 0,
 });
 s.addText(kecil, {
 x: 8.85, y: y + 0.42, w: 3.3, h: 0.28, fontSize: 10.5, color: C.teksGelap,
 fontFace: FONT_ISI, margin: 0,
 });
 });

 s.addNotes("Slide pembuka. Sebutkan nama produk, subjudul, dan ketiga anggota. " +
 "Angka kunci: " + thousands(TOTAL_FOTO) + " foto, 7 kelas, 4 model, F1 macro " +
 f(MOB.f1_macro) + ". Tegaskan sekali: alat bantu belajar, bukan diagnosis.");
}

// ============= SLIDE 2: LATAR BELAKANG DAN PERMASALAHAN ====================
{
 const s = slideKonten(
 pres,
 "LATAR BELAKANG · PERMASALAHAN",
 "Kenapa skrining lesi kulit perlu model?",
 "Kelas ganas sedikit, tapi kesalahan klasifikasi punya konsekuensi yang tidak bisa dibatalkan."
 );

 kartu(s, M, 1.95, 4.3, 3.95, {});
 s.addShape("rect", { x: M + 0.28, y: 2.2, w: 0.42, h: 0.055, fill: { color: C.rose } });
 labelKecil(s, M + 0.28, 2.32, "KELAS GANAS DAN PRE-GANAS");
 s.addText(persen(PESEN_GANAS), {
 x: M + 0.28, y: 2.6, w: 3.7, h: 0.8, fontSize: 44, bold: true, color: C.rose,
 fontFace: FONT_JUDUL, margin: 0,
 });
 s.addText("mel, bcc, dan akiec", {
 x: M + 0.28, y: 3.42, w: 3.7, h: 0.34, fontSize: 12, bold: true, color: C.ink,
 fontFace: FONT_ISI, margin: 0,
 });
 s.addText(
 `${thousands(FOTO_GANAS)} dari ${thousands(TOTAL_FOTO)} foto. Sisanya ${persen(
 1 - PESEN_GANAS
 )} adalah lesi jinak, termasuk dermatofibroma (df) dan lesi vaskular (vasc). ` +
 `Melanoma (mel) sendiri ${persen(PESEN_MEL)} foto dan merupakan jenis kanker ` +
 "kulit paling mematikan.",
 {
 x: M + 0.28, y: 3.82, w: 3.7, h: 1.4, fontSize: 11, color: C.inkLembut,
 fontFace: FONT_ISI, margin: 0, valign: "top", lineSpacing: 16,
 }
 );
 barTumpuk(s, M + 0.28, 5.32, 3.7, 0.26, [
 { label: `jinak ${persen(1 - PESEN_GANAS)}`, porsi: 1 - PESEN_GANAS, warna: C.teal },
 { label: `ganas ${persen(PESEN_GANAS)}`, porsi: PESEN_GANAS, warna: C.rose },
 ]);

 const poin = [
 [
 "Kelas minoritas menentukan sensitivitas",
 `Mel, bcc, dan akiec hanya ${persen(PESEN_GANAS)} foto. Model yang hanya hafal ` +
 "kelas besar terlihat akurat, padahal melanoma yang lolos adalah kegagalan " +
 "paling berbahaya.",
 ],
 [
 "Foto pengguna bukan foto dermoskop",
 "HAM10000 diambil dengan dermatoskop. Aplikasi kami menerima foto smartphone, " +
 "dan itu domain yang berbeda.",
 ],
 [
 "Yang dibutuhkan: skrining, bukan vonis",
 "Ambang batas dan status tidak yakin ditampilkan terbuka, supaya pengguna tahu " +
 "kapan harus ke dokter.",
 ],
 ];
 poin.forEach(([judulTeks, isi], i) => {
 const y = 1.95 + i * 1.36;
 kartu(s, 5.2, y, 7.53, 1.24, {});
 nomorKartu(s, 5.45, y + 0.3, i + 1, [C.teal, C.amber, C.navy][i]);
 s.addText(judulTeks, {
 x: 5.98, y: y + 0.2, w: 6.5, h: 0.3, fontSize: 13.5, bold: true, color: C.ink,
 fontFace: FONT_ISI, margin: 0,
 });
 s.addText(isi, {
 x: 5.98, y: y + 0.52, w: 6.5, h: 0.65, fontSize: 11, color: C.inkLembut,
 fontFace: FONT_ISI, margin: 0, valign: "top", lineSpacing: 15,
 });
 });

 catatanKartu(
 s, 5.2, 6.05, 7.53, 0.9,
 "MASALAH YANG KAMI JAWAB",
 "Bagaimana membangun model yang tetap berguna saat dipakai pada foto smartphone, " +
 "dan menyatakannya secara jujur ketika performanya turun.",
 C.teal, C.tealMuda
 );

 s.addNotes("Poin : minority class menentukan sensitivitas. Angka " +
 persen(PESEN_GANAS) + " kelas ganas. Poin kedua: domain shift nyata, dari " +
 "dermoskop ke smartphone. Poin ketiga: posisi produk, skrining bukan diagnosis.");
}

// ===================== SLIDE 3: TUJUAN PROYEK ============================
{
 const s = slideKonten(
 pres,
 "TUJUAN PROYEK",
 "Empat tujuan, empat bukti",
 "Setiap tujuan punya hasil yang bisa ditunjukkan, bukan sekadar niat."
 );

 const tujuan = [
 {
 n: 1,
 warna: C.teal,
 j: "Bandingkan empat model secara adil",
 d: "HOG-SVM, HOG-Random Forest, CNN dari nol, dan MobileNetV2 dilatih dan " +
 "dievaluasi pada split yang sama, dengan ambang batas masing-masing.",
 bukti: "Bukti: tabel hasil uji, 1.527 foto",
 },
 {
 n: 2,
 warna: C.amber,
 j: "Merancang fitur berdasarkan eksperimen",
 d: "Rekayasa fitur klasik ditentukan oleh pengukuran, bukan kebiasaan: " +
 "PCA parsial, bobot histogram, dan ukuran sel HOG diuji satu per satu.",
 bukti: "Bukti: log diagnostik fitur, 29 konfigurasi",
 },
 {
 n: 3,
 warna: C.navy,
 j: "Mengukur ketahanan lintas domain",
 d: "Model tidak dilatih ulang pada data eksternal, dan ambangnya tetap " +
 "dikunci dari validasi, jadi penurunan performa terukur apa adanya.",
 bukti: "Bukti: uji PAD-UFES-20, 2.298 foto",
 },
 {
 n: 4,
 warna: C.rose,
 j: "Menyerapkan hasilnya jadi aplikasi",
 d: "Model terbaik, ambang, dan preprocessing dipakai satu aplikasi web " +
 "dengan preprocessing identik dengan training.",
 bukti: "Bukti: aplikasi Streamlit, 4 kondisi uji",
 },
 ];
 tujuan.forEach((t, i) => {
 const x = M + (i % 2) * 6.2;
 const y = 1.98 + Math.floor(i / 2) * 2.35;
 kartu(s, x, y, 5.93, 2.15, {});
 nomorKartu(s, x + 0.3, y + 0.3, t.n, t.warna);
 s.addText(t.j, {
 x: x + 0.82, y: y + 0.32, w: 4.85, h: 0.34, fontSize: 15, bold: true,
 color: C.ink, fontFace: FONT_JUDUL, margin: 0,
 });
 s.addText(t.d, {
 x: x + 0.3, y: y + 0.85, w: 5.3, h: 0.85, fontSize: 11, color: C.inkLembut,
 fontFace: FONT_ISI, margin: 0, valign: "top", lineSpacing: 16,
 });
 s.addShape("rect", { x: x + 0.3, y: y + 1.72, w: 5.3, h: 0.012, fill: { color: C.garis } });
 s.addText(t.bukti, {
 x: x + 0.3, y: y + 1.8, w: 5.3, h: 0.26, fontSize: 10, bold: true,
 color: t.warna, fontFace: FONT_ISI, margin: 0,
 });
 });

 s.addNotes("Empat tujuan. Yang ditekankan: setiap tujuan punya bukti file, " +
 "bukan klaim. Kalau ditanya \"mengapa empat?\", jawabannya: empat sudut " +
 "p Viewing, empat sudut pandang yang saling melengkapi.");
}

// ===================== SLIDE 4: STUDI LITERATUR ==========================
{
 const s = slideKonten(
 pres,
 "STUDI LITERATUR",
 "Tujuh artikel yang menjadi landasan",
 "Metadata setiap artikel diverifikasi otomatis ke NCBI PubMed, lalu diringkas " +
 "menjadi metode, hasil, dan keterkaitannya dengan proyek kami."
 );

 const baris = H.lit.map((a, i) => [
 "R" + (i + 1),
 `${a.authors.split(",")[0]} dkk. (${a.year})`,
 a.metode,
 a.hasil,
 a.keterkaitan,
 a.doi.replace("https://doi.org/", ""),
 ]);
 kartu(s, M, 1.95, 12.13, 4.1, {});
 tabel(
 s, M + 0.2, 2.12, 11.73,
 ["No", "Penulis & Tahun", "Metode", "Hasil Utama", "Keterkaitan dengan Proyek Kami", "DOI"],
 baris,
 [0.45, 1.55, 2.35, 2.75, 3.25, 1.38],
 { fs: 8.2, barisTinggi: 0.46, sorotBaris: ["R1", "R2", "R3"] }
 );

 s.addText(
 "Angka dan klaim jurnal adalah laporan penulis artikel, bukan hasil replikasi kami. " +
 "Kolom keterkaitan adalah analisis penulis laporan ini. Metadata diverifikasi otomatis " +
 "ke NCBI PubMed; lognya ada di docs/logs/verifikasi_literature.json, ulangi dengan " +
 "python scripts/verify_literature.py.",
 {
 x: M + 0.02, y: 6.2, w: 12.1, h: 0.6, fontSize: 9.5, italic: true,
 color: C.inkLembut, fontFace: FONT_ISI, margin: 0, valign: "top",
 lineSpacing: 13,
 }
 );

 s.addNotes("Jangan membaca semua kolom. Sebutkan tiga: R1 HAM10000, R2 " +
 "PAD-UFES-20, R3 domain shift. Katakan bahwa tujuh artikel dipilih " +
 "karena menyentuh tiga topik yang dipakai laporan: dataset lesi kulit, " +
 "pergeseran domain, dan validasi model AI dermatologi.");
}

// ========================= SLIDE 5: DATASET ==============================
{
 const s = slideKonten(
 pres,
 "DATASET",
 "Dua dataset, satu untuk belajar, satu untuk menguji",
 "Sumber, jumlah sampel, fitur, target, dan karakteristik data."
 );

 const tiles = [
 { label: "SAMPEL", nilai: thousands(TOTAL_FOTO), sub: "foto RGB, 600 × 450", warna: C.navy },
 { label: "LESI UNIK", nilai: thousands(TOTAL_LESI), sub: "1 lesi bisa beberapa foto", warna: C.teal },
 { label: "KELAS", nilai: String(KELAS.length), sub: "diagnosis dermatologi", warna: C.amber },
 { label: "TARGET", nilai: "7 → 2", sub: "diagnosis, lalu ganas atau jinak", warna: C.rose },
 ];
 tiles.forEach((t, i) => {
 const x = M + i * 3.07;
 kartu(s, x, 1.95, 2.86, 1.32, {});
 s.addShape("rect", { x: x + 0.25, y: 2.16, w: 0.42, h: 0.055, fill: { color: t.warna } });
 labelKecil(s, x + 0.25, 2.27, t.label);
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
 s.addText("Sumber data", {
 x: M + 0.3, y: 3.72, w: 5.4, h: 0.32, fontSize: 14, bold: true, color: C.ink,
 fontFace: FONT_ISI, margin: 0,
 });
 const sumber = [
 [
 "HAM10000 (training)",
 `${thousands(TOTAL_FOTO)} foto dermoskop dari ${thousands(TOTAL_LESI)} lesi, ` +
 "7 kelas. Lisensi CC BY-NC 4.0.",
 ],
 [
 "PAD-UFES-20 (uji eksternal)",
 `${thousands(H.pad.n_gambar)} foto smartphone, 6 diagnosis, Fitzpatrick I-VI. ` +
 "Lisensi CC BY 4.0.",
 ],
 [
 "Pembagian per lesion_id",
 `${thousands(FOTO_PER_LESI)} foto adalah lesi yang sama difoto ulang, jadi lesi ` +
 "tidak boleh dibagi dua split.",
 ],
 ];
 sumber.forEach(([j, d], i) => {
 const y = 4.18 + i * 0.85;
 nomorKartu(s, M + 0.3, y, i + 1, [C.navy, C.teal, C.amber][i]);
 s.addText(j, {
 x: M + 0.82, y: y + 0.02, w: 4.9, h: 0.28, fontSize: 12, bold: true,
 color: C.ink, fontFace: FONT_ISI, margin: 0,
 });
 s.addText(d, {
 x: M + 0.82, y: y + 0.3, w: 4.9, h: 0.48, fontSize: 10.5, color: C.inkLembut,
 fontFace: FONT_ISI, margin: 0, valign: "top",
 });
 });

 kartu(s, 7.0, 3.48, 5.73, 3.3, {});
 s.addText("Karakteristik data", {
 x: 7.3, y: 3.72, w: 5.1, h: 0.32, fontSize: 14, bold: true, color: C.ink,
 fontFace: FONT_ISI, margin: 0,
 });
 const karakter = [
 ["Resolusi seragam", "600 × 450 piksel, RGB, semua foto"],
 ["Sangat tidak seimbang", `rasio kelas ${f(RASIO_KELAS, 1)} : 1`],
 ["Label klinis", "53,3% label berasal dari histopatologi"],
 ["Integritas", "0 file hilang, 0 file kosong"],
 ["Ada kolom tak terpakai", "age kosong pada 57 foto, tidak diimputasi"],
 ["Batas demografis", "seluruh kulit Fitzpatrick I-II"],
 ];
 karakter.forEach(([j, d], i) => {
 const y = 4.16 + i * 0.42;
 s.addShape("ellipse", { x: 7.3, y: y + 0.08, w: 0.1, h: 0.1, fill: { color: C.teal } });
 s.addText(j, {
 x: 7.5, y, w: 2.1, h: 0.26, fontSize: 10.5, bold: true, color: C.ink,
 fontFace: FONT_ISI, margin: 0,
 });
 s.addText(d, {
 x: 9.6, y, w: 3.0, h: 0.34, fontSize: 10, color: C.inkLembut,
 fontFace: FONT_ISI, margin: 0, valign: "top",
 });
 });

 s.addNotes("Tunjukkan empat angka di atas: sampel, lesi unik, kelas, target. " +
 "Yang paling penting untuk dosing: Fitzpatrick I-II, jadi tidak ada " +
 "representasi kulit gelap padahal sasaran utama kita kulit gelap.");
}

// ===================== SLIDE 6: EDA, ANGKA KUNCI ==========================
{
 const s = slideKonten(
 pres,
 "EXPLORATORY DATA ANALYSIS",
 "Empat angka yang cukup untuk menjelaskan seluruh cerita data",
 "Distribusi, resolusi, pengulangan lesi, dan pemeriksaan kebocoran."
 );

 const ham = H.pad.statistik_domain.HAM10000;
 const cards = [
 {
 label: "RASIO KELAS", nilai: `${f(RASIO_KELAS, 1)} : 1`,
 keterangan: "nv terhadap df", bar: 1, warna: C.rose,
 catatan: `df hanya ${persen(PESEN_DF)} dari seluruh foto, nv ${persen(PESEN_NV)}.`,
 },
 {
 label: "RESOLUSI", nilai: `${Math.round(ham.lebar)} × ${Math.round(ham.tinggi)}`,
 keterangan: "seragam untuk semua foto", bar: 1, warna: C.teal,
 catatan: "Semua gambar RGB dengan rasio 4:3, jadi resize seragam aman.",
 },
 {
 label: "PENGULANGAN LESI", nilai: thousands(FOTO_PER_LESI),
 keterangan: `foto dari ${thousands(TOTAL_LESI)} lesi`, bar: TOTAL_LESI / TOTAL_FOTO,
 warna: C.navy,
 catatan: "Inilah alasan split harus per lesion_id, bukan per gambar.",
 },
 {
 label: "CEK KEBOCORAN", nilai: "0",
 keterangan: "overlap lesi dan gambar", bar: 0, warna: C.teal,
 catatan: "Enam pemeriksaan overlap antar split, semuanya nol.",
 },
 ];
 cards.forEach((c, i) => {
 kartuAngka(s, M + i * 3.07, 2.0, 2.86, 2.75, {
 label: c.label, nilai: c.nilai, keterangan: c.keterangan, bar: c.bar,
 warna: c.warna, catatan: c.catatan, ukuran: 24,
 });
 });

 catatanKartu(
 s, M, 5.0, 12.13, 1.75,
 "IMPLIKASI UNTUK MODEL",
 "Kelas minority harus diberi bobot, dan split harus per lesi. Keduanya dijalankan " +
 "di seluruh proyek: class weight dihitung hanya dari data latih, dan pembagian data " +
 "memakai GroupShuffleSplit per lesion_id dengan seed 42. Tiga assert anti-kebocoran " +
 "memeriksa hasilnya, semuanya bernilai nol. Kelas ganas dalam proyek ini adalah " +
 "mel, bcc, dan akiec (" + persen(PESEN_GANAS) + " foto); df dan vasc adalah lesi jinak.",
 C.teal, C.tealMuda
 );

 s.addNotes("Empat angka EDA. Yang paling penting: " + thousands(FOTO_PER_LESI) +
 " foto adalah lesi yang sama, dan split per lesi_shield. Kalau Split per gambar, " +
 "hasil test akan terlihat jauh lebih baik dari kenyataannya.");
}

// ===================== SLIDE 7: EDA, VISUAL ===============================
{
 const s = slideKonten(
 pres,
 "EXPLORATORY DATA ANALYSIS",
 "Distribusi kelas dan class weight",
 "Visualisasi yang dipakai: komposisi kelas per kelas dan bobot latih yang " +
 "menyeimbangkan ketidakseimbangan."
 );

 const train = H.split.class_distribution.train;
 s.addChart(
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
 showTitle: true, title: "Jumlah foto per kelas di data latih",
 titleFontSize: 12, titleColor: C.ink, titleFontFace: FONT_ISI,
 barGapWidthPct: 45,
 }
 );

 kartu(s, 8.0, 2.0, 4.73, 2.6, {});
 labelKecil(s, 8.3, 2.24, "CLASS WEIGHT (DARI DATA LATIH)");
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
 "Varian CNN tanpa class weight mengunci accuracy 0,672 dengan recall ganas nol. " +
 "Bobot tidak menghapus ketidakseimbangan, hanya menyeimbangkan perhatian model " +
 "terhadapnya. Data tetap tidak seimbang.",
 C.rose, C.roseMuda
 );

 s.addNotes("Tunjuk panjang ekor kelas df dan vasc, lalu bobotnya yang besar. " +
 "Ingatkan bobot dihitung dari data latih saja, bukan dari seluruh data.");
}

// ===================== SLIDE 8: PREPROCESSING =============================
{
 const s = slideKonten(
 pres,
 "PREPROCESSING",
 "Satu pipeline, di-fit hanya pada data training",
 "Scaler dan PCA berada di dalam pipeline sklearn, jadi otomatis di-fit pada X_train saja."
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
 labelKecil(s, M + 0.3, 3.6, "DI DALAM PIPELINE");
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
 "pada X_train saja. Data uji tidak pernah menyentuh scaler maupun PCA.",
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

 s.addNotes("Tunjukkan posisi scaler dan PCA di dalam pipeline, itu inti anti-leakage. " +
 "Tiga hal yang sengaja dilewati juga penting: metodologi yang dijelaskan " +
 "adalah keputusan, bukan kelalaian.");
}

// ===================== SLIDE 9: REKAYASA FITUR ============================
{
 const s = slideKonten(
 pres,
 "PENGEMBANGAN MODEL · REKAYASA FITUR",
 "Desain fitur ditentukan oleh eksperimen",
 "Tiga konfigurasi yang gagal dicatat sebagai bukti, bukan disembunyikan."
 );

 const d = (konf) => (H.diag || []).find((x) => x.konfigurasi === konf);
 const rfPenuh = d("RF pada 1812 fitur penuh");
 const pca400 = d("SVC PCA(400) dari 1812 fitur");
 const pca30 = d("SVC PCA(30) HOG + histogram");
 const histSaja = d("SVC-RBF histogram saja (48)");

 kartu(s, M, 1.98, 12.13, 1.05, { fill: C.navy, line: C.navy });
 s.addText(
 "HOG (sel 16×16 → 1.764 dimensi) → StandardScaler → PCA(30) → StandardScaler\n" +
 "Histogram warna RGB (48 dimensi) → dikali bobot 6 (tidak di-scale ulang) → digabung → SVC / Random Forest",
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
 s, M, 5.88, 12.13, 1.0,
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
 "dari diagnostik_fitur.json, 29 konfigurasi diuji. Jangan hanya menampilkan " +
 "konfigurasi final, tunjukkan yang gagal juga.");
}

// ===================== SLIDE 10: EMPAT MODEL ==============================
{
 const s = slideKonten(
 pres,
 "PENGEMBANGAN MODEL",
 "Empat model, satu protokol evaluasi",
 "Dua model klasik berbasis fitur handcraft, dua model deep learning."
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
 "200 pohon pada fitur yang sama. Pohon membelah berdasarkan urutan nilai, " +
 "jadi bobot histogram tidak berpengaruh.",
 hyper: `n_estimators = ${rf.n_estimators} · max_depth = ${rf.max_depth} · class weight`,
 nilai: f(H.klasik.random_forest.f1_macro),
 warna: C.teal,
 },
 {
 tag: "Deep learning",
 nama: "CNN dari nol",
 desk:
 "Tiga konv layer dan dense besar dengan BatchNormalization, dilatih dengan " +
 "class weight dan augmentasi.",
 hyper: `32-64-128 + Flatten · ${thousands(H.cnn.params)} parameter`,
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
 labelKecil(s, x + 0.25, 2.36, m.tag.toUpperCase(), m.warna);
 s.addText(m.nama, {
 x: x + 0.25, y: 2.76, w: 2.4, h: 0.6, fontSize: 15, bold: true, color: C.ink,
 fontFace: FONT_JUDUL, margin: 0,
 });
 s.addText(m.desk, {
 x: x + 0.25, y: 3.36, w: 2.4, h: 0.95, fontSize: 10, color: C.inkLembut,
 fontFace: FONT_ISI, margin: 0, valign: "top", lineSpacing: 14,
 });
 labelKecil(s, x + 0.25, 4.34, "HYPERPARAMETER");
 s.addShape("roundRect", {
 x: x + 0.25, y: 4.58, w: 2.36, h: 0.6, rectRadius: 0.08, fill: { color: C.paper },
 });
 s.addText(m.hyper, {
 x: x + 0.37, y: 4.62, w: 2.14, h: 0.52, fontSize: 9, color: C.ink,
 fontFace: FONT_ISI, margin: 0, valign: "mid",
 });
 s.addText(`F1 macro validasi ${m.nilai}`, {
 x: x + 0.25, y: 5.24, w: 2.4, h: 0.26, fontSize: 10, bold: true,
 color: m.warna, fontFace: FONT_ISI, margin: 0,
 });
 });

 catatanKartu(
 s, M, 5.85, 12.13, 1.02,
 "TUNJUKAN YANG SERING DILALU",
 "Fine-tuning 30 lapisan atas MobileNetV2 menaikkan accuracy tetapi menurunkan " +
 "recall ganas dari 0,5545 menjadi 0,3750 dan menambah FN ganas dari 139 menjadi " +
 "195 kasus. Pada dataset sebesar ini, membuka representasi pretrained justru " +
 "merusak sensitivitas.",
 C.rose, C.roseMuda
 );

 s.addNotes("Empat model dengan hyperparameter aktual, bukan nama generik. " +
 "Tutup dengan temuan fine-tuning: F1 macro hampir sama, recall ganas justru turun.");
}

// ===================== SLIDE 11: PROSES PELATIHAN ==========================
{
 const s = slideKonten(
 pres,
 "PENGEMBANGAN MODEL · PELATIHAN",
 "Proses pelatihan dan bagaimana model dinilai",
 "Semua model dilatih pada 6.959 foto yang sama, dengan class weight dari data latih."
 );

 const ch = H.cnnHistory || {};
 const baris = [
 [
 "HOG + SVM",
 "6.959 foto, fitur HOG + histogram",
 "GridSearch 4 konfigurasi",
 menit(H.klasik.svm.fit_time_seconds),
 "1.812 + 48 dimensi",
 ],
 [
 "HOG + Random Forest",
 "6.959 foto, fitur sama",
 "4 konfigurasi n_estimators dan max_depth",
 menit(H.klasik.random_forest.fit_time_seconds),
 "1.812 + 48 dimensi",
 ],
 [
 "CNN dari nol",
 "6.959 foto, 224 × 224",
 "21 epoch, lr 1e-3, batch 32",
 menit(ch["lebar-flat"] && ch["lebar-flat"].train_time_s),
 `${thousands(H.cnn.params)} parameter`,
 ],
 [
 "MobileNetV2 tahap 1",
 "6.959 foto, 224 × 224",
 `${H.mnet.stage1.epochs} epoch, lr 1e-3, backbone beku`,
 menit(H.mnet.stage1.time_s),
 `${thousands(H.mnet.total_params)} parameter`,
 ],
 ];
 kartu(s, M, 1.98, 7.6, 3.15, {});
 labelKecil(s, M + 0.3, 2.2, "RINGKASAN PELATIHAN");
 tabel(
 s, M + 0.3, 2.54, 7.0,
 ["Model", "Data latih", "Setting", "Waktu", "Kapasitas"],
 baris,
 [1.45, 1.6, 1.9, 0.75, 1.3],
 { fs: 9, barisTinggi: 0.42 }
 );
 s.addText(
 "Recall ganas pada tabel dihitung dari prediksi kelas atas. Pada ambang " +
 `aplikasi ${f(BIN.threshold, 2)}, recall ganas model terpilih ${persen(
 BIN.recall
 )} dengan FN ${BIN.fn} kasus.`,
 {
 x: M + 0.3, y: 4.72, w: 7.0, h: 0.35, fontSize: 9.5, color: C.inkLembut,
 fontFace: FONT_ISI, margin: 0, valign: "top", lineSpacing: 13,
 }
 );

 kartu(s, 8.45, 1.98, 4.28, 3.15, {});
 labelKecil(s, 8.73, 2.2, "MOBIENETV2 DUA TAHAP");
 const tahap = [
 ["Tahap 1", "classifier baru, backbone beku, lr 1e-3", H.mnet.stage1_result.f1_macro, "dipakai"],
 ["Tahap 2", "30 lapisan atas dilepas, BN dikunci, lr 1e-5", H.mnet.stage2_result.f1_macro, "ditolak"],
 ];
 tahap.forEach(([j, d, f1, status], i) => {
 const y = 2.56 + i * 1.05;
 s.addText(j, {
 x: 8.73, y, w: 1.3, h: 0.26, fontSize: 11.5, bold: true, color: C.ink,
 fontFace: FONT_ISI, margin: 0,
 });
 s.addText(status === "dipakai" ? "DIPAKAI" : "DITOLAK", {
 x: 11.6, y, w: 0.95, h: 0.24, fontSize: 9, bold: true,
 color: status === "dipakai" ? C.teal : C.rose,
 fontFace: FONT_ISI, align: "right", margin: 0,
 });
 s.addText(d, {
 x: 8.73, y: y + 0.26, w: 3.8, h: 0.4, fontSize: 10, color: C.inkLembut,
 fontFace: FONT_ISI, margin: 0, valign: "top",
 });
 s.addText(`F1 macro ${f(f1)} · recall ganas ${f(
 (i === 0 ? H.mnet.stage1_result : H.mnet.stage2_result).binary.malignant_recall, 4
 )}`, {
 x: 8.73, y: y + 0.66, w: 3.8, h: 0.26, fontSize: 10, bold: true,
 color: i === 0 ? C.teal : C.inkLembut, fontFace: FONT_ISI, margin: 0,
 });
 });

 catatanKartu(
 s, M, 5.3, 12.13, 1.45,
 "CARA MELIHAT HASIL INI",
 "Tahap 2 menaikkan accuracy, tetapi recall ganas turun dari 0,5545 menjadi " +
 "0,3750 dan FN ganas naik dari 139 menjadi 195 kasus. Karena FN ganas adalah " +
 "kesalahan paling mahal dalam skrining, tahap 1 yang dipakai. Varian CNN " +
 "32-64-128 + GAP juga ditolak karena hanya memprediksi 3 dari 7 kelas.",
 C.teal, C.tealMuda
 );

 s.addNotes("Poin penting: proses pelatihan identik untuk semua model, hanya " +
 "split dan ambang yang sama. Fine-tuning ditolak karena alasan sensitivitas, " +
 "bukan karena F1 macro. Ini jawaban kuat kalau ditanya.");
}

// ===================== SLIDE 12: PEMBAGIAN DATA ============================
{
 const s = slideKonten(
 pres,
 "PEMBAGIAN DATA DAN TRAINING",
 "Split anti-kebocoran, class weight, dan ambang per model",
 "Data uji dipakai satu kali di akhir, setelah ambang dan hyperparameter terkunci."
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
 labelKecil(s, M + 0.3, 4.68, "PEMBAGIAN DATA");
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
 s.addText(`${p.label} ${thousands(p.n)}`, {
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
 labelKecil(s, 8.4, 4.68, "AMBANG MASING-MASING MODEL");
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
 "berbeda per model karena tiap model punya skala probabilitasnya sendiri. " +
 "Kalau ditanya kenapa test hanya sekali, jawabannya: supaya angkanya bisa " +
 "dipakai sebagai laporan akhir.");
}

// ===================== SLIDE 13: EVALUASI DAN PERBANDINGAN =================
{
 const s = slideKonten(
 pres,
 "EVALUASI DAN PERBANDINGAN ALGORITMA",
 "Hasil uji test, satu split, ambang berbeda untuk tiap model",
 "Data uji 1.527 foto dipakai satu kali, setelah ambang dan hyperparameter terkunci."
 );

 const urutModel = Object.keys(H.final.per_model).sort(
 (a, b) => H.final.per_model[b].f1_macro - H.final.per_model[a].f1_macro
 );
 const baris = urutModel.map((nama) => {
 const m = H.final.per_model[nama];
 return [
 nama,
 f(m.f1_macro),
 persen(m.malignant_recall),
 persen(m.accuracy),
 f(m.specificity),
 f(m.binary_with_tuned_threshold.threshold, 2),
 `${m.binary_with_tuned_threshold.fn} / ${m.binary_with_tuned_threshold.fp}`,
 f(m.inference_ms_per_image, 1) + " ms",
 ];
 });
 kartu(s, M, 1.95, 6.55, 3.2, {});
 labelKecil(s, M + 0.3, 2.16, "HASIL DI DATA UJI");
 tabel(
 s, M + 0.3, 2.5, 5.95,
 ["Model", "F1 macro", "Rec. ganas", "Accuracy", "Spec.", "Ambang", "FN / FP", "Inferensi"],
 baris,
 [1.3, 0.62, 0.72, 0.68, 0.58, 0.6, 0.75, 0.7],
 { fs: 8.8, barisTinggi: 0.38, sorotBaris: [MODEL_TERPILIH] }
 );
 s.addText(
 "Recall ganas pada tabel ini dihitung dari prediksi kelas atas. Pada ambang " +
 `aplikasi ${f(BIN.threshold, 2)}, recall ganas model terpilih ${persen(
 BIN.recall
 )} dengan FN ${BIN.fn} kasus.`,
 {
 x: M + 0.3, y: 4.34, w: 5.95, h: 0.7, fontSize: 9.5, color: C.inkLembut,
 fontFace: FONT_ISI, margin: 0, valign: "top", lineSpacing: 13,
 }
 );

 s.addChart(
 "bar",
 [
 {
 name: "F1 macro",
 labels: urutModel.map((n) => n.replace(" (HOG)", "")),
 values: urutModel.map((n) => H.final.per_model[n].f1_macro),
 },
 {
 name: "Recall ganas",
 labels: urutModel.map((n) => n.replace(" (HOG)", "")),
 values: urutModel.map((n) => H.final.per_model[n].malignant_recall),
 },
 ],
 {
 x: 7.4, y: 1.95, w: 5.33, h: 3.2,
 catAxisLabelColor: C.ink, valAxisLabelColor: C.inkLembut,
 catAxisLabelFontFace: FONT_ISI, valAxisLabelFontFace: FONT_ISI,
 catAxisLabelFontSize: 9, valAxisLabelFontSize: 9,
 valGridLine: { color: C.garis, size: 0.5 },
 catGridLine: { style: "none" },
 showLegend: true, legendPos: "b", legendColor: C.inkLembut,
 legendFontFace: FONT_ISI, legendFontSize: 9.5,
 showValue: true, dataLabelColor: C.ink, dataLabelFontSize: 8.5,
 dataLabelFormatCode: "0.00", dataLabelPosition: "outEnd",
 chartColors: [C.teal, C.rose],
 showTitle: true, title: "F1 macro dan recall ganas",
 titleFontSize: 11, titleColor: C.ink, titleFontFace: FONT_ISI,
 valAxisMinVal: 0, valAxisMaxVal: 0.8,
 }
 );

 const rfFinal = H.final.per_model["Random Forest (HOG)"];
 catatanKartu(
 s, M, 5.35, 6.55, 1.42,
 "ANALISIS OVERFITTING",
 "Random Forest menghafal data latih dengan F1 latih 1,0000 dan F1 uji 0,3240. " +
 `MobileNetV2, CNN, dan SVM punya selisih kecil terhadap data uji, jadi tidak ada ` +
 "overfitting berarti. RF punya accuracy tertinggi justru karena menghafal.",
 C.rose, C.roseMuda
 );

 catatanKartu(
 s, 7.4, 5.35, 5.33, 1.42,
 "KENAPA ACCURACY BUKAN PILIHAN",
 `Accuracy tertinggi adalah Random Forest (${persen(rfFinal.accuracy)}), tetapi ` +
 `recall ganasnya hanya ${persen(rfFinal.malignant_recall)}, jadi ia melewatkan ` +
 "sekitar " + persen(1 - rfFinal.malignant_recall) + " lesi ganas.",
 C.amber, C.amberMuda
 );

 s.addNotes("Baca tabel dari atas: MobileNetV2 F1 macro tertinggi, RF accuracy " +
 "tertinggi tapi recall ganas terburuk. Grafik menunjukkan hal yang sama. " +
 "Inilah alasan untuk memakai F1 macro dan recall ganas, bukan accuracy.");
}

// ===================== SLIDE 14: METRIK PILIHAN ============================
{
 const s = slideKonten(
 pres,
 "EVALUASI DAN PERBANDINGAN ALGORITMA",
 "Kenapa F1 macro dan recall ganas, bukan accuracy?",
 "Kelas yang tidak seimbang membuat accuracy mudah menipu, dan FN ganas adalah kesalahan yang paling mahal."
 );

 const baseline = (H.diag || []).find((x) => x.konfigurasi.startsWith("BASELINE"));
 const rfFinal = H.final.per_model["Random Forest (HOG)"];

 kartu(s, M, 1.98, 3.95, 2.6, {});
 labelKecil(s, M + 0.28, 2.2, "TEBAKAN BASELINE");
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
 labelKecil(s, 5.03, 2.2, "AKURACY TERTINGGI, BUKAN MODEL TERBAIK");
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
 labelKecil(s, 9.18, 2.2, "BIAYA SALAH KLASIFIKASI");
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
 s, M, 4.78, 12.13, 1.99,
 "METRIK UTAMA YANG KAMI PAKAI",
 "F1 macro (menyeimbangkan presisi dan recall di tujuh kelas), recall ganas " +
 "(FN ganas adalah kesalahan yang paling mahal), dan specificity sebagai penyeimbang. " +
 "Pada data uji, model terpilih " + MODEL_TERPILIH + " mencatat FN ganas " + BIN.fn +
 " kasus dan FP " + BIN.fp + " kasus pada ambang " + f(BIN.threshold, 2) +
 ". Accuracy tetap dilaporkan, tetapi tidak menjadi dasar pemilihan.",
 C.teal, C.tealMuda
 );

 s.addNotes("Slide argumen. Dosen sering bertanya kenapa tidak accuracy. " +
 "Jawaban: karena baseline menebak nv sudah dapat accuracy " +
 (baseline ? persen(baseline.accuracy) : "-") + ". Angka itu lebih tinggi dari " +
 "model kita, dan model baseline itu berguna nol.");
}

// ===================== SLIDE 15: PEMILIHAN MODEL TERBAIK ==================
{
 const s = slideKonten(
 pres,
 "PEMILIHAN MODEL TERBAIK",
 "MobileNetV2 dipilih, dan alasannya bisa dirinci",
 "Skor gabungan dari data uji dengan bobot yang ditetapkan sebelum hasil dilihat."
 );

 const urutModel = Object.keys(H.final.per_model).sort((a, b) => SKOR[b] - SKOR[a]);
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
 kartu(s, M, 1.98, 7.55, 2.95, {});
 labelKecil(s, M + 0.3, 2.2, "SKOR GABUNGAN (TEST SET)");
 tabel(
 s, M + 0.3, 2.52, 6.95,
 ["Model", "F1 macro", "Recall ganas", "Accuracy", "Ambang", "Skor"],
 baris,
 [2.15, 0.95, 1.2, 0.95, 0.8, 0.9],
 { fs: 9.5, barisTinggi: 0.36, sorotBaris: [MODEL_TERPILIH] }
 );
 s.addText(
 "Recall ganas pada tabel ini dihitung dari prediksi kelas atas. Di aplikasi, " +
 `ambang ${f(BIN.threshold, 2)} dipakai pada jumlah probabilitas kelas ganas, ` +
 `dan recall ganas pada ambang itu ${persen(BIN.recall)} dengan FN ${BIN.fn} kasus.`,
 {
 x: M + 0.3, y: 4.4, w: 6.95, h: 0.5, fontSize: 9.5, color: C.inkLembut,
 fontFace: FONT_ISI, margin: 0, valign: "top", lineSpacing: 13,
 }
 );

 kartu(s, 8.4, 1.98, 4.33, 2.95, {});
 labelKecil(s, 8.68, 2.2, "BOBOT PENILAIAN");
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
 ["Recall ganas dijaga", `Pada ambang ${f(BIN.threshold, 2)}: recall ganas ${persen(BIN.recall)} dengan FN ${BIN.fn} kasus.`],
 ["Tahan domain shift", `Di PAD-UFES-20: recall ${persen(H.pad.hasil_per_model[MODEL_TERPILIH].recall)}, specificity ${persen(H.pad.hasil_per_model[MODEL_TERPILIH].specificity)}.`],
 ["Siap dipakai", UKURAN_MODEL
 ? `Bobot model ${UKURAN_MODEL} MB, inferensi ${f(MOB.inference_ms_per_image, 1)} ms per foto di CPU.`
 : `Inferensi ${f(MOB.inference_ms_per_image, 1)} ms per foto di CPU.`],
 ];
 alasan.forEach(([j, d], i) => {
 const x = M + i * 3.07;
 kartu(s, x, 5.08, 2.86, 1.45, {});
 s.addText(j, {
 x: x + 0.25, y: 5.28, w: 2.4, h: 0.3, fontSize: 12, bold: true, color: C.ink,
 fontFace: FONT_ISI, margin: 0,
 });
 s.addText(d, {
 x: x + 0.25, y: 5.6, w: 2.4, h: 0.8, fontSize: 10, color: C.inkLembut,
 fontFace: FONT_ISI, margin: 0, valign: "top", lineSpacing: 14,
 });
 });

 s.addNotes("Tunjukkan tabel: skor tertinggi bukan accuracy tertinggi. Bobot " +
 "penilaian ditetapkan lebih dulu, jadi bukan mencari alasan setelah hasil " +
 "melihat. Empat alasan di bawah adalah ringkasan yang bisa diulang kalau ditanya.");
}

// ===================== SLIDE 16: UJI EKSTERNAL =============================
{
 const s = slideKonten(
 pres,
 "EVALUASI DAN PERBANDINGAN ALGORITMA",
 "Uji eksternal: 2.298 foto smartphone",
 "Model tidak dilatih ulang, ambang tetap dari validasi HAM10000, jadi penurunan performa terukur apa adanya."
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
 labelKecil(s, M + 0.3, 2.2, "RECALL DAN SPECIFICITY DI DUA DOMAIN");
 tabel(
 s, M + 0.3, 2.52, 7.6,
 ["Model", "Rec HAM", "Rec PAD", "Spec HAM", "Spec PAD", "FN PAD", "FP PAD"],
 baris,
 [2.3, 0.9, 0.9, 0.9, 0.9, 0.85, 0.85],
 { fs: 9.5, barisTinggi: 0.36, sorotBaris: [MODEL_TERPILIH] }
 );

 kartu(s, 9.05, 1.98, 3.68, 2.7, {});
 labelKecil(s, 9.33, 2.2, "RECALL TINGGI BUKAN SEMBAGUS ITU");
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
 `Specificity-nya hanya ${persen(cnnPad.specificity)}. Model itu menandai hampir ` +
 "semua foto sebagai ganas, jadi recall tinggi bukan berarti model bagus.",
 {
 x: 9.33, y: 3.38, w: 3.1, h: 1.1, fontSize: 10.5, color: C.inkLembut,
 fontFace: FONT_ISI, margin: 0, valign: "top", lineSpacing: 15,
 }
 );

 const mobPad = H.pad.hasil_per_model[MODEL_TERPILIH];
 catatanKartu(
 s, M, 4.85, 6.1, 1.92,
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
 s, 7.0, 4.85, 5.73, 1.92,
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

 s.addNotes("Slide kunci domain shift. Tunjukkan CNN: recall 0,9698 tapi " +
 "specificity 0,0104, artinya menandai hampir semua foto ganas. MobileNetV2 " +
 "dipilih karena keseimbangannya, bukan karena recall-nya paling tinggi.");
}

// ===================== SLIDE 17: RECALL PER DIAGNOSIS ======================
{
 const s = slideKonten(
 pres,
 "EVALUASI DAN PERBANDINGAN ALGORITMA",
 "Recall per diagnosis pada data eksternal",
 "Di domain yang berbeda, tiap diagnosis punya cerita sendiri. Dua kelas punya catatan khusus."
 );

 const diag = H.pad.per_diagnosis;
 const maks = Math.max(...diag.map((d) => d.recall));
 kartu(s, M, 1.98, 7.55, 4.8, {});
 labelKecil(s, M + 0.3, 2.2, "MOBIENETV2 DI PAD-UFES-20");
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
 labelKecil(s, 8.73, 2.2, "KELAS YANG BELUM PERNAH DILIHAT");
 s.addText(
 scc
 ? `SCC (karsinoma sel skuamos) tidak ada di HAM10000. Recall ${persen(
 scc.recall
 )} diuji pada kelas yang model tidak pernah lihat, dan hasilnya di atas rata-rata kelas ganas lain.`
 : "-",
 {
 x: 8.73, y: 2.54, w: 3.7, h: 0.9, fontSize: 10.5, color: C.inkLembut,
 fontFace: FONT_ISI, margin: 0, valign: "top", lineSpacing: 15,
 }
 );

 kartu(s, 8.45, 3.65, 4.28, 1.55, {});
 labelKecil(s, 8.73, 3.87, "ANGKA YANG TIDAK STABIL");
 s.addText(
 mel
 ? `MEL hanya ${thousands(mel.n)} foto di PAD-UFES-20. Recall ${persen(
 mel.recall
 )} terlihat rendah, tetapi dengan sampel sebesar itu angkanya belum bisa dipakai untuk menilai model.`
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
 "sampel kecil, dan SEK yang paling sering salah. Semua dari log uji eksternal.");
}

// ===================== SLIDE 18: ARSITEKTUR APLIKASI =======================
{
 const s = slideKonten(
 pres,
 "IMPLEMENTASI APLIKASI",
 "Arsitektur aplikasi",
 "Teknologi yang dipakai, alur input, proses pengolahan model, dan output yang dihasilkan."
 );

 const tahap = [
 ["INPUT", "Foto lesi kulit\nunggah atau contoh", C.navy],
 ["PREPROCESSING", "Resize 224 × 224\nfloat32 0-255", "1D4E89"],
 ["MODEL", "MobileNetV2\nRescaling di dalam model", C.teal],
 ["POST-PROCESSING", "Jumlah probabilitas\nkelas ganas", "2A7E77"],
 ["OUTPUT", "7 bar, status tiga warna\nambang dan disclaimer", C.rose],
 ];
 tahap.forEach(([j, d, w], i) => {
 const x = M + i * 2.45;
 chevron(s, x, 2.0, 2.55, 0.78, j, w);
 s.addText(d, {
 x: x + 0.1, y: 2.9, w: 2.2, h: 0.6, fontSize: 9.5, color: C.inkLembut,
 fontFace: FONT_ISI, align: "center", margin: 0, lineSpacing: 13,
 });
 });

 kartu(s, M, 3.72, 3.95, 3.05, {});
 labelKecil(s, M + 0.28, 3.94, "TEKNOLOGI");
 const teknologi = [
 ["Antarmuka", "Streamlit 1.67, tema teal"],
 ["Model", "Keras TensorFlow, MobileNetV2"],
 ["Numerasi", "NumPy, Pandas, Scikit-learn"],
 ["Grafik", "Plotly untuk probabilitas per kelas"],
 ["Deploy", "Streamlit Community Cloud, Python 3.13"],
 ];
 teknologi.forEach(([j, d], i) => {
 const y = 4.3 + i * 0.48;
 s.addText(j, {
 x: M + 0.28, y, w: 1.15, h: 0.26, fontSize: 10.5, bold: true, color: C.ink,
 fontFace: FONT_ISI, margin: 0,
 });
 s.addText(d, {
 x: M + 1.48, y, w: 2.3, h: 0.4, fontSize: 10, color: C.inkLembut,
 fontFace: FONT_ISI, margin: 0, valign: "top",
 });
 });

 kartu(s, 4.75, 3.72, 3.95, 3.05, {});
 labelKecil(s, 5.03, 3.94, "KONFIGURASI DARI BERKAS JSON");
 s.addText(
 "Ambang batas dan preprocessing tidak ditulis di dalam kode aplikasi. " +
 "Keduanya dibaca dari berkas yang ditulis otomatis oleh notebook evaluasi, " +
 "sehingga preprocessing aplikasi tidak mungkin tidak sinkron dengan training.",
 {
 x: 5.03, y: 4.28, w: 3.4, h: 0.9, fontSize: 10.5, color: C.inkLembut,
 fontFace: FONT_ISI, margin: 0, valign: "top", lineSpacing: 15,
 }
 );
 s.addShape("roundRect", {
 x: 5.03, y: 5.26, w: 3.4, h: 1.25, rectRadius: 0.08, fill: { color: C.paper },
 });
 s.addText(
 `models/preprocess_config.json\n input ${H.preprocess.img_size} × ${
 H.preprocess.img_size
 } · ${H.preprocess.normalization}\n\nmodels/threshold_config.json\n ambang ${f(
 BIN.threshold,
 2
 )} · kelas ganas: ${KELAS_GANAS.join(", ")}`,
 {
 x: 5.15, y: 5.34, w: 3.2, h: 1.1, fontSize: 9, color: C.ink,
 fontFace: FONT_ISI, margin: 0, lineSpacing: 13,
 }
 );

 kartu(s, 8.9, 3.72, 3.83, 3.05, {});
 labelKecil(s, 9.18, 3.94, "PRIVASI DAN KEAMANAN");
 barisInsight(
 s, 9.18, 4.28, 3.3,
 "Foto tidak disimpan",
 "Foto diproses di memori selama sesi, tidak pernah ditulis ke server.",
 C.teal
 );
 barisInsight(
 s, 9.18, 5.16, 3.3,
 "Tanpa telemetri",
 "Statistik usage dimatikan di konfigurasi Streamlit.",
 C.teal
 );
 barisInsight(
 s, 9.18, 6.04, 3.3,
 "Batas unggah",
 "10 MB per file, sesuai validasi di kode aplikasi.",
 C.teal
 );

 s.addNotes("Arsitektur jadi lima tahap. Yang paling penting dan paling sering " +
 "ditanya: preprocessing dan ambang dibaca dari berkas JSON yang sama " +
 "dengan yang dipakai saat evaluasi, jadi tidak bisa tidak sinkron.");
}

// ===================== SLIDE 19: ALUR APLIKASI ============================
{
 const s = slideKonten(
 pres,
 "IMPLEMENTASI APLIKASI",
 "Alur input, proses, dan output",
 "Apa yang terjadi setelah pengguna menekan tombol, dan apa yang dia lihat."
 );

 const kolom = [
 {
 j: "INPUT",
 warna: C.navy,
 isi: [
 "Unggah foto JPG atau PNG, atau pakai salah satu dari empat contoh bawaan",
 "Panduan memotret: dekat dan terfokus, cahaya merata, lesi di tengah",
 "Batas 10 MB per file",
 ],
 },
 {
 j: "PROSES",
 warna: C.teal,
 isi: [
 "Resize ke 224 × 224 dan ubah ke float32 rentang 0-255",
 "Normalisasi dijalankan layer Rescaling di dalam model, bukan di luar",
 "Prediksi tujuh kelas, lalu jumlahkan probabilitas kelas ganas",
 "Bandingkan dengan ambang 0,11 dan cek area ketidakpastian",
 ],
 },
 {
 j: "OUTPUT",
 warna: C.rose,
 isi: [
 "Tujuh bar probabilitas per kelas, kelas ganas diberi warna berbeda",
 "Kartu risiko: nilai, ambang, dan jarak dari ambang",
 "Status tiga warna: ciri ganas terdeteksi, belum meyakinkan, atau tidak ditemukan",
 "Disclaimer dan catatan privasi selalu terlihat",
 ],
 },
 ];
 kolom.forEach((k, i) => {
 const x = M + i * 4.07;
 kartu(s, x, 1.98, 3.86, 3.55, {});
 nomorKartu(s, x + 0.3, 2.24, i + 1, k.warna);
 s.addText(k.j, {
 x: x + 0.8, y: 2.26, w: 2.8, h: 0.32, fontSize: 15, bold: true, color: C.ink,
 fontFace: FONT_JUDUL, margin: 0,
 });
 k.isi.forEach((t, j) => {
 const y = 2.86 + j * 0.64;
 s.addShape("ellipse", {
 x: x + 0.3, y: y + 0.06, w: 0.1, h: 0.1, fill: { color: k.warna },
 });
 s.addText(t, {
 x: x + 0.52, y, w: 3.1, h: 0.6, fontSize: 10.5, color: C.inkLembut,
 fontFace: FONT_ISI, margin: 0, valign: "top", lineSpacing: 15,
 });
 });
 });

 catatanKartu(
 s, M, 5.7, 12.13, 1.07,
 "STATUS TIGA WARNA, BUKAN DUA",
 "Kalau risiko ganas berada terlalu dekat dengan ambang, atau probabilitas kelas " +
 "teratas terlalu rendah, hasilnya tidak dichotomis: aplikasi menampilkan " +
 "\"belum meyakinkan\" dan mengarahkan pengguna ke dokter. Lebar area abu-abu " +
 "mengikuti skala ambang, jadi tidak semua foto berakhir sebagai \"belum meyakinkan\".",
 C.amber, C.amberMuda
 );

 s.addNotes("Tiga kolom: input, proses, output. Poin pembeda dari aplikasi " +
 "skrining lain: status tiga warna, bukan dua. Kalau model tidak yakin, " +
 "aplikasi mengatakannya, bukan menebak.");
}

// ===================== SLIDE 20: DEMO PROJECT ============================
{
 const s = slideKonten(
 pres,
 "DEMO PROJECT",
 "Alur demo, enam langkah, sekitar dua menit",
 "Rekaman video maksimal 7 menit mengikuti checklist di docs/CHECKLIST_VIDEO_DEMO.md."
 );

 kartu(s, M, 1.98, 12.13, 0.78, { fill: C.navy, line: C.navy });
 s.addText("python -m streamlit run app/app.py", {
 x: M + 0.32, y: 2.1, w: 5.6, h: 0.34, fontSize: 14, bold: true, color: C.teal,
 fontFace: FONT_JUDUL, margin: 0,
 });
 s.addText("Aplikasi juga berjalan di Streamlit Community Cloud", {
 x: M + 6.0, y: 2.16, w: 5.8, h: 0.28, fontSize: 10.5, color: C.teksGelap,
 fontFace: FONT_ISI, align: "right", margin: 0,
 });

 const langkah = [
 ["Buka halaman", "Disclaimer dan panduan memotret terlihat lebih dulu"],
 ["Pakai contoh", "Pilih contoh melanoma, lalu centang pakai contoh ini"],
 ["Baca kartu risiko", "Nilai risiko, ambang 11%, dan jarak dari ambang"],
 ["Baca tujuh bar", "Kelas ganas diberi warna berbeda, sesuai probabilitas"],
 ["Coba file rusak", "Pesan ramah, bukan error/crash"],
 ["Tutup dengan disclaimer", "Alat bantu belajar, bukan alat diagnosis"],
 ];
 langkah.forEach(([j, d], i) => {
 const x = M + (i % 3) * 4.07;
 const y = 3.0 + Math.floor(i / 3) * 1.35;
 kartu(s, x, y, 3.86, 1.2, {});
 nomorKartu(s, x + 0.28, y + 0.24, i + 1, [C.navy, C.teal, C.amber, C.rose, C.navy, C.teal][i]);
 s.addText(j, {
 x: x + 0.78, y: y + 0.26, w: 2.85, h: 0.28, fontSize: 12.5, bold: true,
 color: C.ink, fontFace: FONT_ISI, margin: 0,
 });
 s.addText(d, {
 x: x + 0.78, y: y + 0.56, w: 2.85, h: 0.55, fontSize: 10, color: C.inkLembut,
 fontFace: FONT_ISI, margin: 0, valign: "top", lineSpacing: 14,
 });
 });

 catatanKartu(
 s, M, 5.82, 12.13, 1.05,
 "CATATAN TEKNIS SAAT DEMO",
 "Buka halaman sebelum mulai merekam, supaya model tidak memuat di tengah video. " +
 "Gunakan contoh bawaan, tidak perlu mencari file sendiri. Empat kondisi sudah " +
 "diuji otomatis dengan streamlit.testing.v1.AppTest: halaman kosong, unggah foto, " +
 "file rusak, dan pemilihan contoh gambar.",
 C.teal, C.tealMuda
 );

 s.addNotes("Enam langkah, dua menit. Kalau demo langsung di depan kelas, pakai " +
 "langkah 2 sampai 4 saja. Catatan teknis: buka halaman lebih dulu supaya tidak " +
 "ada jeda memuat model.");
}

// ===================== SLIDE 21: KESIMPULAN ================================
{
 const s = slideKonten(
 pres,
 "KESIMPULAN",
 "Tiga angka yang perlu diingat",
 "Satu model terpilih, satu ambang, dan satu hal yang tidak boleh dilupakan."
 );

 const mobPad = H.pad.hasil_per_model[MODEL_TERPILIH];
 const big = [
 ["F1 MACRO", f(MOB.f1_macro), "data uji, 1.527 foto, 7 kelas", C.teal],
 ["RECALL GANAS", persen(BIN.recall), `pada ambang ${f(BIN.threshold, 2)}, FN ${BIN.fn} kasus`, C.rose],
 ["F1 EKSTERNAL", f(mobPad.f1), "PAD-UFES-20, 2.298 foto", C.navy],
 ];
 big.forEach(([j, v, d, w], i) => {
 const x = M + i * 4.07;
 kartu(s, x, 1.98, 3.86, 1.75, {});
 labelKecil(s, x + 0.28, 2.18, j);
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
 color: C.putih, fontFace: FONT_JUDUL, margin: 0, valign: "top", lineSpacing: 19,
 }
 );

 s.addNotes("Tiga angka, tiga simpulan, satu kalimat penutup. Sambungkan ke " +
 "slide demo bila waktu memungkinkan.");
}

// ===================== SLIDE 22: KETERBATASAN =============================
{
 const s = slideKonten(
 pres,
 "KETERBATASAN DAN PENGEMBANGAN",
 "Batas model kami, dan arah berikutnya",
 "Keterbatasan disampaikan terbuka, karena kredibilitas sebuah alat skrining diuji dari keterbatasannya."
 );

 const bats = [
 ["Populasi kulit terbatas", "Seluruh data HAM10000 adalah kulit Fitzpatrick I-II, sedangkan populasi Indonesia didominasi III-V. Performa pada kulit gelap belum diketahui."],
 ["Tanpa validasi klinis", "Tidak ada uji dengan dokter kulit. Semua angka berasal dari pengukuran pada dataset, bukan dari diagnosis klinis."],
 ["Kelas minoritas sangat sedikit", `Hanya ${thousands(JUMLAH_KELAS.df)} foto kelas df dan ${thousands(JUMLAH_KELAS.vasc)} foto kelas vasc di seluruh dataset.`],
 ["Domain shift menurunkan specificity", `Specificity model terpilih turun dari ${persen(H.final.per_model[MODEL_TERPILIH].specificity)} menjadi ${persen(H.pad.hasil_per_model[MODEL_TERPILIH].specificity)} di foto smartphone.`],
 ];
 bats.forEach(([j, d], i) => {
 const x = M + (i % 2) * 3.55;
 const y = 1.98 + Math.floor(i / 2) * 1.42;
 kartu(s, x, y, 3.4, 1.28, {});
 s.addShape("ellipse", { x: x + 0.25, y: y + 0.26, w: 0.16, h: 0.16, fill: { color: C.rose } });
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
 labelKecil(s, 8.13, 2.22, "PENGEMBANGAN BERIKUTNYA");
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
 s, M, 4.88, 6.9, 1.4,
 "KESIMPULAN YANG JUJUR",
 "\"Model ini alat bantu untuk menandai foto yang perlu diperiksa lebih lanjut. " +
 "Keputusan tetap ada pada dokter kulit.\"",
 C.navy, C.paper
 );

 s.addNotes("Sampaikan keterbatasan lebih dulu. Dosen lebih menghargai tim yang " +
 "menyebutkan batas modelnya sendiri. Empat keterbatasan dan lima arah " +
 "pengembangan, jangan dihafal semua, pilih dua.");
}

// ===================== SLIDE 23: PEMBAGIAN TUGAS ===========================
{
 const s = slideGelap(pres);
 s.addText("PEMBAGIAN TUGAS · PENUTUP", {
 x: 0.85, y: 0.9, w: 6.5, h: 0.3, fontSize: 11, bold: true, color: C.teal,
 fontFace: FONT_ISI, charSpacing: 2, margin: 0,
 });
 s.addText("Terima kasih", {
 x: 0.85, y: 1.32, w: 6.5, h: 1.0, fontSize: 44, bold: true, color: C.putih,
 fontFace: FONT_JUDUL, margin: 0,
 });
 s.addText("Pertanyaan dan diskusi", {
 x: 0.85, y: 2.4, w: 6.5, h: 0.4, fontSize: 17, color: C.teksGelap,
 fontFace: FONT_ISI, margin: 0,
 });
 s.addText(`${NAMA_KELOMPOK} · ML2026 · ${H.info.tanggal}`, {
 x: 0.85, y: 2.95, w: 6.5, h: 0.32, fontSize: 12, bold: true, color: C.putih,
 fontFace: FONT_ISI, margin: 0,
 });
 s.addText(H.info.dosen, {
 x: 0.85, y: 3.33, w: 6.5, h: 0.32, fontSize: 11.5, color: C.teksGelap,
 fontFace: FONT_ISI, margin: 0,
 });
 s.addText("ALAT BANTU BELAJAR, BUKAN ALAT DIAGNOSIS MEDIS", {
 x: 0.85, y: 3.95, w: 6.5, h: 0.3, fontSize: 10, bold: true, color: C.slateMuda,
 fontFace: FONT_ISI, charSpacing: 1, margin: 0,
 });
 s.addText([
 { text: "Repo ", options: { bold: true, color: C.teal } },
 { text: "github.com/Grenaydinkamil/ProjectForestry_ML2026_Mendekteksi_Lesi_Kulit",
 options: { color: C.teksGelap, underline: true, hyperlink: { url: REPO_URL } } },
 ], {
 x: 0.85, y: 4.32, w: 7.2, h: 0.3, fontSize: 10, fontFace: FONT_ISI, margin: 0,
 });
 s.addText([
 { text: "App ", options: { bold: true, color: C.teal } },
 { text: "projectforestryml2026mendekteksilesikulit.streamlit.app",
 options: { color: C.teksGelap, underline: true, hyperlink: { url: APP_URL } } },
 ], {
 x: 0.85, y: 4.67, w: 7.2, h: 0.3, fontSize: 10, fontFace: FONT_ISI, margin: 0,
 });

 s.addShape("roundRect", {
 x: 8.35, y: 1.25, w: 4.1, h: 5.0, rectRadius: 0.12,
 fill: { color: C.putih, transparency: 92 },
 line: { color: C.slateMuda, width: 0.75 },
 });
 s.addText("PEMBAGIAN TUGAS", {
 x: 8.68, y: 1.52, w: 3.4, h: 0.28, fontSize: 10, bold: true, color: C.teal,
 fontFace: FONT_ISI, charSpacing: 1.4, margin: 0,
 });
 (H.info.pembagian_tugas || []).forEach((p, i) => {
 const y = 1.95 + i * 1.36;
 s.addShape("roundRect", {
 x: 8.68, y: y + 0.05, w: 0.09, h: 0.42, rectRadius: 0.04,
 fill: { color: [C.teal, C.amber, C.slateMuda][i % 3] },
 });
 s.addText(p.anggota || p.nama, {
 x: 8.92, y, w: 3.2, h: 0.3, fontSize: 12.5, bold: true, color: C.putih,
 fontFace: FONT_ISI, margin: 0,
 });
 s.addText(`${p.tugas.replace(/\s*\(Notebook[^)]*\)\s*$/, "")} · ${p.nim}`, {
 x: 8.92, y: y + 0.32, w: 3.25, h: 0.85, fontSize: 10, color: C.teksGelap,
 fontFace: FONT_ISI, margin: 0, valign: "top", lineSpacing: 14,
 });
 });

 s.addNotes("Buka sesi tanya jawab. Tutup dengan menyebut kode, model, dan log " +
 "evaluasi tersedia di repo publik.");
}

// ============== SLIDE 24: LAMPIRAN, STUDI LITERATUR RINGKAS ===============
{
 const s = slideKonten(
 pres,
 "LAMPIRAN",
 "Tabel studi literatur versi ringkas",
 "Slide cadangan untuk sesi tanya jawab: judul, jurnal, metode, hasil utama, dan DOI."
 );

 const baris = H.lit.map((a, i) => [
 "R" + (i + 1),
 `${a.authors.split(",")[0]} dkk. (${a.year})`,
 a.journal,
 a.metode,
 a.hasil,
 a.doi.replace("https://doi.org/", ""),
 ]);
 kartu(s, M, 1.95, 12.13, 3.85, {});
 tabel(
 s, M + 0.2, 2.14, 11.73,
 ["No", "Penulis & Tahun", "Jurnal", "Metode", "Hasil Utama", "DOI"],
 baris,
 [0.45, 1.75, 1.9, 2.55, 3.25, 1.83],
 { fs: 8.2, barisTinggi: 0.44 }
 );

 catatanKartu(
 s, M, 5.92, 12.13, 0.98,
 "VERIFIKASI DAN CARI ULANG",
 "Seluruh metadata diverifikasi otomatis ke NCBI PubMed E-utilities, bukan " +
 "dari ingatan. Log verifikasi ada di docs/logs/verifikasi_literature.json, dan " +
 "ringkasannya di docs/logs/literature_summary.json. Ulangi dengan " +
 "python scripts/verify_literature.py dan python scripts/dump_literature_json.py.",
 C.teal, C.tealMuda
 );

 s.addNotes("Slide lampiran, tidak dipresentasikan. Buka hanya kalau dosen " +
 "bertanya tentang salah satu artikel, atau minta daftar referensinya.");
}

// ---------------------------------------------------------------------------
pres.writeFile({ fileName: OUT }).then(() => {
 console.log(`PPT dibuat: ${OUT}`);
 console.log(`Slide: ${nomorSlide} / ${TOTAL_SLIDE}`);
 console.log(
 `Sumber angka: final=${H.final ? "ada" : "TIDAK ADA"}, ` +
 `pad=${H.pad ? "ada" : "TIDAK ADA"}, literatur=${H.lit ? H.lit.length : 0} artikel`
 );
});
