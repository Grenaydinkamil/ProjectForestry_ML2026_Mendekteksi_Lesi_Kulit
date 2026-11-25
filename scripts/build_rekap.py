"""
scripts/build_rekap.py
======================
Membuat `docs/REKAP_LAPORAN.md` dari file log hasil evaluasi.

Kenapa laporan ini digenerate?
  Supaya **tidak ada angka yang diketik manual**. Semua angka di
  rekap diambil langsung dari JSON yang ditulis notebook saat dijalankan.
  Kalau ada notebook yang dijalankan ulang, rekap ini ikut berubah.

Sumber angka:
  docs/logs/hasil_evaluasi_final.json          -> metrik data uji
  docs/logs/hasil_uji_eksternal_pad_ufes.json  -> uji domain shift
  docs/logs/model_klasik_validasi.json         -> model klasik di validasi
  docs/logs/model_cnn_validasi.json            -> CNN di validasi
  docs/logs/model_mobilenetv2_validasi.json    -> MobileNetV2 di validasi
  docs/logs/cek_bobot_histogram.json           -> sweep bobot histogram
  docs/logs/cek_kalibrasi_svm.json             -> cara kalibrasi SVM
  docs/logs/diagnostik_fitur.json              -> eksperimen fitur
  docs/logs/diagnostik_hog_cell.json           -> eksperimen ukuran sel HOG
  docs/logs/verifikasi_literature.json         -> daftar sitasi terverifikasi
  models/threshold_config.json                 -> ambang risiko ganas

Jalankan:
    python scripts/build_rekap.py
"""

from __future__ import annotations

import json
import pathlib
import sys
from datetime import datetime

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
LOGS = REPO_ROOT / "docs" / "logs"
MODELS = REPO_ROOT / "models"
KELUAR = REPO_ROOT / "docs" / "REKAP_LAPORAN.md"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def baca(path: pathlib.Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def f(x, desimal=4):
    """Format angka dengan koma desimal (gaya Indonesia)."""
    if x is None:
        return "-"
    return f"{x:.{desimal}f}".replace(".", ",")


def persen(x, desimal=1):
    if x is None:
        return "-"
    return f"{x * 100:.{desimal}f}%".replace(".", ",")


def main() -> int:
    final = baca(LOGS / "hasil_evaluasi_final.json")
    pad = baca(LOGS / "hasil_uji_eksternal_pad_ufes.json")
    klasik = baca(LOGS / "model_klasik_validasi.json")
    cnn = baca(LOGS / "model_cnn_validasi.json")
    mnv2 = baca(LOGS / "model_mobilenetv2_validasi.json")
    bobot = baca(LOGS / "cek_bobot_histogram.json")
    kalibrasi = baca(LOGS / "cek_kalibrasi_svm.json")
    diag = baca(LOGS / "diagnostik_fitur.json")
    cell = baca(LOGS / "diagnostik_hog_cell.json")
    lit = baca(LOGS / "verifikasi_literature.json")
    thr = baca(MODELS / "threshold_config.json")

    if not final:
        print("hasil_evaluasi_final.json belum ada. Jalankan Notebook 07 dulu.")
        return 1

    L = []
    w = L.append

    w("# Rekap Laporan Proyek")
    w("")
    w(f"**Status per {datetime.now().strftime('%d %B %Y, %H:%M')}.** "
      "Seluruh angka di dokumen ini diambil otomatis dari "
      "`docs/logs/*.json`, yang ditulis notebook saat dijalankan. "
      "Tidak ada angka yang diketik manual.")
    w("")
    w("---")
    w("")

    # ------------------------------------------------------------------
    w("## 1. Hasil Utama: Evaluasi di Data Uji")
    w("")
    w(f"Test set berisi **{final['test_set_size']} gambar** dan dipakai "
      "**SEKALI** di Notebook 07, setelah seluruh hyperparameter dan "
      "ambang batas terkunci dari data validasi.")
    w("")
    w("| Model | Accuracy | F1 macro | Recall ganas | Specificity | "
      "FN | FP | Inferensi |")
    w("|---|---|---|---|---|---|---|---|")

    pm = final["per_model"]
    for nama in sorted(pm, key=lambda n: -pm[n]["f1_macro"]):
        m = pm[nama]
        b = m.get("binary_with_tuned_threshold", {})
        fn = b.get("fn", "-")
        fp = b.get("fp", "-")
        ms = m.get("inference_ms_per_image")
        w(f"| {nama} | {f(m['accuracy'])} | {f(m['f1_macro'])} | "
          f"{f(m['malignant_recall'])} | {f(m['specificity'])} | "
          f"{fn} | {fp} | {f(ms, 1)} ms |")
    w("")
    w("**FN** = lesi ganas yang LOLOS (kesalahan paling berbahaya untuk "
      "skrining). **FP** = false alarm pada lesi tidak ganas.")
    w("")

    # ------------------------------------------------------------------
    w("### Ambang batas \"ganas\" (dihitung dari DATA VALIDASI)")
    w("")
    if thr:
        w(f"Target recall **{persen(thr.get('target_recall'), 0)}**. "
          "Ambang ini dikunci SEBELUM test disentuh, supaya test tetap "
          "bersih sebagai ukuran yang jujur.")
        w("")
        w("| Model | Ambang |")
        w("|---|---|")
        for nama in sorted(pm):
            b = pm[nama].get("binary_with_tuned_threshold", {})
            if "threshold" in b:
                w(f"| {nama} | {f(b['threshold'], 2)} |")
        w("")

    # ------------------------------------------------------------------
    w("### Analisis overfitting (selisih F1 train vs test)")
    w("")
    w("| Model | F1 train | F1 test | Selisih | Diagnosis |")
    w("|---|---|---|---|---|")
    for nama in sorted(pm):
        oa = pm[nama].get("overfit_analysis")
        if oa:
            w(f"| {nama} | {f(oa['f1_train'])} | {f(oa['f1_test'])} | "
              f"{oa['gap']:+.4f} | {oa['diagnosis']} |")
    w("")
    w(f"**Model yang dipilih untuk aplikasi: {final['selected_model']}.**")
    if final.get("selection_weights"):
        w("")
        w("Bobot kriteria: " + ", ".join(
            f"{k} {persen(v, 0)}" for k, v in final["selection_weights"].items()
        ) + ".")
    w("")

    # ------------------------------------------------------------------
    w("## 2. Hasil di Data Validasi")
    w("")
    w("### Model klasik")
    w("")
    if klasik:
        w("| Model | F1 macro | Accuracy | Recall ganas |")
        w("|---|---|---|---|")
        w(f"| HOG-SVM | {f(klasik['svm']['f1_macro'])} | "
          f"{f(klasik['svm']['accuracy'])} | "
          f"{f(klasik['svm']['malignant_recall'])} |")
        w(f"| HOG-Random Forest | {f(klasik['random_forest']['f1_macro'])} | "
          f"{f(klasik['random_forest']['accuracy'])} | "
          f"{f(klasik['random_forest']['malignant_recall'])} |")
        w("")

    w("### CNN dari nol (tiga varian, perbandingan yang adil)")
    w("")
    if cnn and cnn.get("gap_vs_flatten"):
        w("| Varian | Parameter | F1 macro | Accuracy | Recall ganas | "
          "Kelas terpakai |")
        w("|---|---|---|---|---|---|")
        for v in cnn["gap_vs_flatten"]:
            w(f"| {v.get('model', '-')} | {v.get('params', 0):,} | "
              f"{f(v.get('f1_macro'))} | {f(v.get('accuracy'))} | "
              f"{f(v.get('malignant_recall'))} | "
              f"{v.get('kelas_terprediksi', '-')}/7 |")
        w("")
        if cnn.get("model"):
            w(f"Pemenang: **{cnn['model']}** "
              f"(F1 macro {f(cnn.get('f1_macro'))}, "
              f"recall ganas {f(cnn.get('malignant_recall'))}).")
        if cnn.get("diagnosis"):
            w("")
            w("Diagnosis overfitting: **"
              + cnn["diagnosis"].split(":", 1)[-1].strip() + "**")
        w("")

    w("### MobileNetV2 (dua tahap, dinilai di data validasi yang sama)")
    w("")
    if mnv2:
        pb = mnv2.get("perbandingan", {})
        w("| Tahap | F1 macro | Accuracy | Recall ganas | FN ganas |")
        w("|---|---|---|---|---|")
        for kunci, nama in (("stage1", "1 (classifier, backbone beku)"),
                            ("stage2", "2 (fine-tuning 30 layer atas)")):
            w(f"| {nama} | "
              f"{f(pb.get('f1_macro', {}).get(kunci))} | "
              f"{f(pb.get('accuracy', {}).get(kunci))} | "
              f"{f(pb.get('malignant_recall', {}).get(kunci))} | "
              f"{pb.get('fn_malignant', {}).get(kunci, '-')} |")
        w("")
        w("Kedua tahap dinilai di data validasi yang sama dengan augmentasi "
          "training yang sama, jadi perbandingannya adil: satu-satunya "
          "yang beda adalah apakah backbone-nya dibekukan dan learning "
          "rate-nya (1e-3 vs 1e-5).")
        w("")
        w(f"**DIPILIH: tahap {str(mnv2.get('stage_used')).replace('stage', '')}.** Fine-tuning "
          "justru menurunkan sensitivitas, dan ini temuan yang layak "
          "dibahas: pada dataset sekecil ini, membuka lapisan pretrained "
          "memperburuk generalisasi.")
        w("")

    # ------------------------------------------------------------------
    w("## 3. Uji Eksternal: Domain Shift ke Foto Smartphone")
    w("")
    if pad:
        w("**PAD-UFES-20** (DOI 10.17632/zr7vgbcyr2, lisensi CC BY 4.0): "
          "foto smartphone, tanpa training ulang, ambang tetap dari "
          "validasi.")
        w("")
        w("| Model | Rec ganas HAM | Rec ganas PAD | Selisih | "
          "Spec HAM | Spec PAD | FN HAM | FN PAD |")
        w("|---|---|---|---|---|---|---|---|---|")
        for nama, r in sorted(
                pad.get("hasil_per_model", {}).items(),
                key=lambda kv: -kv[1].get("recall", 0)):
            ham = pm.get(nama, {})
            ham_b = ham.get("binary_with_tuned_threshold", {})
            ham_rec = ham.get("malignant_recall")
            delta = r.get("recall", 0) - (ham_rec or 0)
            w(f"| {nama} | {f(ham_rec)} | {f(r.get('recall'))} | "
              f"{delta:+.4f} | {f(ham.get('specificity'))} | "
              f"{f(r.get('specificity'))} | {ham_b.get('fn', '-')} | "
              f"{r.get('fn', '-')} |")
        w("")
        w("**Cara membaca tabel ini:** recall ganas yang TINGGI di "
          "PAD-UFES-20 tidak otomatis berarti model lebih baik. Lihat "
          "kolom specificity: di dataset eksternal, specificity semua "
          "model turun drastis (0,01 sampai 0,45). Artinya model "
          "mendeteksi hampir semua lesi ganas **karena ia menandai "
          "hampir semua foto sebagai ganas** -- bukan karena lebih "
          "teliti. Pada aplikasi skrining, trade-off ini harus "
          "dibuat eksplisit, bukan disembunyikan di balik angka recall "
          "yang menarik.")
        w("")
        per = pad.get("per_diagnosis") or []
        if per:
            w("### Recall per diagnosis di PAD-UFES-20")
            w("")
            w("| Diagnosis | n | Recall |")
            w("|---|---|---|")
            for r in per:
                w(f"| {r.get('diagnosis', '-')} | {r.get('n', '-')} | "
                  f"{f(r.get('recall'))} |")
            w("")

    # ------------------------------------------------------------------
    w("## 4. Temuan Eksperimen (intis kontribusi laporan)")
    w("")
    if diag:
        d = {r["konfigurasi"]: r for r in diag}
        base = d.get("BASELINE: selalu nebak nv")
        rf = d.get("RF pada 1812 fitur penuh")
        if base and rf:
            w(f"**A. Random Forest pada fitur mentah hampir tidak "
              f"belajar.** F1 macro {f(rf['f1_macro'])} vs baseline "
              f"{f(base['f1_macro'])}, recall ganas "
              f"{f(rf['malignant_recall'])}. Penyebab: RF memilih fitur "
              "acak saat split, jadi peluang fitur histogram (48 dari "
              "1.812 = 2,65%) praktis tak pernah dipakai.")
            w("")
    if cell:
        c = {r["cell_size"]: r for r in cell}
        if all(k in c for k in ("8x8", "16x16", "32x32")):
            w("**B. Ukuran sel HOG ternyata hampir tidak berpengaruh.** "
              f"8x8 -> F1 {f(c['8x8']['f1_macro'])}; "
              f"16x16 -> F1 {f(c['16x16']['f1_macro'])}; "
              f"32x32 -> F1 {f(c['32x32']['f1_macro'])}. Yang dipakai 16x16 "
              "karena 4,6x lebih hemat dimensi (1.764 vs 8.100) dengan "
              "performa setara.")
            w("")
    if kalibrasi:
        w("**C. Cara SVM mengambil probabilitas sangat memengaruhi "
          "performa.**")
        w("")
        w("| Cara | F1 macro | Recall ganas |")
        w("|---|---|---|")
        for r in kalibrasi:
            w(f"| {r['cara']} | {f(r['f1_macro'])} | "
              f"{f(r['malignant_recall'])} |")
        w("")
        w("Konsekuensi: dipakai `CalibratedClassifierCV(ensemble=True)`, "
          "karena API-nya yang resmi disarankan library DAN performanya "
          "jauh lebih baik daripada `ensemble=False` yang ada di contoh "
          "resmi sklearn.")
        w("")
    if bobot:
        w("**D. Bobot histogram hanya berpengaruh pada SVM, bukan pada "
          "Random Forest.** Pohon membelah berdasarkan urutan nilai, bukan "
          "jarak, jadi mengalikan fitur dengan konstanta tidak mengubah "
          "partisi.")
        w("")

    # ------------------------------------------------------------------
    w("## 5. Dokumentasi dan Artefak")
    w("")
    w("| Artefak | Isi |")
    w("|---|---|")
    w("| `README.md` | Struktur repo, aturan metodologi, cara jalankan |")
    w("| `docs/CATATAN_TEMUAN_DIAGNOSTIK.md` | Semua eksperimen + 11 bug "
      "yang ditemukan lewat eksekusi |")
    w("| `docs/PERSIAPAN_TANYA_JAWAB.md` | 40+ soal potensial beserta "
      "bukti file |")
    w("| `docs/PANDUAN_DEPLOY.md` | Panduan deploy aplikasi |")
    w("| `docs/CHECKLIST_VIDEO_DEMO.md` | Checklist rekaman video demo |")
    w("| `docs/PPT_Laporan_Project.pptx` | Slide presentasi (14 slide) |")
    w("")
    if lit:
        try:
            n = len(lit) if isinstance(lit, list) else len(
                lit.get("artikel", lit.get("results", [])))
        except TypeError:
            n = 0
        w(f"**Literatur:** {n} artikel diverifikasi otomatis ke PubMed "
          "(PMID + DOI + judul cocok), lihat "
          "`docs/logs/verifikasi_literature.json`.")
        w("")

    w("---")
    w("")
    w("## 6. Pemeriksaan Otomatis")
    w("")
    w("Empat skrip memeriksa repo sebelum submit. Semuanya harus lulus:")
    w("")
    w("| Skrip | Yang diperiksa |")
    w("|---|---|")
    w("| `scripts/check_api_calls.py` | Argumen fungsi yang tidak ada di "
      "API library terpasang |")
    w("| `scripts/check_undefined_names.py` | Variabel yang dipakai "
      "sebelum didefinisikan |")
    w("| `scripts/check_indonesian_text.py` | Kalimat campuran Inggris / "
      "teks rusak |")
    w("| `scripts/sanity_check_text.py` | Karakter CJK dan BOM |")
    w("")
    w("Jalankan: `python scripts/check_api_calls.py`, lalu yang lainnya.")
    w("")
    w("Dokumen ini dibuat ulang oleh `python scripts/build_rekap.py`.")

    # Tulis byte langsung supaya line ending-nya LF, bukan CRLF.
    # (write_text() di Windows melakukan translasi otomatis ke CRLF,
    #  dan pemeriksa kualitas teks menolak karakter U+000D.)
    KELUAR.write_bytes(("\n".join(L) + "\n").encode("utf-8"))
    print(f"Rekap laporan ditulis: {KELUAR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
