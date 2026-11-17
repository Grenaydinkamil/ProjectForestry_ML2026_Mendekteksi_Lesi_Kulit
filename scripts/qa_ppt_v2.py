"""
QA tata letak PPT v2 tanpa LibreOffice.

Karena mesin ini tidak punya LibreOffice/pdftoppm, render visual tidak
bisa dilakukan. Skrip ini memeriksa hal yang paling sering salah:

  1. Shape yang keluar dari batas slide.
  2. Kotak teks yang terlalu kecil untuk isinya (estimasi kasar).
  3. Slide yang tidak punya catatan pembicara.
  4. Slide yang seluruhnya teks (tanpa shape/chart/table).

Jalankan:
    python scripts/qa_ppt_v2.py
"""

from __future__ import annotations

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PPT = REPO_ROOT / "docs" / "PPT_Laporan_Project_v2.pptx"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pptx import Presentation  # noqa: E402
from pptx.util import Emu  # noqa: E402

prs = Presentation(PPT)
W = prs.slide_width / 914400
H = prs.slide_height / 914400

print(f"Ukuran slide: {W:.2f} x {H:.2f} inci")
print(f"Jumlah slide: {len(prs.slides)}")
print()

masalah = 0

for i, slide in enumerate(prs.slides, 1):
    n_shape = len(slide.shapes)
    n_chart = sum(1 for s in slide.shapes if s.has_chart)
    n_table = sum(1 for s in slide.shapes if s.has_table)
    n_teks = sum(1 for s in slide.shapes
                 if s.has_text_frame and s.text_frame.text.strip())
    punya_notes = bool(slide.has_notes_slide and
                       slide.notes_slide.notes_text_frame.text.strip())

    # 1. Batas slide
    for sh in slide.shapes:
        x = sh.left / 914400 if sh.left is not None else 0
        y = sh.top / 914400 if sh.top is not None else 0
        w = sh.width / 914400 if sh.width is not None else 0
        h = sh.height / 914400 if sh.height is not None else 0
        if x < -0.02 or y < -0.02 or x + w > W + 0.02 or y + h > H + 0.02:
            print(f"[X] Slide {i}: shape '{sh.name}' keluar dari slide "
                  f"({x:.2f},{y:.2f} {w:.2f}x{h:.2f})")
            masalah += 1

    # 2. Estimasi overflow teks
    for sh in slide.shapes:
        if not sh.has_text_frame or not sh.text_frame.text.strip():
            continue
        teks = sh.text_frame.text
        w = sh.width / 914400 if sh.width else 1
        h = sh.height / 914400 if sh.height else 1
        # Ambil ukuran font terbesar di shape ini
        ukuran = 12
        for p in sh.text_frame.paragraphs:
            for r in p.runs:
                if r.font.size:
                    ukuran = max(ukuran, r.font.size.pt)
        baris = teks.count("\n") + 1
        # Approx: 1 inci lebar ~ 9 karakter per 10pt
        kap = max(1, int(w * (10 / ukuran) * 9))
        estimasi = sum(max(1, -(-len(b) // kap)) for b in teks.split("\n"))
        tinggi_per_baris = ukuran / 72 * 1.25
        butuh = estimasi * tinggi_per_baris
        if butuh > h * 1.6 and len(teks) > 40:
            print(f"[?] Slide {i}: '{sh.name}' mungkin overflow "
                  f"(perkiraan {butuh:.2f}\" vs kotak {h:.2f}\", "
                  f"font {ukuran:.0f}pt) :: {teks[:55]!r}")
            masalah += 1

    # 3. Notes
    if not punya_notes:
        print(f"[X] Slide {i}: tidak ada catatan pembicara")
        masalah += 1

    # 4. Visual element
    if n_chart == 0 and n_table == 0 and n_shape <= n_teks + 1:
        print(f"[?] Slide {i}: hanya teks ({n_shape} shape)")
        masalah += 1

    status = "OK" if (n_chart or n_table or n_shape > n_teks + 1) else "TEKS"
    print(f"  Slide {i:>2}: {n_shape:>2} shape | {n_teks:>2} teks | "
          f"{n_chart} chart | {n_table} tabel | notes {'ya' if punya_notes else 'TIDAK'} | {status}")

print()
print(f"Total yang perlu ditinjau: {masalah}")
sys.exit(0)
