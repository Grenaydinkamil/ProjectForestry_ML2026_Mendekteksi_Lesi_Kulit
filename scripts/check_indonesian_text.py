"""
Cari kalimat campuran Inggris / teks rusak di dalam komentar dan markdown.

Tujuan: proyek ini ditulis dalam Bahasa Indonesia. Kadang ada kata Inggris
yang "nyelip" di dalam kalimat (mis. "ia bisa learns representasi"), atau
ada fragmen yang hilang (mis. "cirinya tidak(None)secara"). Semua itu
terdeteksi otomatis di sini.

Cara kerja:
  1. Ambil hanya baris komentar (# ...) dari file sumber.
  2. Cari kata dari daftar "kirmau" (bahasa Inggris yang jelas tidak boleh
     dipakai sebagai kata Indonesia) serta pola keretakan.
  3. Tampilkan baris yang perlu ditinjau manusia.

Kata teknis yang sah (model, data, batch, epoch, ...) dikecualikan.

Jalankan:  python scripts/check_indonesian_text.py
"""

from __future__ import annotations

import pathlib
import re
import sys

# Console Windows sering memakai cp1252, yang tidak bisa mencetak
# karakter CJK. Kita paksa utf-8 dengan errors="replace" supaya
# pemeriksa ini tidak ikut crash saat menemukan masalah.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Kata bahasa Inggris yang jelas tidak boleh muncul sebagai kata Indonesia.
#
# Daftar ini SENGAJA dibuat konservatif: kalau terlalu banyak kata, pemeriksa
# jadi GMPUNG (banyak false positive dari nama fungsi seperti
# `predict_with_proba` atau `build_svm`) dan kegunaannya hilang.
# Kata seperti "mean" atau "with" sengaja TIDAK dimasukkan karena sering
# muncul sebagai bagian nama variabel/fungsi.
KIRMAU = {
    # Kata yang hampir selalu salah di dalam kalimat Indonesia
    "learns", "motivates", "harfiah", "andil", "however", "therefore",
    "thus", "hence", "makes", "made", "gives", "shown", "built", "tried",
    "known", "means", "there", "their", "which", "when", "where", "while",
    "into", "about", "after", "before", "because", "would", "should",
    "cannot", "been", "were", "will", "have", "has",
}

# Pola keretakan yang sering muncul saat menulis cepat
POLA_RUSAK = [
    re.compile(r"\(None\)"),
    re.compile(r"\bNone\b"),
    re.compile(r"[a-z]\)[A-Za-z]"),
    # Karakter CJK (Tionghoa/Jepang/Korea) tidak pernah muncul dalam
    # teks Indonesia yang benar.
    re.compile(r"[\u2e80-\u9fff\uf900-\ufaff\uff00-\uffef]"),
]


def baris_komentar(path: pathlib.Path) -> list[tuple[int, str]]:
    """
    Ambil baris yang merupakan komentar.

    Untuk file .md, semua baris dianggap teks (dokumenasi bukan kode),
    tapi baris di dalam blok kode dan tabel markdown tetap diperiksa
    karena isinya juga bisa salah ketik.
    """
    hasil = []
    in_code_block = False
    for i, line in enumerate(
            path.read_text(encoding="utf-8-sig").split("\n"), 1):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        if path.suffix == ".md":
            if stripped:
                hasil.append((i, stripped.lstrip("#").strip()))
        elif stripped.startswith("#"):
            isi = stripped.lstrip("#").strip()
            if isi:
                hasil.append((i, isi))
    return hasil


def main() -> int:
    pola = ("src/*.py", "app/*.py", "scripts/*.py",
            "scripts/notebook_src/*.py",
            "*.md", "docs/*.md")
    total = 0
    file_diperiksa = 0

    for pola_file in pola:
        for path in sorted(REPO_ROOT.glob(pola_file)):
            file_diperiksa += 1
            # Untuk markdown, hanya pola keretakan yang diperiksa.
            # Dokumentasi memang sah memuat kutipan bahasa Inggris,
            # pesan error library, dan nama fungsi -- semuanya bukan
            # kesalahan.
            pola_terpakai = POLA_RUSAK if path.suffix == ".md" else (
                KIRMAU, POLA_RUSAK)
            for nomor, isi in baris_komentar(path):
                words = set(re.findall(r"[A-Za-z']+", isi.lower()))
                kirmau = sorted(words & KIRMAU) if path.suffix != ".md" else []
                rusak = [p.pattern for p in POLA_RUSAK if p.search(isi)]
                if kirmau or rusak:
                    print(f"[?] {path.relative_to(REPO_ROOT)}:{nomor} "
                          f"perlu ditinjau: {kirmau or rusak}")
                    print(f"    {isi[:110]}")
                    total += 1

    print(f"\nFile diperiksa: {file_diperiksa} | baris perlu ditinjau: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
