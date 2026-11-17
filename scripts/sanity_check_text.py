"""
Memeriksa dan memperbaiki kualitas teks pada file sumber Python.

Dua masalah yang pernah terjadi saat menulis kode dengan bantuan AI:

1. KARAKTER CJK (Tionghoa/Jepang) yang menyelinap ke dalam komentar.
   Karakter ini tidak pernah muncul dalam kode Indonesia yang benar.
   Kalau ada, berarti ada glitch saat penulisan, dan teksnya harus
   diganti dengan kalimat Indonesia yang benar.

2. BOM (Byte Order Mark) UTF-8 di awal file.
   BOM menyebabkan Python 3 menganggap baris pertama berbeda, dan
   beberapa tooling (termasuk nbformat) bisa tersandung.

Tabel pasangan teks yang perlu diganti disimpan di file terpisah
(scripts/text_replacements.json) supaya skrip ini tidak pernah mengubah
tabelnya sendiri.

Jalankan:  python scripts/sanity_check_text.py
Keluar dengan kode 0 kalau semua file bersih.
"""

from __future__ import annotations

import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
REPLACEMENTS_FILE = pathlib.Path(__file__).resolve().parent / "text_replacements.json"

# Rentang karakter yang boleh: ASCII, Latin-1, Latin Ext-A, panah,
# simbol matematika, tanda centang/silang
ALLOWED_RANGES = [
    (0x20, 0x7E),
    (0xA0, 0x24F),
    (0x2013, 0x2014),
    (0x2018, 0x2019),
    (0x201C, 0x201D),
    (0x2190, 0x21FF),
    (0x2200, 0x22FF),
    (0x2713, 0x2717),
    (0x2500, 0x257F),  # karakter gambar kotak untuk tabel ASCII
    (0x0394, 0x0394),  # Delta untuk tabel perbandingan
]


def load_replacements() -> dict[str, str]:
    """Memuat tabel penggantian teks dari file JSON terpisah."""
    if not REPLACEMENTS_FILE.exists():
        return {}
    with open(REPLACEMENTS_FILE, encoding="utf-8") as f:
        return json.load(f)


def is_allowed(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in ALLOWED_RANGES)


def check_file(path: pathlib.Path) -> list[str]:
    """Mengembalikan daftar masalah yang ditemukan pada satu file."""
    problems = []
    raw = path.read_bytes()

    if raw[:3] == b"\xef\xbb\xbf":
        problems.append("BOM UTF-8 ditemukan (harus dihapus)")

    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        problems.append(f"bukan UTF-8 yang valid: {exc}")
        return problems

    for i, line in enumerate(text.split("\n"), 1):
        bad = [ch for ch in line if not is_allowed(ch)]
        if bad:
            uniq = sorted({ch for ch in bad})
            problems.append(
                f"baris {i}: karakter tak diizinkan "
                f"{[f'U+{ord(c):04X}' for c in uniq]}"
            )
    return problems


def fix_file(path: pathlib.Path, replacements: dict[str, str]) -> bool:
    """Memperbaiki teks CJK dan menghapus BOM. True jika ada perubahan."""
    raw = path.read_bytes()
    original = raw

    if raw[:3] == b"\xef\xbb\xbf":
        raw = raw[3:]

    text = raw.decode("utf-8", errors="replace")
    for wrong, right in replacements.items():
        if wrong in text:
            text = text.replace(wrong, right)

    new_bytes = text.encode("utf-8")
    if new_bytes != original:
        path.write_bytes(new_bytes)
    return True


def main() -> int:
    replacements = load_replacements()

    targets: list[pathlib.Path] = []
    for pattern in ("src/*.py", "scripts/*.py", "scripts/**/*.py",
                    "app/*.py", "notebooks/*.py",
                    # Dokumentasi dan generator PPT juga bisa tersesat
                    # karakter CJK, jadi ikut diperiksa.
                    "*.md", "docs/*.md", "scripts/*.js"):
        targets.extend(REPO_ROOT.glob(pattern))
    targets = sorted(set(targets))

    for path in targets:
        # SKIP file tabel itu sendiri: isinya MEMANG mengandung karakter
        # CJK (that's the point), jadi akan selalu dilaporkan sebagai masalah.
        if path.resolve() == REPLACEMENTS_FILE.resolve():
            continue
        fix_file(path, replacements)

    problems = 0
    for path in targets:
        if path.resolve() == REPLACEMENTS_FILE.resolve():
            continue
        issues = check_file(path)
        if issues:
            problems += 1
            print(f"[X] {path.relative_to(REPO_ROOT)}")
            for issue in issues:
                print(f"     {issue}")

    print(f"\nFile diperiksa: {len(targets)} | "
          f"masih bermasalah: {problems}")
    if problems:
        print("Perbaiki manual baris yang ditandai, lalu jalankan lagi.")
    else:
        print("Semua file bersih.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
