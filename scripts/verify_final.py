"""
scripts/verify_final.py
=======================
Menjalankan SEMUA pemeriksaan proyek dalam satu perintah.

Pemeriksaan yang dilakukan:

  1. `check_api_calls.py`        -> argumen fungsi yang tidak ada di API
  2. `check_undefined_names.py`  -> variabel dipakai sebelum didefinisikan
  3. `sanity_check_text.py`      -> karakter CJK dan BOM
  4. `check_indonesian_text.py`  -> kalimat campuran Inggris / teks rusak
  5. `verify_literature.py`      -> sitasi diverifikasi ke PubMed
  6. `test_app_content.py`       -> isi aplikasi Streamlit (AppTest)
  7. `cek_uncertain.py`          -> perilaku "tidak yakin" di aplikasi
  8. `build_ppt.js`              -> membangun ulang PPT dari log
  9. `build_rekap.py`            -> membangun ulang rekap laporan

Tambahan:
  7b. `audit_kontras.py`         -> rasio kontras WCAG tema
  7c. `cek_import_runtime.py`    -> semua import runtime ada di requirements

Jalankan:
    python scripts/verify_final.py
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def jalankan(nama: str, perintah: list[str]) -> bool:
    """Menjalankan satu pemeriksaan. True kalau berhasil."""
    print("-" * 70)
    print(f">> {nama}")
    print("-" * 70)
    try:
        hasil = subprocess.run(
            perintah, cwd=REPO_ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=1800,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"    [X] Gagal menjalankan: {exc}")
        return False

    keluaran = (hasil.stdout or "") + (hasil.stderr or "")
    baris = [b for b in keluaran.splitlines() if b.strip()]
    for b in baris[-6:]:
        print(f"    {b}")
    if hasil.returncode != 0:
        print(f"    [X] Kode keluar {hasil.returncode}")
        return False
    return True


def main() -> int:
    py = sys.executable
    node = "node"

    langkah = [
        ("1. Argumen fungsi yang tidak ada di API library",
         [py, str(SCRIPTS / "check_api_calls.py")]),
        ("2. Variabel yang dipakai sebelum didefinisikan",
         [py, str(SCRIPTS / "check_undefined_names.py")]),
        ("3. Karakter CJK dan BOM di semua file sumber",
         [py, str(SCRIPTS / "sanity_check_text.py")]),
        ("4. Kalimat campuran Inggris / teks rusak",
         [py, str(SCRIPTS / "check_indonesian_text.py")]),
        ("5. Sitasi diverifikasi ke PubMed",
         [py, str(SCRIPTS / "verify_literature.py")]),
        ("6. Isi aplikasi Streamlit (AppTest resmi)",
         [py, str(SCRIPTS / "test_app_content.py")]),
        ("7. Perilaku 'tidak yakin' di aplikasi",
         [py, str(SCRIPTS / "cek_uncertain.py")]),
        ("7b. Audit kontras WCAG (antislop R-25)",
         [py, str(SCRIPTS / "audit_kontras.py")]),
        ("7c. Paket runtime tercakup app/requirements.txt",
         [py, str(SCRIPTS / "cek_import_runtime.py")]),
        ("8. Bangun ulang PPT versi 1 (14 slide)",
         [node, str(SCRIPTS / "build_ppt.js")]),
        ("9. Bangun ulang PPT versi 2 (12 slide)",
         [node, str(SCRIPTS / "build_ppt_v2.js")]),
        ("10. QA tata letak PPT versi 2",
         [py, str(SCRIPTS / "qa_ppt_v2.py")]),
        ("11. Bangun ulang rekap laporan",
         [py, str(SCRIPTS / "build_rekap.py")]),
    ]

    print("=" * 70)
    print(" VERIFIKASI AKHIR PROYEK ML2026")
    print("=" * 70)

    gagal = []
    for nama, perintah in langkah:
        if not jalankan(nama, perintah):
            gagal.append(nama)

    print()
    print("=" * 70)
    if gagal:
        print(f" HASIL: {len(gagal)} pemeriksaan GAGAL:")
        for g in gagal:
            print(f"   - {g}")
        print("=" * 70)
        return 1
    print(" HASIL: SEMUA PEMERIKSAAN LULUS.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
