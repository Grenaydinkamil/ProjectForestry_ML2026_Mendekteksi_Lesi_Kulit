"""
Cek apakah versi paket yang di-pin punya wheel untuk Linux + Python 3.13.

Streamlit Community Cloud berjalan di Linux. Kalau salah satu paket hanya
punya wheel untuk Windows (atau hanya untuk Python 3.12), deploy akan
gagal dengan "No matching distribution found".

Skrip ini membaca app/requirements.txt, lalu untuk setiap paket
memeriksa daftar file dari PyPI dan mencari wheel yang cocok:
  - manylinux (atau linux) untuk x86_64
  - tag cp313 (Python 3.13) atau abi3 / py3 (universal)

Jalankan:
    python scripts/cek_wheel_linux.py
"""

from __future__ import annotations

import json
import pathlib
import re
import sys
import urllib.request

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
REQ = REPO_ROOT / "app" / "requirements.txt"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_req(path: pathlib.Path) -> list[tuple[str, str]]:
    """Baca requirements.txt -> [(nama, versi)]."""
    hasil = []
    for baris in path.read_text(encoding="utf-8").splitlines():
        baris = baris.split("#")[0].strip()
        if not baris:
            continue
        m = re.match(r"^([A-Za-z0-9._-]+)\s*==\s*([0-9][0-9A-Za-z.\-]*)$", baris)
        if m:
            hasil.append((m.group(1), m.group(2)))
    return hasil


def cek(nama: str, versi: str) -> tuple[bool, str]:
    """True kalau ada wheel Linux cp313 / universal untuk versi tsb."""
    url = f"https://pypi.org/pypi/{nama}/{versi}/json"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = json.loads(r.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        return False, f"GAGAL ambil data: {exc}"

    nama_file = []
    ada_sdist = False
    for f in data.get("urls", []):
        fn = f.get("filename", "")
        if fn.endswith(".tar.gz"):
            # sdist = sumber paket; pip bisa membangunnya sendiri.
            # Untuk paket pure-Python ini cara install yang normal.
            ada_sdist = True
            continue
        if not fn.endswith(".whl"):
            continue
        if "linux" not in fn:
            continue
        if "x86_64" not in fn and "none-any" not in fn:
            continue
        # cp313 = Python 3.13; abi3 = binary kompatibel ke depan;
        # py3-none-any = wheel universal pure-Python
        if ("cp313" in fn or "abi3" in fn or "none-any" in fn
                or re.search(r"-py3\.py3-|py2\.py3-", fn)):
            nama_file.append(fn)

    if nama_file:
        return True, nama_file[0]
    if ada_sdist:
        return True, "sdist (pure-python, pip build sendiri)"
    return False, "tidak ada wheel linux cp313/abi3/universal"


def main() -> int:
    paket = parse_req(REQ)
    print(f"Sumber: {REQ.relative_to(REPO_ROOT)}")
    print(f"{'Paket':<18}{'Versi':<12}{'Status':<10} Wheel")
    print("-" * 92)
    gagal = 0
    for nama, versi in paket:
        ok, info = cek(nama, versi)
        if not ok:
            gagal += 1
        print(f"{nama:<18}{versi:<12}{'OK' if ok else 'GAGAL':<10} {info[:50]}")
    print("-" * 92)
    print(f"Paket tanpa wheel Linux cp313: {gagal}")
    return 1 if gagal else 0


if __name__ == "__main__":
    sys.exit(main())
