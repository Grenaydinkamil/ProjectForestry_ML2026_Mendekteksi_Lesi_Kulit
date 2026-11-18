"""
Audit kontras WCAG untuk palet warna aplikasi Streamlit.

Membaca warna dari .streamlit/config.toml dan memastikan warna aksen
(judul, metrik, grafik) di app/app.py masih memakai nilai yang diaudit.
Antislop R-25 mensyaratkan minimal 4.5:1 untuk teks normal, 3:1 untuk
teks besar dan elemen grafis.

Jalankan:
    python scripts/audit_kontras.py
"""

from __future__ import annotations

import pathlib
import re
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
APP = REPO_ROOT / "app" / "app.py"
CONFIG = REPO_ROOT / ".streamlit" / "config.toml"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Nilai yang diminta brief "Teal & Warm Alabaster". Kalau config berubah,
# audit harus ikut berubah -- bukan diam-diam menguji palet lama.
TOKOHARAP = {
    "primaryColor": "#0D9488",
    "backgroundColor": "#FAF9F6",
    "secondaryBackgroundColor": "#FFFFFF",
    "textColor": "#1C1917",
}
AKSEN_APP = ["#0F766E", "#0D9488", "#E11D48"]


def rgb(h):
    h = h.lstrip("#")
    return [int(h[i:i+2], 16) / 255 for i in (0, 2, 4)]


def lum(c):
    if isinstance(c, str):
        c = rgb(c)
    c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4
         for x in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def ratio(a, b):
    la, lb = lum(a), lum(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def blend(fg: str, bg: str, alpha: float) -> tuple[float, float, float]:
    """
    Warna efektif saat fg diberi transparansi alpha di atas bg.

    Dihitung tanpa pembulatan ke 8-bit: browser melakukan compositing
    alpha dalam floating point, dan pembulatan lebih dulu bisa menggeser
    rasio kontras di ambang 4,5:1.
    """
    f, b = rgb(fg), rgb(bg)
    return tuple(alpha * f[i] + (1 - alpha) * b[i] for i in range(3))


def hexdari(c) -> str:
    if isinstance(c, str):
        return c
    return "#" + "".join(f"{round(x * 255):02X}" for x in c)


def baca_toml(kunci: str) -> str | None:
    if not CONFIG.exists():
        return None
    teks = CONFIG.read_text(encoding="utf-8")
    m = re.search(rf'{kunci}\s*=\s*"(#[0-9A-Fa-f]{{6}})"', teks)
    return m.group(1) if m else None


def main() -> int:
    print(f"Sumber: {APP.relative_to(REPO_ROOT)} + "
          f"{CONFIG.relative_to(REPO_ROOT)}")

    gagal = 0
    print("\n[1] Nilai theme sesuai brief")
    for kunci, harap in TOKOHARAP.items():
        ada = baca_toml(kunci)
        ok = ada == harap
        if not ok:
            gagal += 1
        print(f"    {'[OK]' if ok else '[X]'} {kunci}: {ada} "
              f"(harus {harap})")

    print("\n[2] Warna aksen dipakai di app.py")
    app_teks = APP.read_text(encoding="utf-8")
    for warna in AKSEN_APP:
        ok = warna in app_teks
        if not ok:
            gagal += 1
        print(f"    {'[OK]' if ok else '[X]'} {warna}")

    tinta = baca_toml("textColor") or "#1C1917"
    kertas = baca_toml("backgroundColor") or "#FAF9F6"
    kartu = baca_toml("secondaryBackgroundColor") or "#FFFFFF"

    pasangan = [
        ("teks utama / kertas", tinta, kertas, 4.5),
        ("caption 60% / kertas", blend(tinta, kertas, 0.6), kertas, 4.5),
        ("teks utama / kartu", tinta, kartu, 4.5),
        ("caption 60% / kartu", blend(tinta, kartu, 0.6), kartu, 4.5),
        ("judul teal / kertas", "#0F766E", kertas, 3.0),
        ("nilai metrik teal / kartu", "#0D9488", kartu, 3.0),
        ("batang jinak / kartu", "#0D9488", kartu, 3.0),
        ("batang ganas / kartu", "#E11D48", kartu, 3.0),
        ("teks putih / lencana ganas", "#FFFFFF", "#E11D48", 4.5),
    ]

    print("\n[3] Rasio kontras")
    print(f"{'Pasangan':<30}{'Fg':>9}{'Bg':>9}{'Rasio':>8}{'Min':>7}  Status")
    print("-" * 74)
    for nama, fg, bg, minimum in pasangan:
        r = ratio(fg, bg)
        ok = r >= minimum
        if not ok:
            gagal += 1
        print(f"{nama:<30}{hexdari(fg):>9}{hexdari(bg):>9}{r:>8.2f}"
              f"{minimum:>7.1f}  {'OK' if ok else 'GAGAL'}")
    print("-" * 74)
    print(f"Gagal: {gagal}")
    return 1 if gagal else 0


if __name__ == "__main__":
    sys.exit(main())
