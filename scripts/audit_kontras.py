"""
Audit kontras WCAG untuk palet warna aplikasi Streamlit.

Membaca warna LANGSUNG dari app/app.py dan .streamlit/config.toml, lalu
menghitung rasio kontras pasangan foreground/background. Antislop R-25
mensyaratkan minimal 4.5:1 untuk teks normal dan 3:1 untuk teks besar.

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


def rgb(h):
    h = h.lstrip("#")
    return [int(h[i:i+2], 16) / 255 for i in (0, 2, 4)]


def lum(c):
    c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4
         for x in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def ratio(a, b):
    la, lb = lum(rgb(a)), lum(rgb(b))
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def baca_token(nama: str, default: str) -> str:
    """Ambil nilai CSS variable `--nama` dari app.py."""
    m = re.search(rf"--{re.escape(nama)}:\s*(#[0-9A-Fa-f]{{6}})", APP.read_text(encoding="utf-8"))
    return m.group(1) if m else default


def baca_toml(kunci: str, default: str) -> str:
    if not CONFIG.exists():
        return default
    m = re.search(rf'{kunci}\s*=\s*"(#[0-9A-Fa-f]{{6}})"', CONFIG.read_text(encoding="utf-8"))
    return m.group(1) if m else default


paper = baca_toml("backgroundColor", "#FBFAF7")
tinta = baca_toml("textColor", "#1A1D21")
lembut = baca_token("ink-soft", "#4A5058")
faint = baca_token("ink-faint", "#6A717A")
navy = baca_token("accent", "#2F5D7C")
merah = baca_token("danger", "#9E2A2B")
merah_muda = baca_token("danger-dim", "#F7E9E8")
hijau = baca_token("ok", "#2E6B4F")
hijau_muda = baca_token("ok-dim", "#E6F0EA")
kuning = baca_token("warn", "#8A5A15")
kuning_muda = baca_token("warn-dim", "#FBF0DC")

pasangan = [
    ("teks utama / paper", tinta, paper, 4.5),
    ("teks lembut / paper", lembut, paper, 4.5),
    ("teks faint / paper", faint, paper, 4.5),
    ("putih / navy (header)", "#FFFFFF", navy, 4.5),
    ("merah / bata muda", merah, merah_muda, 3.0),
    ("hijau / hijau muda", hijau, hijau_muda, 3.0),
    ("kuning / kuning muda", kuning, kuning_muda, 3.0),
]

print(f"Sumber: {APP.relative_to(REPO_ROOT)} + {CONFIG.relative_to(REPO_ROOT)}")
print(f"{'Pasangan':<28}{'Fg':>9}{'Bg':>9}{'Rasio':>8}{'Min':>7}  Status")
print("-" * 70)
gagal = 0
for nama, fg, bg, minimum in pasangan:
    r = ratio(fg, bg)
    ok = r >= minimum
    if not ok:
        gagal += 1
    print(f"{nama:<28}{fg:>9}{bg:>9}{r:>8.2f}{minimum:>7.1f}  {'OK' if ok else 'GAGAL'}")
print("-" * 70)
print(f"Gagal: {gagal}")
sys.exit(1 if gagal else 0)
