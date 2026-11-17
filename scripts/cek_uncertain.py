"""
Cek perilaku `is_uncertain()` dengan ambang yang dipakai aplikasi.

Masalah yang ditemukan: ambang "ganas" untuk MobileNetV2 sekarang 0,11
(bukan 0,25 seperti sebelumnya).>Fungsi `is_uncertain` memakai band
lebar 0,15 di sekitar ambang. Dengan ambang 0,11, seluruh rentang risiko
jinak (0 sampai 0,11) jatuh DI DALAM band itu, jadi hampir semua foto
lesi jinak akan diberi label "belum meyakinkan" -- kotak hijau "Tidak
ditemukan ciri ganas" nyaris tidak pernah muncul.

Skrip ini mengukur perilaku lama dan baru pada contoh gambar nyata,
lalu menampilkan hasilnya supaya perbaikannya bisa dinilai dengan angka.

Jalankan:
    python scripts/cek_uncertain.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src import predictor as PR

CONTOH = REPO_ROOT / "app" / "contoh"


def is_uncertain_lama(result: dict, band: float = 0.15) -> bool:
    """Versi lama: band absolut 0,15."""
    risk = result["risk_malignant"]
    threshold = result["threshold"]
    top_prob = float(result["probabilities"].max())
    return abs(risk - threshold) < band or top_prob < 0.60


def is_uncertain_baru(result: dict, band: float | None = None) -> bool:
    """Versi baru -- memakai implementasi yang benar-benar dipakai aplikasi."""
    return PR.is_uncertain(result, band)


def main() -> int:
    predictor = PR.load_predictor()
    thr = predictor["threshold"]
    print(f"Model: {predictor['model_name']} | ambang: {thr:.3f}")
    print(f"Band lama: 0,150 (absolut) | Band baru: "
          f"{max(0.02, 0.30 * thr):.3f} (mengikuti skala ambang)\n")

    print(f"{'gambar':<28}{'kelas':>8}{'risiko':>9}{'top':>8}"
          f"{'lama':>7}{'baru':>7}")
    print("-" * 70)

    n_lama = n_baru = 0
    total = 0
    for path in sorted(CONTOH.glob("*")):
        img = PR.Image.open(path).convert("RGB")
        result = PR.predict(predictor, img)
        lama = is_uncertain_lama(result)
        baru = is_uncertain_baru(result)
        n_lama += lama
        n_baru += baru
        total += 1
        print(f"{path.name:<28}{result['predicted_cls']:>8}"
              f"{result['risk_malignant']:>9.3f}"
              f"{result['probabilities'].max():>8.3f}"
              f"{'ABU' if lama else 'pasti':>7}"
              f"{'ABU' if baru else 'pasti':>7}")

    print(f"\nDari {total} contoh: {n_lama} ditandai 'tidak yakin' "
          f"oleh logika LAMA, {n_baru} oleh logika BARU.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
