"""
scripts/dump_literature_json.py
==============================
Mengubah isi `scripts/literature.py` menjadi JSON ringkas untuk dipakai
generator PPT.

Kenapa perlu:
    Slide studi literatur butuh ringkasan metode, hasil, dan keterkaitan
    tiap artikel. Kalau teks itu ditulis ulang di dalam generator PPT,
    ada risiko isinya berbeda dari modul yang sudah diverifikasi ke
    PubMed. Dengan cara ini, sumbernya tetap satu berkas.

Yang ditulis:
    docs/logs/literature_summary.json

Jalankan:
    python scripts/dump_literature_json.py
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from literature import VERIFIED_ARTICLES  # noqa: E402


def kalimat_pertama(teks: str) -> str:
    """Ambil kalimat pertama sebagai versi ringkas."""
    teks = " ".join(teks.split())
    for pemisah in (". ", "; "):
        if pemisah in teks:
            return teks.split(pemisah)[0].rstrip(".") + "."
    return teks.rstrip(".") + "."


def main() -> int:
    keluar = []
    for a in VERIFIED_ARTICLES:
        keluar.append(
            {
                "key": a["key"],
                "pmid": a.get("pmid"),
                "authors": a["authors"],
                "year": a["year"],
                "title": " ".join(a["title"].split()),
                "journal": a["journal"],
                "volume_pages": a.get("volume_pages"),
                "doi": a["doi"],
                "metode": kalimat_pertama(a["metode"]),
                "metrik": kalimat_pertama(a["metrik"]),
                "hasil": kalimat_pertama(a["hasil_utama"]),
                "hasil_panjang": " ".join(a["hasil_utama"].split()),
                "dataset": kalimat_pertama(a["dataset"]),
                "keterkaitan": kalimat_pertama(a["perbedaan"]),
                "keterkaitan_panjang": " ".join(a["perbedaan"].split()),
            }
        )

    tujuan = REPO / "docs" / "logs" / "literature_summary.json"
    tujuan.write_text(
        json.dumps(keluar, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"Ringkasan literatur ditulis: {tujuan} ({len(keluar)} artikel)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
