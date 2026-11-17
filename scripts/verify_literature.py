"""
scripts/verify_literature.py
============================
Memverifikasi daftar artikel jurnal terhadap NCBI PubMed.

Tujuan: memastikan TIDAK ADA sitasi yang dikarang. Untuk setiap artikel
dari `scripts/literature.py`, skrip ini mengambil data langsung dari
indeks PubMed lalu mencocokkan judul, tahun, jurnal, dan DOI.

Jalankan:
    python scripts/verify_literature.py

Keluar dengan kode 0 kalau semua artikel terverifikasi.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from literature import VERIFIED_ARTICLES  # noqa: E402

UA = {"User-Agent": "ml-lesi-kulit-project/1.0"}
PUBMED_URL = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
              "?db=pubmed&retmode=xml&id={pmid}")


def fetch_pubmed(pmid: str) -> dict | None:
    """Mengambil data satu artikel dari PubMed."""
    url = PUBMED_URL.format(pmid=pmid)
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=45) as resp:
            xml = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}

    if "<PubmedArticle>" not in xml:
        return None

    def grab(tag: str) -> str:
        seg = xml.split(f"<{tag}>")
        if len(seg) < 2:
            return ""
        return " ".join(seg[1].split(f"</{tag}>")[0].split())

    # Tahun terbit. PubMed menaruh <Year> di dalam <PubDate>, jadi kita
    # ambil year's tag spesifik, bukan seluruh blok PubDate.
    tahun = ""
    for tag in ("ArticleDate", "PubDate"):
        if f"<{tag}>" in xml:
            blok = xml.split(f"<{tag}>")[1].split(f"</{tag}>")[0]
            if "<Year>" in blok:
                tahun = blok.split("<Year>")[1].split("</Year>")[0].strip()
                if tahun:
                    break
    if not tahun:
        tahun = grab("Year") or grab("MedlineDate")[:4]

    doi = ""
    for chunk in xml.split('<ArticleId IdType="doi">')[1:]:
        doi = chunk.split("</ArticleId>")[0].strip()
        break

    return {
        "pmid": grab("PMID"),
        "title": grab("ArticleTitle"),
        "journal": grab("Title"),
        "year": tahun,
        "doi": doi,
    }


def _normalisasi(teks: str) -> str:
    """
    Membersihkan teks sebelum dibandingkan.

    Dua hal perlu ditangani:
      1. Entitas HTML di XML PubMed, misalnya "&amp;" dan "&lt;"
      2. Judul yang dipotong oleh PubMed dengan "..."
    """
    import html
    teks = html.unescape(teks)
    teks = teks.replace(chr(8230), "...").replace("...", "")
    teks = " ".join(teks.split())
    return teks.lower().rstrip(". ")


def main() -> int:
    print("=" * 78)
    print("VERIFIKASI SITASI JURNAL MELALUI PUBMED")
    print("=" * 78)
    print(f"Artikel yang diverifikasi: {len(VERIFIED_ARTICLES)}\n")

    results = []
    semua_cocok = True

    for art in VERIFIED_ARTICLES:
        data = fetch_pubmed(art["pmid"])
        print(f"[{art['key']}] {art['authors'].split(',')[0]} dkk. ({art['year']})")
        print(f"  Judul lokal : {art['title'][:88]}")

        if data is None:
            print("  [X] PMID tidak ditemukan di PubMed")
            semua_cocok = False
            results.append({"key": art["key"], "found": False})
            continue
        if "error" in data:
            print(f"  [X] Gagal diakses: {data['error']}")
            semua_cocok = False
            results.append({"key": art["key"], "found": False,
                            "error": data["error"]})
            continue

        judul_cocok = (_normalisasi(data["title"])
                        == _normalisasi(art["title"]))
        doi_cocok = data["doi"].lower() == art["doi"].lower()
        # PubMed mengembalikan tanggal lengkap seperti "2018 Aug 14" atau
        # "2020 Oct", jadi tahun dicek dengan cara yang sama.
        tahun_cocok = str(art["year"]) in str(data["year"])

        print(f"  Judul PubMed: {data['title'][:88]}")
        print(f"  Jurnal      : {data['journal']}")
        print(f"  DOI PubMed  : {data['doi']}")
        print(f"  DOI lokal   : {art['doi']}")
        status = ("[OK] Judul, tahun, dan DOI COCOK" if
                  (judul_cocok and doi_cocok and tahun_cocok) else "[!] Ada yang TIDAK COCOK")
        print(f"  -> {status}")
        if not (judul_cocok and doi_cocok and tahun_cocok):
            semua_cocok = False
            print(f"     judul_cocok={judul_cocok} doi_cocok={doi_cocok} "
                  f"tahun_cocok={tahun_cocok}")
            print(f"     judul_lokal_normal: {_normalisasi(art['title'])[:95]}")
            print(f"     judul_pubmed_norm : {_normalisasi(data['title'])[:95]}")

        results.append({
            "key": art["key"], "found": True,
            "title_match": judul_cocok, "doi_match": doi_cocok,
            "year_match": tahun_cocok,
            "pubmed": data,
        })
        print()
        time.sleep(0.4)

    out = Path(__file__).resolve().parent.parent / "docs" / "logs"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "verifikasi_literature.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("=" * 78)
    if semua_cocok:
        print("HASIL: SEMUA ARTIKEL TERVERIFIKASI di PubMed.")
    else:
        print("HASIL: Ada artikel yang tidak cocok. Periksa detail di atas.")
    print(f"Detail disimpan: {out / 'verifikasi_literature.json'}")
    print("=" * 78)
    return 0 if semua_cocok else 1


if __name__ == "__main__":
    sys.exit(main())
