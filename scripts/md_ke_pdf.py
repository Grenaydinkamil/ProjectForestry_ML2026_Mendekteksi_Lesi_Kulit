"""
scripts/md_ke_pdf.py
====================
Ubah berkas markdown menjadi PDF bergaya rapi.

Kenapa skrip ini ada:
    Dua dokumen untuk pengumpulan (tabel studi literatur dan panduan
    deployment) perlu dikirim sebagai PDF. Menulis PDF langsung dari
    Python itu berat; markdown jauh lebih mudah dirawat, lalu dikonversi.

Cara pakai:
    python scripts/md_ke_pdf.py <masukan.md> <keluaran.pdf> [--landscape]
                                     [--col-widths 3,10,12,13,11,10,13,19,9]

Ketentuan:
    - Tidak menambah dependensi baru ke aplikasi. Paket yang dipakai
      (playwright) hanya untuk tooling, bukan untuk app/.
    - PDF memakai browser Edge yang sudah ada di mesin ini, jadi tidak
      perlu mengunduh Chromium.
    - Lebar kolom tabel diatur lewat --col-widths karena tabel sembilan
      kolom yang diratakan sama lebarnya jadi tidak terbaca.
"""

from __future__ import annotations

import argparse
import html
import pathlib
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# MARKDOWN -> HTML
# ---------------------------------------------------------------------------
# Cakupannya sengaja dibatasi pada elemen yang dipakai dua dokumen ini:
# heading, paragraf, daftar, tabel GFM, blok kode, kutip, garis, dan
# format sebaris.
# ---------------------------------------------------------------------------


def esc(teks: str) -> str:
    return html.escape(teks, quote=False)


def inline(teks: str) -> str:
    """Format sebaris: kode, tebal, miring, tautan, dan autolink."""
    # Autolink <https://...> diubah dulu jadi tautan biasa supaya tidak
    # dimakan browser sebagai tag HTML yang tidak dikenal.
    teks = re.sub(r"<(https?://[^\s>]+)>", r"[\1](\1)", teks)

    bagian = re.split(r"(`[^`]+`)", teks)
    hasil = []
    for bagian_item in bagian:
        if bagian_item.startswith("`") and bagian_item.endswith("`") and len(bagian_item) > 1:
            hasil.append(f"<code>{esc(bagian_item[1:-1])}</code>")
            continue
        t = esc(bagian_item)
        t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', t)
        t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
        t = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", t)
        t = re.sub(r"(?<![\w_])_([^_]+)_(?![\w_])", r"<em>\1</em>", t)
        hasil.append(t)
    return "".join(hasil)


def sel_baris(baris: str) -> list[str]:
    isi = baris.strip()
    if isi.startswith("|"):
        isi = isi[1:]
    if isi.endswith("|"):
        isi = isi[:-1]
    return [sel.strip() for sel in isi.split("|")]


def md_ke_html(md: str, col_widths: list[float] | None = None) -> str:
    baris = md.split("\n")
    keluar: list[str] = []
    i = 0
    while i < len(baris):
        baris_teks = baris[i]

        if baris_teks.startswith("```"):
            i += 1
            isi = []
            while i < len(baris) and not baris[i].startswith("```"):
                isi.append(baris[i])
                i += 1
            i += 1
            keluar.append(f"<pre><code>{esc(chr(10).join(isi))}</code></pre>")
            continue

        # Tabel GFM: baris header diikuti baris pembatas kolom.
        if baris_teks.strip().startswith("|") and i + 1 < len(baris) and re.match(
            r"^\s*\|[\s:|-]+\|?\s*$", baris[i + 1]
        ):
            header = sel_baris(baris_teks)
            i += 2
            isi_baris = []
            while i < len(baris) and baris[i].strip().startswith("|"):
                isi_baris.append(sel_baris(baris[i]))
                i += 1
            kolom = "".join(f"<th>{inline(h)}</th>" for h in header)
            baris_html = []
            for baris_data in isi_baris:
                isi_sel = "".join(f"<td>{inline(s)}</td>" for s in baris_data)
                baris_html.append(f"<tr>{isi_sel}</tr>")
            colgroup = ""
            if col_widths and len(col_widths) == len(header):
                colgroup = "<colgroup>" + "".join(
                    f'<col style="width:{w}%">' for w in col_widths
                ) + "</colgroup>"
            keluar.append(
                f"<table>{colgroup}<thead><tr>{kolom}</tr></thead>"
                f"<tbody>{''.join(baris_html)}</tbody></table>"
            )
            continue

        m = re.match(r"^(#{1,4})\s+(.*)$", baris_teks)
        if m:
            tingkat = len(m.group(1))
            keluar.append(f"<h{tingkat}>{inline(m.group(2))}</h{tingkat}>")
            i += 1
            continue

        if re.match(r"^\s*([-*_])\1{2,}\s*$", baris_teks):
            keluar.append("<hr>")
            i += 1
            continue

        if baris_teks.startswith(">"):
            isi = []
            while i < len(baris) and baris[i].startswith(">"):
                isi.append(baris[i].lstrip(">").strip())
                i += 1
            keluar.append(f"<blockquote>{inline(' '.join(isi))}</blockquote>")
            continue

        if re.match(r"^\s*([-*]|\d+\.)\s+", baris_teks):
            terurut = bool(re.match(r"^\s*\d+\.\s", baris_teks))
            nomor_berikut = 1
            butir = []
            while i < len(baris):
                m_item = re.match(r"^\s*([-*]|(\d+)\.)\s+(.*)$", baris[i])
                if not m_item:
                    break
                if m_item.group(2) is not None:
                    angka = int(m_item.group(2))
                    if angka != nomor_berikut:
                        # Nomor tidak berlanjut, jadi baris ini bukan item
                        # baru melainkan lanjutan item sebelumnya. Tanpa
                        # aturan ini, baris seperti "484. DOI ..." akan
                        # salah menjadi daftar baru.
                        if butir:
                            butir[-1] += " " + baris[i].strip()
                            i += 1
                            continue
                        break
                    nomor_berikut = angka + 1
                teks_item = m_item.group(3)
                i += 1
                # Baris lanjutan yang lebih terindentasi masih bagian item.
                while (
                    i < len(baris)
                    and baris[i].strip()
                    and re.match(r"^\s{2,}\S", baris[i])
                    and not re.match(r"^\s*([-*]|\d+\.)\s+", baris[i])
                ):
                    teks_item += " " + baris[i].strip()
                    i += 1
                butir.append(teks_item)
            tag = "ol" if terurut else "ul"
            isi_html = "".join(f"<li>{inline(b)}</li>" for b in butir)
            keluar.append(f"<{tag}>{isi_html}</{tag}>")
            continue

        if not baris_teks.strip():
            i += 1
            continue

        paragraf = [baris_teks]
        i += 1
        while (
            i < len(baris)
            and baris[i].strip()
            and not re.match(
                r"^\s*(#{1,4}\s|[-*]\s|\d+\.\s|>|\||```)", baris[i]
            )
        ):
            paragraf.append(baris[i])
            i += 1
        teks = " ".join(p.strip() for p in paragraf)
        keluar.append(f"<p>{inline(teks)}</p>")

    return "\n".join(keluar)


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
def buat_css(landscape: bool) -> str:
    # Ukuran halaman ditulis sebagai "lebar x tinggi".
    ukuran = "11in 8.5in" if landscape else "8.5in 11in"
    return f"""
@page {{
  size: {ukuran};
  margin: 15mm 13mm 16mm 13mm;
}}
* {{ box-sizing: border-box; }}
body {{
  font-family: Calibri, Arial, sans-serif;
  font-size: 10.5pt;
  line-height: 1.5;
  color: #1B2333;
  margin: 0;
}}
h1 {{
  font-size: 19pt;
  color: #14213D;
  margin: 0 0 6pt 0;
  line-height: 1.2;
}}
h2 {{
  font-size: 12.5pt;
  color: #14213D;
  margin: 16pt 0 6pt 0;
  padding-bottom: 3pt;
  border-bottom: 1px solid #DCE1E8;
  break-after: avoid;
}}
h3 {{
  font-size: 10.8pt;
  color: #0D9488;
  margin: 12pt 0 4pt 0;
  break-after: avoid;
}}
p {{ margin: 0 0 7pt 0; }}
ul, ol {{ margin: 0 0 8pt 0; padding-left: 17pt; }}
li {{ margin-bottom: 3pt; }}
strong {{ color: #14213D; }}
code {{
  font-family: Consolas, "Courier New", monospace;
  font-size: 8.8pt;
  background: #EEF2F7;
  padding: 1pt 1pt 1pt 3pt;
  border-radius: 3px;
  color: #14213D;
}}
pre {{
  background: #14213D;
  color: #E4E9F2;
  padding: 9pt 11pt;
  border-radius: 5px;
  font-size: 8.8pt;
  line-height: 1.45;
  overflow-wrap: break-word;
  white-space: pre-wrap;
  break-inside: avoid;
  margin: 0 0 9pt 0;
}}
pre code {{ background: none; color: inherit; padding: 0; font-size: 8.8pt; }}
blockquote {{
  margin: 0 0 9pt 0;
  padding: 7pt 10pt;
  background: #E3F2F0;
  border-left: 3px solid #0D9488;
  color: #1B2333;
}}
hr {{ border: none; border-top: 1px solid #DCE1E8; margin: 12pt 0; }}
a {{ color: #0B6E66; text-decoration: none; overflow-wrap: anywhere; }}
table {{
  width: 100%;
  border-collapse: collapse;
  margin: 0 0 10pt 0;
  font-size: {"8.4pt" if landscape else "9pt"};
  table-layout: fixed;
}}
th {{
  background: #14213D;
  color: #FFFFFF;
  text-align: left;
  padding: 5pt 5pt;
  font-size: {"8.4pt" if landscape else "9pt"};
  line-height: 1.25;
}}
td {{
  border-bottom: 1px solid #DCE1E8;
  padding: 5pt 5pt;
  vertical-align: top;
  line-height: 1.32;
  overflow-wrap: anywhere;
}}
tbody tr:nth-child(even) {{ background: #F6F8FB; }}
tr {{ break-inside: avoid; }}
thead {{ display: table-header-group; }}
"""


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------
def buat_pdf(
    sumber: pathlib.Path,
    keluaran: pathlib.Path,
    landscape: bool,
    judul_kop: str,
    kaki_kop: str,
    col_widths: list[float] | None,
) -> int:
    from playwright.sync_api import sync_playwright

    md = sumber.read_text(encoding="utf-8")
    isi = md_ke_html(md, col_widths)
    html_dokumen = f"""<!DOCTYPE html>
<html lang="id"><head><meta charset="utf-8">
<style>{buat_css(landscape)}</style></head>
<body>{isi}</body></html>"""

    keluaran.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge")
        halaman = browser.new_page()
        halaman.set_content(html_dokumen, wait_until="load")
        halaman.pdf(
            path=str(keluaran),
            prefer_css_page_size=True,
            print_background=True,
            display_header_footer=True,
            header_template=f"""
              <div style="font-family:Calibri,Arial,sans-serif;font-size:7.5pt;
                          color:#9AA8C2;width:100%;padding:0 13mm;">
                <span>{html.escape(judul_kop)}</span>
              </div>""",
            footer_template=f"""
              <div style="font-family:Calibri,Arial,sans-serif;font-size:7.5pt;
                          color:#9AA8C2;width:100%;padding:0 13mm;
                          display:flex;justify-content:space-between;">
                <span>{html.escape(kaki_kop)}</span>
                <span>Halaman <span class="pageNumber"></span> dari
                  <span class="totalPages"></span></span>
              </div>""",
        )
        browser.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Markdown ke PDF")
    parser.add_argument("masukan", type=pathlib.Path)
    parser.add_argument("keluaran", type=pathlib.Path)
    parser.add_argument("--landscape", action="store_true")
    parser.add_argument("--col-widths", default="")
    parser.add_argument("--judul", default="")
    parser.add_argument("--kaki", default="")
    args = parser.parse_args()

    if not args.masukan.exists():
        print(f"Tidak ditemukan: {args.masukan}", file=sys.stderr)
        return 1

    lebar = []
    if args.col_widths:
        lebar = [float(x) for x in args.col_widths.split(",")]

    judul = args.judul or args.masukan.stem.replace("_", " ")
    kaki = args.kaki or "ML2026 - Klasifikasi Lesi Kulit 7 Kelas"
    kode = buat_pdf(args.masukan, args.keluaran, args.landscape, judul, kaki, lebar or None)
    ukuran = args.keluaran.stat().st_size if args.keluaran.exists() else 0
    print(f"PDF dibuat: {args.keluaran} ({ukuran/1024:.0f} KB)")
    return kode


if __name__ == "__main__":
    sys.exit(main())
