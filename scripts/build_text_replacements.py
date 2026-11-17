"""
Regenerasi tabel penggantian teks (scripts/text_replacements.json).

Tabel ini berisi pasangan (teks salah, teks benar) untuk karakter CJK yang
menyelinap ke komentar kode. Kunci ditulis sebagai escape \\uXXXX supaya
file JSON ini sendiri bebas dari karakter CJK (kalau tidak, tabel akan
memperbaiki dirinya sendiri secara tidak sengaja).

Jalankan:  python scripts/build_text_replacements.py
"""

from __future__ import annotations

import json
import pathlib

REPLACEMENTS = {
    # Istilah yang sempat muncul sebagai karakter CJK
    "\u624b\u52a8": "secara manual",
    "\u89e3\u51b3": "mengatasi",
    "\u5206\u652f": "membelah",
    "\u5206\u6b90": "percabangan",
    "\u4f46": "tapi",
    "\u610f\u601d\u662f": "artinya",
    "\u5927\u6982": "kira-kira",
    "\u771f\u6b63": "sesungguhnya",
    "\u76f4\u89c2": "visual",
    "\u533a\u5206": "membedakan",
    "\u5927\u7ea6": "sekitar",
    "\u8bad\u7ec3": "melatih",
    "\u8ba9": "membuat",
    "\u5b83": "ia",
    "\u6240\u4ee5": "sehingga",
    "\u56e0\u4e3a": "karena",
    "\u5982\u679c": "jika",
    "\u53ef\u4ee5": "bisa",
    "\u5df2\u7ecf": "sudah",
    "\u8fd9\u4e9b": "ini",
    "\u6211\u4eec": "kita",
    "\u4ed6\u4eec": "mereka",
    "\u6570\u636e": "data",
    "\u6a21\u578b": "model",
    "\u7ed3\u679c": "hasil",
    "\u95ee\u9898": "masalah",
    "\u65b9\u6cd5": "metode",
    "\u51c6\u786e": "akurat",
    "\u9519\u8bef": "kesalahan",
    "\u8fd9\u91cc": "di sini",
    "\u4e00\u4e2a": "satu",
    "\u4f7f\u7528": "menggunakan",
    "\u9700\u8981": "perlu",
    "\u901a\u8fc7": "melalui",
    "\u5e94\u8be5": "seharusnya",
    "\u5e76\u4e14": "dan",
    "\u5bf9\u4e8e": "untuk",
    "\u5176\u5b9e\u662f": "sesungguhnya",
    "\u5176\u4ed6": "yang lain",
    "\u5c3d\u53ef\u80fd": "sebanyak mungkin",
    "\u53ef\u80fd\u7684": "kemungkinan",
    "\u5e94\u7528": "untuk aplikasi",
    "\u670d\u7528": "pengguna",
    "\u65e2\u6709\u7684": "yang sudah ada",
    "\u7406\u7531\u7684": "alasannya",
    "\u4f20\u7edf": "tradisional",
    "\u5730\u7b49": "lokasi",
    "\u4ee4\u547d": "perintah",
    "\u4f8b\u5b50": "contoh",
    "\u65b0": "foto",
    "\u4e2d\u6587": "Tionghoa",
}


def main() -> int:
    out = pathlib.Path(__file__).resolve().parent / "text_replacements.json"
    out.write_text(
        json.dumps(REPLACEMENTS, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"Tabel ditulis: {out} ({len(REPLACEMENTS)} pasangan)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
