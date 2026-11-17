"""
Memindai seluruh file sumber untuk mencari argumen yang TIDAK ADA di
API library yang dipakai.

Tujuan: menangkap "glitch" seperti `palambatte` (yang seharusnya
`palette`) dan `labels=` pada boxplot matplotlib 3.11.

Cara kerja: introspeksi signature fungsi dari library yang terpasang, lalu
cek setiap pemanggilan di kode kita.

Jalankan:  python scripts/check_api_calls.py
"""
from __future__ import annotations

import ast
import inspect
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Peta nama fungsi yang kita pakai -> library asalnya
KNOWN = {
    "sns": "seaborn",
    "plt": "matplotlib.pyplot",
    "np": "numpy",
    "pd": "pandas",
    "tf": "tensorflow",
    "sklearn": "sklearn",
}


def main() -> int:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import tensorflow as tf

    modul = {
        "sns": sns, "plt": plt, "np": np, "pd": pd, "tf": tf,
    }

    masalah = 0
    file_diperiksa = 0

    pola = ("src/*.py", "app/*.py", "scripts/*.py",
            "scripts/notebook_src/*.py")

    for pola_file in pola:
        for path in sorted(REPO_ROOT.glob(pola_file)):
            file_diperiksa += 1
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError as exc:
                print(f"[X] {path.name}: SyntaxError {exc}")
                masalah += 1
                continue

            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if not isinstance(func, ast.Attribute):
                    continue
                # Ambil nama basis: mis. sns.boxplot
                base = func.value
                if not isinstance(base, ast.Name):
                    continue
                if base.id not in modul:
                    continue
                owner = modul[base.id]
                nama_fungsi = func.attr

                # Cari fungsi yang benar-benar tersedia
                target = getattr(owner, nama_fungsi, None)
                if target is None or not callable(target):
                    continue
                try:
                    sig = inspect.signature(target)
                except (TypeError, ValueError):
                    continue
                params = set(sig.parameters)
                if any(p.kind == inspect.Parameter.VAR_KEYWORD
                       for p in sig.parameters.values()):
                    continue

                # Cek keyword yang dipakai
                for kw in node.keywords:
                    if kw.arg is None:
                        continue
                    if kw.arg not in params:
                        print(f"[X] {path.relative_to(REPO_ROOT)}:{node.lineno} "
                              f"{base.id}.{nama_fungsi}(... {kw.arg}=...) "
                              f"-> parameter '{kw.arg}' TIDAK ADA "
                              f"(pilihan: {sorted(params)[:8]})")
                        masalah += 1

    print(f"\nFile diperiksa: {file_diperiksa} | masalah: {masalah}")
    return 1 if masalah else 0


if __name__ == "__main__":
    sys.exit(main())
