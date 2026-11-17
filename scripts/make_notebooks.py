"""
scripts/make_notebooks.py
=========================
Membuat file .ipynb dari sumber .py agar notebook bisa dibuat reproducible.

Kenapa pakai skrip?
  Menulis file .ipynb sebagai JSON manual itu panjang dan rawan typo
  (terutama escaping newline di dalam sel). Skrip ini menyimpan logika
  notebook sebagai kode .py yang biasa, lalu mengubahnya menjadi .ipynb.

Cara pakai:
    python scripts/make_notebooks.py            # buat semua notebook
    python scripts/make_notebooks.py 01_eda     # buat hanya notebook tertentu
    python scripts/make_notebooks.py --force 04_svm_rf   # paksa tulis ulang

PERINGATAN PENTING:
    Notebook yang sudah dieksekusi berisi OUTPUT (hasil, grafik, tabel).
    Menulis ulang .ipynb menghapus output itu, dan outputnya hanya bisa
    didapat dengan menjalankan notebook lagi (bisa memakan waktu 10-60
    menit untuk notebook training).

    Karena itu skrip ini LEWATI notebook yang sudah punya output, kecuali
    diberi flag `--force`. Ini mencegah output hilang karena salah perintah.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"


def sudah_punya_output(path: Path) -> bool:
    """True kalau notebook di lokasi itu sudah punya output hasil eksekusi."""
    if not path.exists():
        return False
    try:
        with open(path, encoding="utf-8") as f:
            nb = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return any(c.get("outputs") for c in nb.get("cells", [])
               if c.get("cell_type") == "code")


def parse_cells(source_py: Path) -> list[dict]:
    """
    Mengubah file .py menjadi daftar sel notebook.

    Aturan parsing (sederhana tapi cukup untuk proyek ini):
      - Baris yang diawali "# %% [markdown]" memulai sel markdown
      - Baris yang diawali "# %%"       memulai sel kode
      - Sel markdown: isi sel adalah SELURUH baris yang diawali "#"
        (tanpa tanda "# " di depan)
      - Sel kode: isi sel adalah baris apa adanya

    CATATAN PENTING soal format `source`:
      Format notebook resmi menyimpan `source` sebagai DAFTAR baris, dan
      tiap baris (kecuali yang terakhir) harus BERAKHIR newline.
      Asyncron: nbformat membaca daftar ini dengan ''.join(...), bukan
      '\n'.join(...). Kalau newline tidak disertakan, semua baris akan
      menempel jadi satu paragraf panjang dan markdown-nya rusak.
      Karena itu kita memakai splitlines(keepends=True).
    """
    lines = source_py.read_text(encoding="utf-8").split("\n")

    cells: list[dict] = []
    current_type = None
    current_lines: list[str] = []
    cell_counter = 0

    def flush():
        nonlocal cell_counter
        if current_type is None:
            return
        cell_counter += 1
        cell_id = f"cell-{cell_counter:03d}"

        if current_type == "markdown":
            # Setiap baris markdown diawali "# " (atau "#"). Kita buang
            # AWAL baris itu saja, sisanya dipertahankan utuh.
            stripped_lines = []
            for ln in current_lines:
                if ln.startswith("# "):
                    stripped_lines.append(ln[2:])
                elif ln.startswith("#"):
                    stripped_lines.append(ln[1:])
                else:
                    stripped_lines.append(ln)
            body = "\n".join(stripped_lines)
        else:
            body = "\n".join(current_lines)
            if not body.strip():
                return

        cell = {
            "id": cell_id,
            "cell_type": current_type,
            "metadata": {},
            "source": body.splitlines(keepends=True),
        }
        if current_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# %% [markdown]"):
            flush()
            current_type = "markdown"
            current_lines = []
        elif stripped.startswith("# %%"):
            flush()
            current_type = "code"
            current_lines = []
        else:
            if current_type is not None:
                current_lines.append(line)
    flush()

    return cells


def build_notebook(cells: list[dict], title: str) -> dict:
    """Membangun struktur file .ipynb yang valid."""
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.13.9",
                "mimetype": "text/x-python",
                "file_extension": ".py",
                "pygments_lexer": "ipython3",
                "nbconvert_exporter": "python",
                "codemirror_mode": {"name": "ipython", "version": 3},
            },
            "title": title,
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    src_dir = REPO_ROOT / "scripts" / "notebook_src"
    src_dir.mkdir(parents=True, exist_ok=True)
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

    argv = sys.argv[1:]
    force = "--force" in argv
    targets = [a for a in argv if not a.startswith("--")]

    built = 0
    skipped = 0

    for src in sorted(src_dir.glob("*.py")):
        name = src.stem
        if targets and name not in targets:
            continue

        out = NOTEBOOKS_DIR / f"{name}.ipynb"
        if not force and sudah_punya_output(out):
            print(f"[LEWATI] {out.name:32s} sudah punya output "
                  f"(pakai --force untuk menimpa)")
            skipped += 1
            continue

        cells = parse_cells(src)
        nb = build_notebook(cells, title=name.replace("_", " ").title())
        with open(out, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write("\n")

        n_md = sum(1 for c in cells if c["cell_type"] == "markdown")
        n_code = len(cells) - n_md
        print(f"[OK] {out.name:32s} {n_md:3d} sel markdown, {n_code:3d} sel kode")
        built += 1

    print(f"\nTotal notebook dibuat: {built} | dilewati (punya output): {skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
