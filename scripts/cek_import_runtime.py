"""
scripts/cek_import_runtime.py
=============================
Memeriksa bahwa semua paket pihak ketiga yang diimpor berkas yang dipakai
aplikasi sudah terdaftar di app/requirements.txt.

Kenapa pemeriksaan ini perlu:
    src/features.py mengimpor tqdm di level modul. Aplikasi memakai model
    deep learning, jadi fitur klasik tidak pernah dipanggil, tqdm tidak
    ikut dipasang di server, dan deploy gagal dengan ModuleNotFoundError
    padahal semua pemeriksaan lokal lulus.

Skrip ini membaca daftar import dari app/app.py dan src/*.py, lalu
mencocokkannya dengan app/requirements.txt.

Jalankan:
    python scripts/cek_import_runtime.py
"""

from __future__ import annotations

import ast
import pathlib
import re
import sys
from importlib.metadata import distributions

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Modul milik proyek yang bisa diimpor aplikasi.
MODUL_PROYEK = {
    str(p.relative_to(REPO_ROOT)).replace("\\", "/").removesuffix(".py")
    for p in (REPO_ROOT / "src").glob("*.py")
} | {"app/app"}

# Peta nama modul ke nama paket di PyPI, untuk modul yang namanya
# berbeda dari nama paketnya.
ALIAS = {
    "PIL": "pillow",
    "cv2": "opencv-python",
    "skimage": "scikit-image",
    "sklearn": "scikit-learn",
    "yaml": "pyyaml",
    "bs4": "beautifulsoup4",
    "tensorflow": "tensorflow-cpu",
    "streamlit_shadcn_ui": "streamlit-shadcn-ui",
}

# Paket bawaan Python: tidak perlu ada di requirements.
STDLIB = set(sys.stdlib_module_names)


def import_dari(berkas: str) -> tuple[set[str], set[str]]:
    """Mengembalikan (modul pihak ketiga, modul milik proyek) dari berkas."""
    pihak: set[str] = set()
    internal: set[str] = set()
    path = REPO_ROOT / f"{berkas}.py"
    if not path.exists():
        return pihak, internal
    pohon = ast.parse(path.read_text(encoding="utf-8"), filename=berkas)
    for simpul in ast.walk(pohon):
        if isinstance(simpul, ast.Import):
            for alias in simpul.names:
                nama = alias.name.split(".")[0]
                (internal if nama in {m.split("/")[-1] for m in MODUL_PROYEK}
                 else pihak).add(nama)
        elif isinstance(simpul, ast.ImportFrom):
            if simpul.level and simpul.level > 0:
                # Import relatif: "from . import config" menunjuk src/config.py
                for alias in simpul.names:
                    internal.add(f"src/{alias.name}")
                if simpul.module:
                    internal.add(f"src/{simpul.module.split('.')[-1]}")
                continue
            if not simpul.module:
                continue
            nama = simpul.module.split(".")[0]
            if nama == "src":
                for alias in simpul.names:
                    internal.add(f"src/{alias.name}")
            elif nama in {m.split("/")[-1] for m in MODUL_PROYEK}:
                internal.add(f"src/{nama}")
            else:
                pihak.add(nama)
    return pihak, internal


def rantai_runtime() -> set[str]:
    """Berkas yang benar-benar dimuat aplikasi saat runtime."""
    visited: set[str] = set()
    antre = ["app/app"]
    while antre:
        saat = antre.pop()
        if saat in visited:
            continue
        visited.add(saat)
        _, internal = import_dari(saat)
        for m in internal:
            if m in MODUL_PROYEK and m not in visited:
                antre.append(m)
    return visited


def semua_import(berkas: list[str]) -> set[str]:
    """Mengumpulkan nama modul pihak ketiga dari daftar berkas."""
    hasil: set[str] = set()
    for rel in berkas:
        pihak, _ = import_dari(rel)
        hasil |= pihak
    return hasil


def paket_terpasang() -> set[str]:
    """Nama paket yang terpasang di lingkungan ini."""
    return {
        d.metadata["Name"].lower()
        for d in distributions()
        if d.metadata["Name"]
    }


def main() -> int:
    req_path = REPO_ROOT / "app" / "requirements.txt"
    isi = req_path.read_text(encoding="utf-8")
    dipin = {
        m.group(1).lower()
        for m in re.finditer(r"^([A-Za-z0-9_.\-]+)\s*==", isi, re.MULTILINE)
    }
    terpasang = paket_terpasang()

    rantai = sorted(rantai_runtime())
    modul = semua_import(rantai)
    pihak_ketiga = sorted(
        m for m in modul if m not in STDLIB and m not in {"src", "app"}
    )

    bermasalah: list[tuple[str, str, bool]] = []
    for m in pihak_ketiga:
        nama = ALIAS.get(m, m).lower()
        if nama in dipin:
            continue
        bermasalah.append((m, nama, m.lower() in terpasang))

    print("Berkas runtime yang benar-benar dimuat aplikasi:")
    for rel in rantai:
        print(f"  - {rel}.py")
    print(f"\nPaket dipin di app/requirements.txt ({len(dipin)}):")
    print("  " + ", ".join(sorted(dipin)))

    print("\nImport pihak ketiga yang dipakai:")
    for m in pihak_ketiga:
        print(f"  - {m}")

    if bermasalah:
        print(
            "\nHASIL: GAGAL. Paket berikut diimpor runtime tapi tidak "
            "terdaftar di app/requirements.txt:"
        )
        for m, nama, ada_di_lokal in bermasalah:
            ket = (
                "terpasang lokal, tapi tidak ada di requirements"
                if ada_di_lokal
                else "tidak ditemukan di lingkungan ini"
            )
            print(f"  [X] {m} -> paket {nama} ({ket})")
        print("\nTambahkan paketnya ke app/requirements.txt, lalu deploy ulang.")
        return 1

    print(
        "\nHASIL: LOLUS. Semua import runtime tercakup app/requirements.txt."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
