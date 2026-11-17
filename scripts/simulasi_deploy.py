"""
scripts/simulasi_deploy.py
=========================
Simulasi deploy ke Streamlit Community Cloud, sebelum upload.

Kenapa perlu?
  Deploy ke cloud tidak bisa diulang dengan mudah, dan kesalahannya
  baru ketahuan setelah repository sudah publicly tersedia. Skrip ini
  reproducing apa yang terjadi di server:

    1. Mengekspor isi git ke folder BERSIH (hanya berkas yang ter-commit)
    2. Menjalankan aplikasi dari folder itu
    3. Memastikan model, konfigurasi, dan contoh gambar terbaca
    4. Menjalankan uji isi aplikasi (AppTest) dari folder simulasi

  Kalau langkah ini lulus, kemungkinan besar deploy-nya juga lulus.

Yang TIDAK ikut (dan memang tidak boleh ikut):
    - data/         (dataset,lisensi terpisah)
    - .venv/        (environment lokal)
    - *.joblib      (model klasik, tidak dipakai aplikasi)

Jalankan:
    python scripts/simulasi_deploy.py
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Berkas yang WAJIB ada di server agar aplikasi bisa jalan
WAJIB = [
    "app/app.py",
    "app/requirements.txt",
    "app/contoh",
    "src/config.py",
    "src/predictor.py",
    "models/mobilenetv2_best.keras",
    "models/label_map.json",
    "models/preprocess_config.json",
    "models/threshold_config.json",
]

# Yang TIDAK boleh ikut
DILARANG = ["data", ".venv", "node_modules"]


def jalankan(cmd: list[str], cwd=None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


def main() -> int:
    print("=" * 72)
    print(" SIMULASI DEPLOY  -  apa yang dilihat server Streamlit Community Cloud")
    print("=" * 72)

    # 1. Ekspor isi git ke folder bersih
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="simulasi_deploy_"))
    simulasi = tmp / "repo"
    simulasi.mkdir()
    tar = tmp / "repo.tar"

    r = jalankan(["git", "archive", "-o", str(tar), "HEAD"], cwd=REPO_ROOT)
    if r.returncode != 0:
        print(f"[X] git archive gagal: {r.stderr.strip()[:200]}")
        return 1

    r = jalankan(["tar", "-xf", str(tar), "-C", str(simulasi)])
    if r.returncode != 0:
        print(f"[X] ekstrak gagal: {r.stderr.strip()[:200]}")
        return 1

    semua = sorted(p.relative_to(simulasi).as_posix()
                   for p in simulasi.rglob("*") if p.is_file())
    ukuran = sum((simulasi / f).stat().st_size for f in semua)

    print(f"\n[1] Isi yang diterima server")
    print(f"    jumlah berkas : {len(semua)}")
    print(f"    total ukuran  : {ukuran / 1024 / 1024:.1f} MB")
    print(f"    batas GitHub  : 100 MB/berkas, 1 GB/repo")
    print(f"    batas SCC     : 200 MB/berkas, 1 GB/repo")

    # 2. Berkas wajib
    print("\n[2] Berkas wajib untuk aplikasi")
    hilang = []
    for w in WAJIB:
        ada = (simulasi / w).exists()
        if not ada:
            hilang.append(w)
        print(f"    {'[OK]  ' if ada else '[X]   '}{w}")

    # 3. Yang tidak boleh ikut
    print("\n[3] Yang tidak boleh ikut")
    for d in DILARANG:
        n = len([f for f in semua if f.startswith(d + "/")])
        print(f"    {'[OK]  ' if n == 0 else '[X]   '}{d}/  ({n} berkas)")

    # 4. Hanya satu requirements.txt
    req = [f for f in semua if f.endswith("requirements.txt")]
    print("\n[4] File dependensi")
    print(f"    ditemukan: {req}")
    if len(req) != 1:
        print("    [X] Harus tepat satu requirements.txt (dokumentasi resmi "
              "Streamlit: hanya file PERTAMA yang dibaca)")
        hilang.append("requirements.txt")

    # 5. Uji aplikasi dari folder simulasi
    print("\n[5] Menjalankan aplikasi dari folder simulasi")
    r = jalankan([sys.executable,
                  str(simulasi / "scripts" / "test_app_content.py")],
                 cwd=simulasi)
    lulus = r.returncode == 0 and "SEMUA PEMERIKSAAN LULUS" in (
        (r.stdout or "") + (r.stderr or ""))
    print(f"    {'[OK]  ' if lulus else '[X]   '}Uji isi aplikasi "
          f"(AppTest) dari folder simulasi")
    if not lulus:
        gagal = (r.stdout or "") + (r.stderr or "")
        for baris in gagal.splitlines()[-12:]:
            print("       " + baris)
        hilang.append("uji aplikasi")

    # Ringkasan
    print("\n" + "=" * 72)
    if hilang:
        print(f" HASIL: GAGAL - {len(hilang)} masalah:")
        for h in hilang:
            print(f"   - {h}")
        kode = 1
    else:
        print(" HASIL: SIMULASI LOLUS")
        print(" Repo siap di-deploy ke Streamlit Community Cloud.")
        print(" Ingat: saat deploy, pilih Python 3.13 di 'Advanced settings'.")
        kode = 0
    print("=" * 72)

    shutil.rmtree(tmp, ignore_errors=True)
    return kode


if __name__ == "__main__":
    sys.exit(main())
