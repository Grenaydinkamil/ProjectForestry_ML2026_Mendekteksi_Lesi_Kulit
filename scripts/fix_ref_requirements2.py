"""
Perbarui rujukan requirements.txt di docs/PANDUAN_PENGGUNAAN.md dan
app/requirements.txt setelah file root di-rename menjadi
requirements-training.txt.
"""

import pathlib

REPO = pathlib.Path(
    r"M:\GrenPlums Roses\Program32bit\Forestrypansy"
    r"\ML2026_Kelas_3_Kelompok_1_LesiKulit"
)

# ------------------------------------------------- PANDUAN_PENGGUNAAN.md
p = REPO / "docs" / "PANDUAN_PENGGUNAAN.md"
t = p.read_text(encoding="utf-8")

n = t.count("pip install -r requirements.txt")
t = t.replace("pip install -r requirements.txt",
              "pip install -r requirements-training.txt")
print(f"PANDUAN_PENGGUNAAN.md: {n} rujukan 'pip install' diperbarui")

lama = "4. Tambahkan `requirements.txt` dari folder `app/`."
baru = ("4. Tidak perlu menambahkan apa pun. Platform membaca "
        "`app/requirements.txt` secara otomatis karena entrypoint "
        "berada di folder `app/`.")
if lama in t:
    t = t.replace(lama, baru)
    print("  catatan deploy diperbarui")

p.write_bytes(t.encode("utf-8"))

# ---------------------------------------------------- app/requirements.txt
p = REPO / "app" / "requirements.txt"
t = p.read_text(encoding="utf-8")

lama = """# Bedakan dengan requirements.txt di root:
#   - requirements.txt (root)  -> untuk TRAINING di laptop (butuh jupyter, dsb)
#   - requirements.txt (app/)   -> untuk DEPLOY di Streamlit Community Cloud"""
baru = """# File INI adalah satu-satunya file dependensi bernama requirements.txt
# di repo ini, dan itu disengaja.
#
# Streamlit Community Cloud mencari file dependensi mulai dari folder
# entrypoint (app/), lalu root, dan hanya membaca file PERTAMA yang
# ditemukan. Kalau ada dua file requirements.txt, platform bisa saja
# memilih yang salah dan deploy gagal.
#
# Karena itu:
#   - app/requirements.txt      -> untuk DEPLOY (tensorflow-cpu)  [DIPAKAI]
#   - requirements-training.txt  -> untuk TRAINING di laptop (root, berisi
#                                    tensorflow penuh + jupyterlab)"""
if lama in t:
    t = t.replace(lama, baru)
    print("app/requirements.txt: komentar diperbarui")

p.write_bytes(t.encode("utf-8"))
print("selesai")
