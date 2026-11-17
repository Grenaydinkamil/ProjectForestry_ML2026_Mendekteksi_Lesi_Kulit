"""
Perbarui semua rujukan `requirements.txt` di repo setelah file root
diberi nama `requirements-training.txt`.

Alasan perubahan: Streamlit Community Cloud mencari file dependensi mulai
dari folder entrypoint lalu root, dan hanya membaca file PERTAMA yang
ditemukan. Dua file `requirements.txt` (satu di root berisi tensorflow
penuh ~3 GB, satu di app/ berisi tensorflow-cpu) adalah jebakan: kalau
suatu saat entrypoint diubah, platform bisa saja memilih yang salah dan
deploy gagal karena ukuran. Dengan hanya satu file requirements.txt di
repo (yaitu app/requirements.txt), tidak ada lagi kemungkinan salah pilih.
"""

import pathlib

REPO = pathlib.Path(
    r"M:\GrenPlums Roses\Program32bit\Forestrypansy"
    r"\ML2026_Kelas_3_Kelompok_1_LesiKulit"
)

# ---------------------------------------------------------------- README
p = REPO / "README.md"
t = p.read_text(encoding="utf-8")

lama = """│   ├── contoh/                # Contoh gambar untuk demo (4 file)
│   └── requirements.txt       # Requirement khusus deploy (tensorflow-cpu)"""
baru = """│   ├── contoh/                # Contoh gambar untuk demo (4 file)
│   └── requirements.txt       # Requirement DEPLOY (tensorflow-cpu)"""
assert lama in t
t = t.replace(lama, baru)

lama2 = """├── requirements.txt
├── .gitignore"""
baru2 = """├── requirements-training.txt    # Requirement untuk training & EDA di laptop
├── .gitignore"""
assert lama2 in t
t = t.replace(lama2, baru2)

lama3 = "pip install -r requirements.txt"
baru3 = "pip install -r requirements-training.txt"
assert lama3 in t
t = t.replace(lama3, baru3)

p.write_bytes(t.encode("utf-8"))
print("README.md diperbarui")

# ------------------------------------------------- PANDUAN_PENGGUNAAN.md
p = REPO / "docs" / "PANDUAN_PENGGUNAAN.md"
t = p.read_text(encoding="utf-8")

lama4 = """```bash
# Instalasi
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt"""
baru4 = """```bash
# Instalasi
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\Activate.ps1
pip install -r requirements-training.txt"""
assert lama4 in t
t = t.replace(lama4, baru4)

lama5 = "| `ModuleNotFoundError` | Virtual environment belum aktif | Aktifkan `.venv` lalu `pip install -r requirements.txt` |"
baru5 = "| `ModuleNotFoundError` | Virtual environment belum aktif | Aktifkan `.venv` lalu `pip install -r requirements-training.txt` |"
assert lama5 in t
t = t.replace(lama5, baru5)

lama6 = """4. Tambahkan `requirements.txt` dari folder `app/`."""
baru6 = """4. Tidak perlu menambahkan apa pun. Platform membaca `app/requirements.txt`
   secara otomatis karena entrypoint berada di folder `app/`."""
assert lama6 in t
t = t.replace(lama6, baru6)

lama7 = """```bash
# Instalasi
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt

# Cek environment"""
baru7 = """```bash
# Instalasi
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\Activate.ps1
pip install -r requirements-training.txt

# Cek environment"""
assert lama7 in t
t = t.replace(lama7, baru7)

p.write_bytes(t.encode("utf-8"))
print("docs/PANDUAN_PENGGUNAAN.md diperbarui")

# ---------------------------------------------------- app/requirements.txt
p = REPO / "app" / "requirements.txt"
t = p.read_text(encoding="utf-8")

lama8 = """# Bedakan dengan requirements.txt di root:
#   - requirements.txt (root)  -> untuk TRAINING di laptop (butuh jupyter, dsb)
#   - requirements.txt (app/)   -> untuk DEPLOY di Streamlit Community Cloud"""
baru8 = """# File INI adalah satu-satunya file dependensi di repo ini, dan itu
# disengaja. Streamlit Community Cloud mencari file dependensi mulai dari
# folder entrypoint (app/), lalu root, dan hanya membaca file PERTAMA
# yang ditemukan. Kalau ada dua file requirements.txt,platform bisa saja
# memilih yang salah dan deploy gagal.
#
# Karena itu:
#   - app/requirements.txt      -> untuk DEPLOY (tensorflow-cpu)  [DIPAKAI]
#   - requirements-training.txt  -> untuk TRAINING di laptop (root, punya
#                                    tensorflow penuh + jupyterlab)"""
assert lama8 in t
t = t.replace(lama8, baru8)
p.write_bytes(t.encode("utf-8"))
print("app/requirements.txt diperbarui")
