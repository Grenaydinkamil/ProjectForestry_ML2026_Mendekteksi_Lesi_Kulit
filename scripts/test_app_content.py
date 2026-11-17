"""
scripts/test_app_content.py
==========================
Uji isi (konten) aplikasi Streamlit memakai AppTest, framework resmi
Streamlit untuk pengujian.

KENAPA PERLU SKRIP INI?
  Streamlit merender konten lewat WebSocket, bukan lewat respons HTTP.
  Jadi kalau kita cuma mengambil HTML dengan urllib, kita hanya dapat
  "cangkang" aplikasi: teks kontennya TIDAK akan terlihat, dan cek
  otomatisnya jadi tidak berguna (selalu lolos although aplikasinya
  salah).

  AppTest menjalankan aplikasi sungguhan di dalam proses dan memberi kita
  akses ke elemen yang dirender. Dari situ kita bisa memeriksa:
    - Semua elemen penting benar-benar muncul
    - Upload file benar-benar menghasilkan prediksi
    - File yang bukan gambar ditangani dengan pesan ramah
    - Tidak ada tag HTML yang bocor ke tampilan pengguna

Jalankan:
    python scripts/test_app_content.py
"""

from __future__ import annotations

import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from streamlit.testing.v1 import AppTest  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
UPLOAD_TEST_DIR = pathlib.Path(
    r"C:\Users\GrenWisesLeaf\AppData\Local\Temp\opencode\upload_test"
)

# Elemen yang HARUS ada di halaman utama
WAJIB_HALAMAN = [
    "Skrining Risiko Lesi Kulit dari Foto",
    "alat bantu belajar",
    "Unggah foto",
    "Panduan memotret",
    "Privasi",
    "Model yang dipakai",
]


def teks_semua(at) -> str:
    """Menggabungkan seluruh teks yang dirender Streamlit."""
    bagian = []
    for daftar in (at.markdown, at.sidebar.markdown, at.title,
                   at.header, at.subheader, at.caption, at.info,
                   at.success, at.warning, at.error):
        for e in daftar:
            try:
                bagian.append(e.value)
            except Exception:  # noqa: BLE001
                pass
    return "\n".join(bagian)


def cek_html_bocor(at) -> list[str]:
    """
    Mencari markdown yang merender tag HTML mentah ke pengguna.

    Blok yang SENGAJA memakai HTML (disclaimer, privacy-note) akan
    terdeteksi juga, jadi kita kecualikan baris di dalam blok Those.
    """
    bocor = []
    for daftar in (at.markdown, at.sidebar.markdown):
        for e in daftar:
            v = e.value
            if 'class="privacy-note"' in v or 'class="disclaimer"' in v:
                continue
            for baris in v.split("\n"):
                s = baris.strip()
                if s.startswith("<p>"):
                    bocor.append(s[:90])
    return bocor


def main() -> int:
    print("=" * 70)
    print("UJI ISI APLIKASI STREAMLIT (AppTest)")
    print("=" * 70)

    gagal = 0

    # --- 1. Kondisi kosong ---
    print("\n[1] Kondisi kosong (belum ada file)")
    at = AppTest.from_file(str(REPO / "app" / "app.py"), default_timeout=300)
    at.run()
    if at.exception:
        print(f"    [X] Exception: {at.exception}")
        gagal += 1
    else:
        print("    [OK] Aplikasi start tanpa exception")

    teks = teks_semua(at)
    for wajib in WAJIB_HALAMAN:
        ada = wajib.lower() in teks.lower()
        if not ada:
            gagal += 1
        print(f"    {'[OK]' if ada else '[X]'} '{wajib}'")

    bocor = cek_html_bocor(at)
    if bocor:
        gagal += 1
        print(f"    [X] Tag HTML mentah bocor: {bocor}")
    else:
        print("    [OK] Tidak ada tag HTML yang bocor")

    # --- 2. Upload gambar sungguhan ---
    print("\n[2] Upload foto lesi (kondisi hasil)")
    # CATATAN: AppTest Streamlit punya bug saat mengunggah file BINER:
    # ia mencoba men-decode byte JPEG sebagai UTF-8 dan gagal
    # (UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff).
    # Karena itu file uji dibuat sebagai PNG.
    gambar = UPLOAD_TEST_DIR / "lesi_1.png"
    if not gambar.exists():
        print(f"    [X] File uji tidak ada: {gambar}")
        return 1

    at2 = AppTest.from_file(str(REPO / "app" / "app.py"), default_timeout=300)
    at2.run()
    at2.file_uploader[0].upload(gambar.name, gambar.read_bytes(),
                                    "image/png").run()

    if at2.exception:
        print(f"    [X] Exception setelah upload: {at2.exception}")
        gagal += 1
    else:
        print("    [OK] Upload tidak menyebabkan exception")

    teks2 = teks_semua(at2)
    for kunci in ["Probabilitas per kelas", "Diagnosis paling mungkin"]:
        ada = kunci.lower() in teks2.lower()
        if not ada:
            gagal += 1
        print(f"    {'[OK]' if ada else '[X]'} '{kunci}'")

    # Bar probabilitas dihitung dari blok yang benar-benar berisi bar.
    # CSS di atas juga mengandung teks "bar-row" (definisi .bar-row), jadi
    # kita hanya menghitung blok yang diawali dengan <div class="bar-row">.
    n_bar = sum(
        e.value.count('<div class="bar-row">')
        for e in at2.markdown
    )
    print(f"    {'[OK]' if n_bar == 7 else '[X]'} Bar probabilitas: "
          f"{n_bar} (harus 7, satu per kelas)")
    if n_bar != 7:
        gagal += 1

    if "bukan diagnosis" in teks2.lower():
        print("    [OK] Disclaimer masih ada setelah upload")
    else:
        gagal += 1
        print("    [X] Disclaimer hilang")

    # --- 3. File dengan ekstensi .png tapi isinya bukan gambar ---
    #
    # CATATAN: file .txt/.pdf DITOLAK oleh Streamlit sendiri di level
    # file_uploader (pesan "Invalid file extension"), jadi jalur itu tidak
    # pernah sampai ke aplikasi. Untuk menguji penanganan gambar rusak,
    # kita pakai file .png yang isinya bukan gambar sungguhan.
    print("\n[3] File .png yang isinya bukan gambar")
    at3 = AppTest.from_file(str(REPO / "app" / "app.py"), default_timeout=300)
    at3.run()
    rusak = UPLOAD_TEST_DIR / "rusak.png"
    rusak.write_bytes(b"INI BUKAN GAMBAR PNG YANG BENAR")
    at3.file_uploader[0].upload(rusak.name, rusak.read_bytes(),
                                "image/png").run()

    if at3.exception:
        print(f"    [X] Exception: {at3.exception}")
        gagal += 1
    else:
        print("    [OK] Tidak ada exception (ditangani di dalam aplikasi)")

    # Pesan error tampil lewat st.error, jadi ada di at.error (bukan markdown)
    pesan_ramah = any("bukan gambar yang bisa dibaca" in e.value
                      for e in at3.error)
    print(f"    {'[OK]' if pesan_ramah else '[X]'} Pesan ramah untuk file "
          f"rusak ditampilkan")
    for e in at3.error:
        print(f"      {e.value[:120]}")
    if not pesan_ramah:
        gagal += 1

    # --- 4. Contoh gambar siap pakai ---
    #
    # Fitur contoh gambar dipakai untuk demo, jadi harus benar-benar
    # bisa dipilih dan menghasilkan prediksi.
    print("\n[4] Contoh gambar siap pakai")
    contoh_dir = REPO / "app" / "contoh"
    if not any(contoh_dir.glob("*")):
        print("    [X] Folder app/contoh kosong. Jalankan:"
              " python scripts/siapkan_contoh_gambar.py")
        gagal += 1
    else:
        at4 = AppTest.from_file(str(REPO / "app" / "app.py"),
                                default_timeout=300)
        at4.run()

        if not at4.checkbox:
            print("    [X] Checkbox 'Pakai contoh ini' tidak ditemukan")
            gagal += 1
        else:
            at4.checkbox[0].check().run()
            if at4.exception:
                print(f"    [X] Exception: {at4.exception}")
                gagal += 1
            else:
                print("    [OK] Contoh gambar bisa dipilih tanpa exception")

            n_bar4 = sum(
                e.value.count('<div class="bar-row">')
                for e in at4.markdown
            )
            print(f"    {'[OK]' if n_bar4 == 7 else '[X]'} Bar probabilitas "
                  f"dari contoh: {n_bar4} (harus 7)")
            if n_bar4 != 7:
                gagal += 1

    print("\n" + "=" * 70)
    if gagal == 0:
        print("HASIL: SEMUA PEMERIKSAAN LULUS.")
    else:
        print(f"HASIL: {gagal} pemeriksaan gagal.")
    print("=" * 70)
    return 1 if gagal else 0


if __name__ == "__main__":
    sys.exit(main())
