"""
app/app.py
==========
Aplikasi web Streamlit untuk skrining awal risiko keganasan lesi kulit.

ALUR APLIKASI
-------------
    Input (unggah foto) -> Preprocessing -> Model -> Prediksi -> Hasil

Aturan yang dijaga di file ini:
  1. Preprocessing memakai fungsi yang SAMA dengan notebook training
     (lewat src.predictor), sehingga hasil di aplikasi konsisten.
  2. Model dimuat dengan @st.cache_resource agar tidak di-load ulang
     tiap interaksi.
  3. Ada tiga kondisi tampilan yang lengkap: kosong, sedang memproses,
     dan gagal. Tidak ada layar kosong tanpa penjelasan.
  4. Disclaimer tidak bisa dilewatkan dan selalu terlihat.
  5. Foto pengguna TIDAK disimpan dan TIDAK dikirim ke pihak mana pun.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st
from PIL import Image, UnidentifiedImageError

from src import config as C
from src import predictor as PR

# ---------------------------------------------------------------------------
# KONFIGURASI HALAMAN
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Skrining Lesi Kulit",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# DESAIN: IDENTITAS VISUAL
# ---------------------------------------------------------------------------
# Brief: aplikasi untuk skrining dermatologi. Arah desainnya "klinis
# tenang", bukan "dashboard SaaS":
#   - Latar putih hangat (paper), bukan abu-abu gelap
#   - Aksen warna: biru Slate untuk netral, Merah bata untuk tanda bahaya
#   - Tipografi: sans-serif untuk isi, angka tabular untuk metrik
#   - Pemisah garis tipis, bukan kartu-kotak dengan bayangan
# Semua token warna dikumpulkan di sini supaya konsisten dan mudah
# diaudit (termasuk untuk rasio kontras WCAG AA).
CSS = """
<style>
    :root {
        --paper:      #FBFAF7;
        --ink:        #1A1D21;
        --ink-soft:   #4A5058;
        --ink-faint:  #6A717A;
        --rule:       #DFDCD5;
        --rule-soft:  #EDEAE3;
        --accent:     #2F5D7C;
        --accent-dim: #E8EFF4;
        --danger:     #9E2A2B;
        --danger-dim: #F7E9E8;
        --warn:       #8A5A15;
        --warn-dim:   #FBF0DC;
        --ok:         #2E6B4F;
        --ok-dim:     #E6F0EA;
    }

    .stApp { background: var(--paper); }

    /* Judul: pakai serif supaya terasa seperti laporan klinis, bukan
       template SaaS */
    h1, h2, h3 {
        font-family: Georgia, "Times New Roman", serif;
        color: var(--ink);
        letter-spacing: -0.01em;
    }
    h1 { font-size: 1.9rem; font-weight: 600; line-height: 1.2; }
    h2 { font-size: 1.25rem; font-weight: 600; margin-top: 1.6rem; }
    h3 { font-size: 1.02rem; font-weight: 600; }

    p, li, label, .stMarkdown {
        font-family: "Helvetica Neue", Arial, sans-serif;
        color: var(--ink-soft);
        line-height: 1.6;
    }

    /* Angka metrik: tabular supaya sejajar */
    .metric-value {
        font-variant-numeric: tabular-nums;
        font-family: "Helvetica Neue", Arial, sans-serif;
        font-weight: 600;
        color: var(--ink);
    }

    /* Aturan pemisah tipis, bukan kartu berbayang */
    hr { border: none; border-top: 1px solid var(--rule); margin: 1.6rem 0; }

    /* Panel disclaimer: satu blok dengan warna PAPER hangat, tanpa
       gradien, tanpa shadow, tanpa radius besar */
    .disclaimer {
        background: var(--danger-dim);
        border-left: 3px solid var(--danger);
        padding: 0.85rem 1.05rem;
        margin: 0.9rem 0 1.2rem;
        border-radius: 2px;
    }
    .disclaimer p {
        color: #6B2021;
        margin: 0 0 0.45rem;
        font-size: 0.9rem;
        line-height: 1.55;
    }
    .disclaimer p:last-child { margin-bottom: 0; }
    .disclaimer strong { color: var(--danger); }

    .privacy-note {
        background: var(--accent-dim);
        border-left: 3px solid var(--accent);
        padding: 0.7rem 1rem;
        border-radius: 2px;
        margin: 0.8rem 0;
    }
    .privacy-note p {
        color: #24485F;
        margin: 0;
        font-size: 0.88rem;
    }

    /* Label kecil di atas nilai: bukan badge capsule, cuma teks kecil
       dengan letter-spacing normal */
    .eyebrow {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        color: var(--ink-faint);
        font-weight: 600;
        margin-bottom: 0.15rem;
    }

    /* Bar hasil: lebar proporsional, warna menandai kelas ganas */
    .bar-row {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        margin-bottom: 0.34rem;
    }
    .bar-label {
        width: 10.5rem;
        font-size: 0.84rem;
        color: var(--ink);
        flex-shrink: 0;
    }
    .bar-track {
        flex: 1;
        height: 9px;
        background: var(--rule-soft);
        border-radius: 1px;
        overflow: hidden;
    }
    .bar-fill { height: 100%; border-radius: 1px; }
    .bar-value {
        width: 4.2rem;
        text-align: right;
        font-size: 0.84rem;
        font-variant-numeric: tabular-nums;
        color: var(--ink-soft);
        flex-shrink: 0;
    }

    /* Kotak status risiko */
    .status-box {
        padding: 1rem 1.15rem;
        border-radius: 2px;
        border-left: 3px solid;
        margin: 0.5rem 0 1rem;
    }
    .status-aman      { background: var(--ok-dim);     border-color: var(--ok); }
    .status-kabur     { background: var(--warn-dim);   border-color: var(--warn); }
    .status-waspada   { background: var(--danger-dim); border-color: var(--danger); }
    .status-title {
        font-weight: 600;
        font-size: 1.02rem;
        margin-bottom: 0.3rem;
        font-family: "Helvetica Neue", Arial, sans-serif;
    }
    .status-body { font-size: 0.9rem; margin: 0; line-height: 1.55; }

    /* Panduan memotret: daftar bernomor dengan bentuk, bukan ikon */
    .guide-item {
        display: flex;
        gap: 0.7rem;
        margin-bottom: 0.55rem;
        align-items: flex-start;
    }
    .guide-num {
        flex-shrink: 0;
        width: 1.35rem;
        height: 1.35rem;
        border: 1px solid var(--accent);
        color: var(--accent);
        border-radius: 50%;
        font-size: 0.72rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-top: 0.12rem;
    }
    .guide-text { font-size: 0.89rem; line-height: 1.5; }
    .guide-text strong { color: var(--ink); }

    /* Link dan tombol */
    a { color: var(--accent); }
    .stButton > button {
        border-radius: 2px;
        border: 1px solid var(--accent);
        background: var(--accent);
        color: #FFFFFF;
        font-weight: 600;
        cursor: pointer;
    }
    .stButton > button:hover {
        background: #24485F;
        border-color: #24485F;
    }
    .stButton > button:focus-visible {
        outline: 2px solid var(--accent);
        outline-offset: 2px;
    }
    .stFileUploader { border-radius: 2px; }

    /* Caption kecil */
    .caption-muted { font-size: 0.82rem; color: var(--ink-faint); }

    /* Kotak info kelas model */
    .meta-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.86rem;
    }
    .meta-table td {
        padding: 0.32rem 0;
        border-bottom: 1px solid var(--rule-soft);
        color: var(--ink-soft);
    }
    .meta-table td:first-child {
        color: var(--ink-faint);
        width: 40%;
    }

    @media (prefers-reduced-motion: reduce) {
        * { animation: none !important; transition: none !important; }
    }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# FUNGSI BANTU
# ---------------------------------------------------------------------------
def render_disclaimer() -> None:
    """Menampilkan disclaimer medis. Selalu dipanggil, tidak bisa dilewatkan."""
    st.markdown(
        """
        <div class="disclaimer">
          <p><strong>Alat bantu belajar, bukan diagnosis.</strong>
          Hasil di bawah ini dihasilkan dari model machine learning yang
          dilatih pada foto dermoskop. Model ini <strong>tidak divalidasi
          klinis</strong> dan bisa salah.</p>
          <p>Hasil <strong>"tidak ganas" tidak berarti lesi Anda aman</strong>.
          Segera periksakan ke dokter kulit bila lesi berubah bentuk, warna,
          atau ukuran, berdarah, terasa tidak nyaman, atau tidak pernah Anda punya
          sebelumnya. Diagnosis hanya bisa ditegakkan oleh dokter.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_guide() -> None:
    """Panduan memotret lesi agar hasil lebih bisa dipercaya."""
    st.markdown('<div class="eyebrow">Cara memotret</div>', unsafe_allow_html=True)
    items = [
        ("Dekat dan terfokus", "isi sebagian besar frame, jangan terlalu jauh"),
        ("Cahaya merata", "hindari bayangan keras dan pantulan cahaya"),
        ("Lesi di tengah", "letakkan lesi di tengah foto, jangan di sudut"),
        ("Tidak dig-filter", "kirim foto asli, jangan pakai filter atau editor"),
        ("Kulit terlihat jelas", "jika ada rambut, tata dulu atau gunakan cahaya lebih baik"),
    ]
    html = "".join(
        f'<div class="guide-item"><div class="guide-num">{i}</div>'
        f'<div class="guide-text"><strong>{judul}</strong><br>{ket}</div></div>'
        for i, (judul, ket) in enumerate(items, 1)
    )
    st.markdown(f'<div style="margin: 0.5rem 0 1rem;">{html}</div>',
                unsafe_allow_html=True)


def render_probability_bars(result: dict) -> None:
    """Menampilkan probabilitas tiap kelas sebagai bar horizontal."""
    malignant = set(result["malignant_classes"])
    classes = result["class_names"]
    dx_indo = result.get("dx_indo", {})
    probs = result["probabilities"]

    # Urutkan dari probabilitas tertinggi
    order = sorted(range(len(classes)), key=lambda i: probs[i], reverse=True)

    rows = []
    for i in order:
        cls = classes[i]
        is_mal = cls in malignant
        color = "var(--danger)" if is_mal else "var(--accent)"
        label = dx_indo.get(cls, cls)
        pct = probs[i] * 100
        rows.append(
            f'<div class="bar-row">'
            f'<div class="bar-label">{label}</div>'
            f'<div class="bar-track">'
            f'<div class="bar-fill" style="width:{pct:.1f}%;background:{color};">'
            f"</div></div>"
            f'<div class="bar-value">{pct:.1f}%</div>'
            f"</div>"
        )

    st.markdown("".join(rows), unsafe_allow_html=True)


def render_status(result: dict, uncertain: bool) -> None:
    """Menampilkan kotak status risiko dengan warna yang sesuai."""
    risk = result["risk_malignant"]
    thr = result["threshold"]
    dx_indo = result.get("dx_indo", {})
    pred = result["predicted_cls"]

    if uncertain:
        st.markdown(
            f"""
            <div class="status-box status-kabur">
              <div class="status-title">Hasil belum meyakinkan</div>
              <p class="status-body">
                Model tidak bisa memastikan apakah lesi ini menunjukkan
                ciri ganas. Persentase risiko ganas
                <strong>{risk * 100:.1f}%</strong> berada di area abu-abu
                (ambang <strong>{thr * 100:.0f}%</strong>). Dalam kondisi ini,
                foto Anda mungkin kurang ideal. <strong>Periksakan ke dokter
                kulit</strong> untuk memastikan.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    if risk >= thr:
        st.markdown(
            f"""
            <div class="status-box status-waspada">
              <div class="status-title">Ciri ganas terdeteksi</div>
              <p class="status-body">
                Model menemukan ciri yang sering berkaitan dengan lesi ganas.
                Perkiraan risiko ganas: <strong>{risk * 100:.1f}%</strong>
                (ambang {thr * 100:.0f}%). Diagnosis yang paling mungkin:
                <strong>{dx_indo.get(pred, pred)}</strong>.<br>
                <strong>Segera periksakan ke dokter kulit.</strong> Deteksi
                dini meningkatkan peluang kesembuhan secara signifikan.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="status-box status-aman">
              <div class="status-title">Tidak ditemukan ciri ganas</div>
              <p class="status-body">
                Perkiraan risiko ganas: <strong>{risk * 100:.1f}%</strong>
                (ambang {thr * 100:.0f}%). Kemungkinan besar:
                <strong>{dx_indo.get(pred, pred)}</strong>.<br>
                Ini <strong>bukan berarti lesi Anda pasti aman</strong>.
                Model bisa salah, terutama pada lesi yang tidak umum.
                Tetap periksakan ke dokter bila ada perubahan.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_model_info(predictor: dict) -> None:
    """Menampilkan informasi teknis model yang dipakai."""
    st.markdown('<div class="eyebrow">Model yang dipakai</div>',
                unsafe_allow_html=True)
    jenis = ("Klasik (HOG + histogram warna)"
             if predictor["model_type"] == "classic"
             else "Deep learning (CNN / MobileNetV2)")
    rows = [
        ("Model", predictor["model_name"]),
        ("Jenis", jenis),
        ("Ukuran input", f"{predictor['img_size']} x {predictor['img_size']} piksel"),
        ("Ambang risiko ganas", f"{predictor['threshold'] * 100:.0f}%"),
        ("Sumber", "Dataset HAM10000 (dermoskop), lisensi CC BY-NC 4.0"),
    ]
    html = "".join(
        f'<tr><td>{k}</td><td>{v}</td></tr>' for k, v in rows
    )
    st.markdown(f'<table class="meta-table">{html}</table>',
                unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# CONTOH GAMBAR
# ---------------------------------------------------------------------------
# Kenapa ada fitur ini?
#   Untuk demo dan untuk coba-coba: pengguna (atau dosen) tidak perlu
#   mencari file foto sendiri. Contoh diambil dari dataset yang dipakai
#   training, jadi hasilnya bisa dibandingkan dengan angka di laporan.
#
# Contoh-contoh ini BUKAN data training: file-nya disalin dari split
# test (HAM10000) dan dari dataset eksternal (PAD-UFES-20), dan hanya
# dipakai sebagai bahan demo, tidak pernah dipakai untuk fitting.
CONTOH_DIR = REPO_ROOT / "app" / "contoh"

CONTOH_TERSEDIA = {
    "contoh_melanoma.jpg": "Melanoma (dermoskop)",
    "contoh_bcc.jpg": "Karsinoma sel basal (dermoskop)",
    "contoh_nevus.jpg": "Nevus jinak (dermoskop)",
    "contoh_smartphone_mel.png": "Melanoma (foto smartphone)",
}


def daftar_contoh() -> list[tuple[str, str]]:
    """Mengembalikan daftar (nama_file, keterangan) contoh yang ada."""
    hasil = []
    for nama, ket in CONTOH_TERSEDIA.items():
        if (CONTOH_DIR / nama).exists():
            hasil.append((nama, ket))
    return hasil


def muat_contoh(nama_file: str) -> Image.Image:
    """Membaca satu file contoh menjadi gambar PIL."""
    return Image.open(CONTOH_DIR / nama_file).convert("RGB")


# ---------------------------------------------------------------------------
# MUAT MODEL (dengan cache agar tidak di-load tiap interaksi)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat model...")
def get_predictor():
    """Memuat model sekali per sesi browser, bukan tiap klik."""
    return PR.load_predictor()


# ---------------------------------------------------------------------------
# TAMPILKAN
# ---------------------------------------------------------------------------
st.markdown('<div class="eyebrow">Proyek Pembelajaran Mesin</div>',
            unsafe_allow_html=True)
st.title("Skrining Risiko Lesi Kulit dari Foto")
st.markdown(
    "Unggah foto lesi kulit untuk melihat perkiraan kelas diagnosis "
    "dan risiko keganasan. Model dilatih pada 10.015 foto dermoskop "
    "(HAM10000) dengan perbandingan SVM, Random Forest, CNN, dan "
    "MobileNetV2."
)

render_disclaimer()

# --- Sidebar: panduan memotret ---
with st.sidebar:
    st.markdown("### Panduan memotret")
    render_guide()

    st.markdown('<div class="eyebrow">Privasi</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="privacy-note"><p>Foto Anda diproses di memori selama '
        "sesi ini saja. Tidak disimpan di server, tidak dicatat, dan tidak "
        "dikirim ke pihak ketiga mana pun.</p></div>",
        unsafe_allow_html=True,
    )

# --- Kolom kiri: input, kanan: hasil ---
col_input, col_output = st.columns([1, 1.25], gap="large")

with col_input:
    st.markdown("### Unggah foto")
    uploaded = st.file_uploader(
        "Pilih foto lesi kulit (JPG atau PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        help="Ukuran maksimal 10 MB. Format JPG atau PNG.",
    )

    # --- Pilihan cepat: contoh gambar siap pakai ---
    #
    # Berguna untuk demo dan untuk-who yang mau coba-coba tanpa harus
    # mencari file foto sendiri. Contoh diambil dari dataset publik,
    # bukan dari data training.
    contoh = daftar_contoh()
    nama_contoh = None
    pakai_contoh = False
    if contoh:
        st.markdown('<div class="eyebrow">Atau pakai contoh</div>',
                    unsafe_allow_html=True)
        nama_contoh = st.selectbox(
            "Contoh gambar",
            [n for n, _ in contoh],
            format_func=lambda n: dict(contoh)[n],
        )
        pakai_contoh = st.checkbox("Pakai contoh ini", value=False)

    # Hasil akhir panel input:
    #   gambar_uggahan : gambar PIL siap diproses, atau None
    #   pesan_tunggu   : penjelasan kalau belum ada gambar yang bisa diproses
    gambar_uggahan = None
    pesan_tunggu = None

    if uploaded is None and not (pakai_contoh and nama_contoh):
        # KONDISI KOSONG: jelaskan apa yang harus dilakukan berikutnya
        pesan_tunggu = (
            "Belum ada foto. Pilih file di atas, atau pakai salah satu "
            "contoh gambar. Setelah dipilih, hasil analisis akan muncul "
            "di panel kanan."
        )
    elif uploaded is None:
        # Contoh gambar
        gambar_uggahan = muat_contoh(nama_contoh)
        st.image(
            gambar_uggahan,
            caption=f"Contoh: {dict(contoh)[nama_contoh]}",
            alt="Foto contoh lesi kulit yang dipakai untuk demo prediksi",
        )
    else:
        # Validasi ukuran file
        MAX_MB = 10
        size_mb = uploaded.size / (1024 * 1024)
        if size_mb > MAX_MB:
            st.error(
                f"Ukuran file {size_mb:.1f} MB melebihi batas {MAX_MB} MB. "
                "Gunakan foto dengan resolusi lebih rendah, atau crop foto "
                "sekitar lesi saja."
            )
            pesan_tunggu = (
                "Berkas terlalu besar, belum diproses. Pilih file yang lebih "
                "kecil atau pakai salah satu contoh gambar."
            )
        else:
            # Baca dan validasi file SEKALI di sini, lalu dipakai ulang
            # di kolom hasil. Validasi di kolom input penting karena
            # st.image() pada file yang bukan gambar akan melempar
            # UnidentifiedImageError, dan itu harus ditangani dengan pesan
            # yang ramah, bukan crash.
            #
            # CATATAN: `st.image(uploaded)` TIDAK bisa dipakai langsung.
            # Objek UploadedFile dari Streamlit tidak punya atribut
            # `.format` yang dibutuhkan Streamlit untuk mendeteksi tipe
            # MIME, dan hasilnya:
            #     AttributeError: 'UploadedFile' object has no attribute 'format'
            # Solusinya: baca byte-nya dulu, lalu pakai BytesIO.
            try:
                data_unggahan = uploaded.getvalue()
                gambar_uggahan = Image.open(io.BytesIO(data_unggahan)).convert("RGB")
                st.image(
                    io.BytesIO(data_unggahan),
                    caption=f"{uploaded.name} ({size_mb:.2f} MB)",
                    alt="Foto lesi kulit yang diunggah pengguna",
                )
            except UnidentifiedImageError:
                st.error(
                    "File yang dipilih bukan gambar yang bisa dibaca. "
                    "Pilih file JPG atau PNG berisi foto lesi kulit."
                )
                pesan_tunggu = (
                    "File yang dipilih bukan gambar. Pilih file JPG atau PNG "
                    "berisi foto lesi kulit, atau pakai contoh gambar."
                )
            except Exception as exc:  # noqa: BLE001
                st.error(
                    "File tidak bisa dibaca. Detail teknis: "
                    f"{exc}"
                )
                pesan_tunggu = (
                    "File tidak bisa dibaca. Pilih file JPG atau PNG berisi "
                    "foto lesi kulit, atau pakai contoh gambar."
                )

with col_output:
    st.markdown("### Hasil analisis")

    if gambar_uggahan is None:
        # KONDISI KOSONG di panel hasil
        st.markdown(
            '<p class="caption-muted">Hasil akan muncul di sini setelah '
            "foto dipilih.</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="privacy-note"><p>Yang akan ditampilkan: perkiraan '
            "kelas diagnosis, persentase risiko keganasan, dan probabilitas "
            "masing-masing kelas.</p></div>",
            unsafe_allow_html=True,
        )
    elif pesan_tunggu:
        st.markdown(
            f'<p class="caption-muted">{pesan_tunggu}</p>',
            unsafe_allow_html=True,
        )
    else:
        # Proses dengan penanganan error yang ramah
        try:
            with st.spinner("Menganalisis foto..."):
                predictor = get_predictor()

                # Gunakan gambar yang SUDAH divalidasi di kolom input
                # (dibaca satu kali saja, tidak dua kali).
                image = gambar_uggahan
                result = PR.predict(predictor, image)
                uncertain = PR.is_uncertain(result)

            # --- Status risiko ---
            render_status(result, uncertain)

            # --- Detail diagnosis ---
            st.markdown('<div class="eyebrow">Diagnosis paling mungkin</div>',
                        unsafe_allow_html=True)
            dx_indo = result.get("dx_indo", {})
            dx_full = result.get("dx_full_name", {})
            cls = result["predicted_cls"]
            top_prob = result["probabilities"].max() * 100
            st.markdown(
                f'<p style="font-size: 1.1rem; color: var(--ink); '
                f'margin: 0.1rem 0 0.2rem;"><strong>'
                f"{dx_indo.get(cls, cls)}</strong></p>"
                f'<p class="caption-muted" style="margin:0;">'
                f"{dx_full.get(cls, '')} (keyakinan model {top_prob:.1f}%)"
                "</p>",
                unsafe_allow_html=True,
            )

            # --- Probabilitas per kelas ---
            st.markdown('<div class="eyebrow">Probabilitas per kelas</div>',
                        unsafe_allow_html=True)
            render_probability_bars(result)

            # --- Penjelasan ---
            if uncertain:
                st.warning(
                    "Model tidak yakin. Periksa lagi kualitas foto, atau "
                    "langsung periksakan ke dokter kulit."
                )

            st.markdown(
                '<p class="caption-muted" style="margin-top:0.9rem;">'
                "Bar merah menandai kelas yang masuk kategori ganas "
                "(melanoma, karsinoma sel basal, keratosis atipis). "
                "Ambang risiko ganas "
                f"<strong>{result['threshold'] * 100:.0f}%</strong> "
                "dihitung dari data validasi, bukan dari data uji.</p>",
                unsafe_allow_html=True,
            )

        except FileNotFoundError as exc:
            # KONDISI ERROR: model belum tersedia
            st.error(
                f"Model belum siap dipakai. {exc}"
            )
        except UnidentifiedImageError:
            # File bukan gambar (misal .txt atau .pdf yang diunggah)
            st.error(
                "File yang dipilih bukan gambar yang bisa dibaca. "
                "Pilih file JPG atau PNG berisi foto lesi kulit."
            )
        except Exception as exc:  # noqa: BLE001
            st.error(
                "Foto tidak bisa diproses. Pastikan file adalah gambar "
                f"JPG atau PNG yang valid. Detail teknis: {exc}"
            )

# --- Panel bawah: informasi model ---
st.markdown("---")
left, right = st.columns([1, 1.25], gap="large")

with left:
    render_guide()

with right:
    try:
        predictor = get_predictor()
        render_model_info(predictor)
    except Exception:  # noqa: BLE001
        st.markdown(
            '<p class="caption-muted">Informasi model belum tersedia '
            "karena model belum dimuat.</p>",
            unsafe_allow_html=True,
        )

st.markdown("---")
st.markdown(
    '<p class="caption-muted">Dibangun untuk tugas akhir Mata Kuliah '
    "Pembelajaran Mesin (ML2026). Dataset HAM10000: Tschandl dkk. (2018), "
    '<em>Scientific Data</em> 5:180214, lisensi CC BY-NC 4.0. '
    "Aplikasi ini untuk keperluan pendidikan, bukan penggunaan medis."
    "</p>",
    unsafe_allow_html=True,
)
