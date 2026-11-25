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
  4. Disclaimer selalu ada di bagian bawah halaman.
  5. Foto pengguna TIDAK disimpan dan TIDAK dikirim ke pihak mana pun.

CATATAN TATA LETAK
------------------
Bagian ini hanya mengatur tampilan (judul, kartu, tab, grafik, warna).
Alur prediksi, ambang, dan pemanggilan src.predictor tidak diubah: foto
tetap divalidasi sekali, lalu objek PIL yang sama dipakai untuk prediksi.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import plotly.graph_objects as go
import streamlit as st
import streamlit_shadcn_ui as ui
from PIL import Image, UnidentifiedImageError

from src import config as C
from src import predictor as PR

# ---------------------------------------------------------------------------
# KONFIGURASI HALAMAN
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DermAssist Clinical",
    page_icon=":material/biotech:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# DESAIN: CSS MINIMAL
# ---------------------------------------------------------------------------
# Hanya dua aturan. Warna, radius, dan tipografi diatur lewat
# .streamlit/config.toml supaya tahan upgrade Streamlit; CSS hanya dipakai
# untuk dua hal yang belum punya opsi resmi:
#   1. menyembunyikan tombol Deploy saat presentasi,
#   2. merapatkan jarak atas konten (default 6rem terlalu lega).
st.markdown(
    """
    <style>
        [data-testid="stAppDeployButton"] { display: none; }
        [data-testid="stMainBlockContainer"] { padding-top: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# FUNGSI BANTU
# ---------------------------------------------------------------------------
def render_disclaimer() -> None:
    """
    Satu disclaimer ringkas di bagian bawah halaman.

    Teksnya tetap menyebut batas alat ini: bukan diagnosis, tidak
    divalidasi klinis, dan hasil "tidak ganas" bukan jaminan aman.
    """
    st.caption(
        "**Alat bantu belajar, bukan diagnosis.** Model dilatih pada foto "
        "dermoskop dan tidak divalidasi klinis. Hasil \"tidak ganas\" tidak "
        "berarti lesi Anda aman: periksakan ke dokter kulit bila lesi berubah "
        "bentuk, warna, atau ukuran, berdarah, atau terasa tidak nyaman."
    )


def render_guide() -> None:
    """
    Panduan memotret, hanya di sidebar.

    Sengaja berupa daftar poin dengan pemimpin tebal, bukan lingkaran
    bernomor: lebih ringkas, tetap terbaca, dan tidak bersaing dengan
    judul halaman.
    """
    st.markdown(
        "- **Dekat dan terfokus**: isi sebagian besar frame, jangan "
        "terlalu jauh.\n"
        "- **Cahaya merata**: hindari bayangan keras dan pantulan "
        "cahaya.\n"
        "- **Lesi di tengah**: jangan letakkan lesi di sudut foto.\n"
        "- **Tanpa filter**: kirim foto asli, tanpa editor atau filter.\n"
        "- **Kulit terlihat jelas**: tata rambut atau tambah cahaya bila "
        "perlu."
    )


def render_probability_chart(result: dict) -> None:
    """
    Grafik batang horizontal probabilitas tiap kelas.

    Warna membawa makna, bukan hiasan: teal untuk kelas jinak dan rose
    untuk kelas ganas. Gridline dimatikan supaya yang menonjol hanya
    panjang batang dan angkanya.
    """
    malignant = set(result["malignant_classes"])
    classes = result["class_names"]
    dx_indo = result.get("dx_indo", {})
    probs = result["probabilities"]

    # Plotly menggambar batang horizontal dari bawah ke atas, jadi urutan
    # probabilitas terkecil diletakkan lebih dulu agar nilai terbesar
    # muncul paling atas.
    order = sorted(range(len(classes)), key=lambda i: probs[i])
    labels = [dx_indo.get(classes[i], classes[i]) for i in order]
    values = [float(probs[i]) for i in order]
    colors = [
        "#E11D48" if classes[i] in malignant else "#0D9488"
        for i in order
    ]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker={"color": colors, "line": {"width": 0}},
            text=[f"{v * 100:.1f}%" for v in values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:.1%}<extra></extra>",
        )
    )
    fig.update_layout(
        height=310,
        margin={"l": 0, "r": 34, "t": 4, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#1C1917", "size": 12},
        showlegend=False,
        bargap=0.38,
        xaxis={
            "visible": False,
            "range": [0, 1.15],
            "fixedrange": True,
        },
        yaxis={
            "showgrid": False,
            "zeroline": False,
            "showline": False,
            "ticks": "",
            "automargin": True,
            "ticklabelstandoff": 8,
            "fixedrange": True,
        },
    )

    st.plotly_chart(
        fig,
        theme=None,
        width="stretch",
        config={"displayModeBar": False},
        key="grafik_probabilitas",
    )


def render_metric_cards(result: dict, uncertain: bool) -> None:
    """Dua angka utama: kelas prediksi dan risiko ganas dibanding ambang."""
    risk = result["risk_malignant"]
    thr = result["threshold"]
    dx_indo = result.get("dx_indo", {})
    dx_full = result.get("dx_full_name", {})
    cls = result["predicted_cls"]
    top_prob = result["probabilities"].max() * 100

    if uncertain:
        keterangan_risiko = f"Ambang {thr * 100:.0f}% · area abu-abu"
    elif risk >= thr:
        keterangan_risiko = f"Ambang {thr * 100:.0f}% · di atas ambang"
    else:
        keterangan_risiko = f"Ambang {thr * 100:.0f}% · di bawah ambang"

    kiri, kanan = st.columns(2, gap="small")
    with kiri:
        ui.metric_card(
            label="Prediksi",
            value=dx_indo.get(cls, cls),
            description=(
                f"{dx_full.get(cls, '')} · keyakinan model {top_prob:.1f}%"
            ),
            key="kartu_prediksi",
        )
    with kanan:
        ui.metric_card(
            label="Risiko ganas",
            value=f"{risk * 100:.1f}%",
            description=keterangan_risiko,
            key="kartu_risiko",
        )


def render_status(result: dict, uncertain: bool) -> None:
    """
    Lencana status dan satu paragraf penjelasan.

    Tidak memakai kotak berbingkai kiri tebal: status sudah dibawa oleh
    lencana dan angka di kartu metrik, jadi teks cukup menjelaskan
    artinya dan langkah berikutnya.
    """
    risk = result["risk_malignant"]
    thr = result["threshold"]
    dx_indo = result.get("dx_indo", {})
    pred = result["predicted_cls"]

    if uncertain:
        ui.badges(
            [("Perlu konfirmasi", "secondary")],
            key="lencana_status",
        )
        st.markdown(
            f"Hasil belum meyakinkan. Risiko ganas **{risk * 100:.1f}%** "
            f"berada di area abu-abu di sekitar ambang "
            f"**{thr * 100:.0f}%**. Foto mungkin kurang ideal. "
            "**Periksakan ke dokter kulit** untuk memastikan."
        )
        return

    if risk >= thr:
        ui.badges(
            [("Risiko ganas", "destructive"), ("Segera periksa", "outline")],
            key="lencana_status",
        )
        st.markdown(
            f"Ciri ganas terdeteksi. Risiko ganas **{risk * 100:.1f}%** "
            f"berada di atas ambang **{thr * 100:.0f}%**; diagnosis paling "
            f"mungkin **{dx_indo.get(pred, pred)}**. "
            "**Segera periksakan ke dokter kulit.** Deteksi dini "
            "meningkatkan peluang kesembuhan."
        )
    else:
        ui.badges(
            [("Risiko rendah", "secondary")],
            key="lencana_status",
        )
        st.markdown(
            f"Tidak ditemukan ciri ganas. Risiko ganas "
            f"**{risk * 100:.1f}%** berada di bawah ambang "
            f"**{thr * 100:.0f}%**; kemungkinan besar "
            f"**{dx_indo.get(pred, pred)}**. Ini bukan berarti lesi Anda "
            "pasti aman: model bisa salah, terutama pada lesi yang tidak umum."
        )


def render_model_info(predictor: dict) -> None:
    """Informasi teknis model, ditampilkan di dalam expander."""
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
    st.markdown("\n".join(f"- **{k}:** {v}" for k, v in rows))


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
# Header tiga baris: nama produk, penjelasan singkat, dan label kecil.
# Lencana versi diletakkan di kolom kanan supaya tidak menambah tinggi.
kepala_kiri, kepala_kanan = st.columns([4, 1], vertical_alignment="center")
with kepala_kiri:
    st.markdown(
        '<h1 style="color:#0F766E; font-size:2.3rem; line-height:1.15; '
        'margin:0 0 0.15rem;">DermAssist Clinical</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Alat bantu skrining awal berbasis pembelajaran mesin untuk lesi kulit."
    )
    st.caption("Skrining Risiko Lesi Kulit dari Foto")
with kepala_kanan:
    ui.badges([("Versi 1.0", "outline")], key="lencana_versi")

# --- Sidebar: panduan memotret dan privasi ---
with st.sidebar:
    st.markdown("### Cara Memotret")
    render_guide()

    st.divider()
    st.markdown("**Privasi**")
    st.caption(
        "Foto Anda diproses di memori selama sesi ini saja. Tidak disimpan "
        "di server, tidak dicatat, dan tidak dikirim ke pihak ketiga."
    )

# --- Dua kartu: input di kiri, hasil di kanan ---
col_input, col_output = st.columns([1, 1.25], gap="large")

with col_input:
    with st.container(border=True):
        st.subheader("Foto lesi")

        # Model hanya dilatih pada foto lesi dari dekat. Tanpa batas ini,
        # gambar di luar domain (wajah, tangkapan layar) tetap dipaksa
        # menjadi salah satu dari 7 kelas dan tampak seperti hasil yakin.
        st.caption(
            "**Input yang didukung:** foto lesi kulit dari dekat. Foto wajah, "
            "tangkapan layar, atau gambar tanpa lesi yang jelas dapat "
            "menghasilkan prediksi yang menyesatkan. Contoh foto yang "
            "sesuai ada di tab Gunakan Sampel."
        )

        uploaded = None
        nama_contoh = None
        pakai_contoh = False
        contoh = daftar_contoh()

        tab_unggah, tab_contoh = st.tabs(["Unggah Foto", "Gunakan Sampel"])

        with tab_unggah:
            uploaded = st.file_uploader(
                "Pilih foto lesi kulit (JPG atau PNG)",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=False,
                help="Ukuran maksimal 10 MB. Format JPG atau PNG.",
            )

        with tab_contoh:
            # Checkbox dipertahankan supaya perilakunya sama dengan versi
            # sebelumnya: unggahan selalu menang bila ada file terpilih.
            if contoh:
                nama_contoh = st.selectbox(
                    "Contoh gambar",
                    [n for n, _ in contoh],
                    format_func=lambda n: dict(contoh)[n],
                    key="pilih_contoh",
                )
                pakai_contoh = st.checkbox(
                    "Pakai contoh ini", value=False, key="pakai_contoh"
                )
            else:
                st.caption("Contoh gambar belum tersedia.")

        # Hasil akhir panel input:
        #   gambar_uggahan : gambar PIL siap diproses, atau None
        #   pesan_tunggu   : penjelasan kalau ada file yang gagal dibaca
        gambar_uggahan = None
        pesan_tunggu = None
        keterangan_gambar = None

        if uploaded is None and not (pakai_contoh and nama_contoh):
            # KONDISI KOSONG: panel hasil yang menjelaskan langkah berikutnya.
            pass
        elif uploaded is None:
            # Contoh gambar
            gambar_uggahan = muat_contoh(nama_contoh)
            keterangan_gambar = f"Contoh: {dict(contoh)[nama_contoh]}"
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
                    gambar_uggahan = Image.open(
                        io.BytesIO(data_unggahan)
                    ).convert("RGB")
                    keterangan_gambar = f"{uploaded.name} ({size_mb:.2f} MB)"
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

        # Pratinjau dibatasi lebarnya supaya foto tidak mendominasi kartu.
        if gambar_uggahan is not None:
            st.image(
                gambar_uggahan,
                width=360,
                alt="Pratinjau foto lesi kulit yang akan dianalisis",
            )
            if keterangan_gambar:
                st.caption(keterangan_gambar)

with col_output:
    with st.container(border=True):
        st.subheader("Hasil analisis")

        if gambar_uggahan is None and pesan_tunggu is None:
            # KONDISI KOSONG di panel hasil
            st.caption("Belum ada foto yang dianalisis.")
            st.caption(
                "Pilih foto di tab Unggah Foto, atau coba contoh di tab "
                "Gunakan Sampel."
            )
        elif pesan_tunggu:
            st.caption(pesan_tunggu)
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

                render_metric_cards(result, uncertain)
                render_status(result, uncertain)

                st.markdown("**Probabilitas per kelas**")
                render_probability_chart(result)

                st.caption(
                    "Merah menandai kelas ganas (melanoma, karsinoma sel "
                    "basal, keratosis atipis); teal menandai kelas jinak. "
                    "Ambang risiko ganas "
                    f"{result['threshold'] * 100:.0f}% dihitung dari data "
                    "validasi, bukan dari data uji."
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

# --- Panel bawah: informasi model dan disclaimer ---
st.divider()
with st.expander("Tentang model"):
    try:
        predictor = get_predictor()
        render_model_info(predictor)
    except Exception:  # noqa: BLE001
        st.caption(
            "Informasi model belum tersedia karena model belum dimuat."
        )

st.caption(
    "Dibangun untuk tugas akhir Mata Kuliah Pembelajaran Mesin (ML2026). "
    "Dataset HAM10000: Tschandl dkk. (2018), Scientific Data 5:180161, "
    "lisensi CC BY-NC 4.0."
)
render_disclaimer()
