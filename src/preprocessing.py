"""
preprocessing.py
================
Preprocessing citra untuk model DEEP LEARNING (CNN sederhana dan MobileNetV2).

Prinsip utama modul ini
------------------------
Satu fungsi, dipakai bersama oleh notebook training DAN aplikasi web.
Kalau preprocessing di dua tempat ditulis terpisah, cepat atau lambat
nilainya akan berbeda, dan hasil prediksi aplikasi bisa tidak cocok dengan
hasil training. Ini bukan hal kecil: kalau training pakai normalisasi A
lalu aplikasi pakai normalisasi B, akurasi model di aplikasi bisa anjlok
dari 75% jadi 40% tanpa ada yang salah secara kode.

Aturan normalisasi penting (sering jadi sumber bug)
--------------------------------------------------
- CNN sederhana kita-built sendiri dari nol: normalisasi 1/255 (nilai 0..1)
  dilakukan oleh layer Rescaling DI DALAM model.
- MobileNetV2 memakai pretrained weight ImageNet: normalisasinya HARUS
  persis fungsi bawaan `mobilenet_v2.preprocess_input` (nilai -1..1).
  Jangan menambah Rescaling 1/255 lagi, karena itu "normalisasi dua kali"
  dan membuat input model berada di rentang yang tidak pernah dilihat
  saat pretraining.

Modul ini juga menyediakan augmentasi. Augmentasi HANYA dijalankan pada
data latih (setelah split), karena augmentasi pada data validasi/test
merusak pengukuran performa.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from . import config as C


# ---------------------------------------------------------------------------
# 1. RESIZE
# ---------------------------------------------------------------------------
def resize_image(img, size: int):
    """
    Mengubah ukuran gambar ke (size x size).

    Kenapa BILINEAR dan bukan LANCZOS?
      Lanczos lebih tajam tapi butuh komputasi lebih mahal dan bisa
      menghasilkan ringing (garis-garis tipis) di tepi. Untuk input CNN
      yang akan di-augmentasi lagi, bilinear sudah cukup dan jauh lebih
      cepat. Aplikasi web pun memakai fungsi yang sama, jadi konsisten.
    """
    if isinstance(img, np.ndarray):
        img = tf.convert_to_tensor(img)
    return tf.image.resize(img, (size, size), method="bilinear")


# ---------------------------------------------------------------------------
# 2. NORMALISASI (dipisah agar tidak salah pasang)
# ---------------------------------------------------------------------------
def normalize_for_cnn(img):
    """
    Normalisasi untuk CNN yang dilatih dari nol.

    Catatan: Model Sequential CNN kita punya layer Rescaling(1/255) sebagai
    layer PERTAMA di dalam model. Jadi fungsi ini HANYA mengubah dtype
    uint8 -> float32 dan mengembalikan nilai 0..255. Normalisasi ke 0..1
    dilakukan oleh layer di dalam model.

    Kenapa dipisah seperti ini?
      Supaya preprocessing (resize, dtype) sama persis antara training dan
      inference, sementara normalisasi tetap milik model (tersimpan bersama
      model .keras, jadi tidak mungkin lupa).
    """
    return tf.cast(img, tf.float32)


def normalize_for_mobilenet(img):
    """
    Normalisasi untuk MobileNetV2 pretrained.

    `tf.keras.applications.mobilenet_v2.preprocess_input` mengubah nilai
    0..255 menjadi -1..1 dengan rumus: (x / 127.5) - 1.

    PENTING: fungsi ini harus dipanggil PADA TENSOR RGB yang SUDAH di-resize
    ke 224x224, dan TIDAK boleh digabung dengan Rescaling(1/255).
    """
    return tf.keras.applications.mobilenet_v2.preprocess_input(
        tf.cast(img, tf.float32)
    )


# ---------------------------------------------------------------------------
# 3. AUGMENTASI (hanya untuk data latih)
# ---------------------------------------------------------------------------
def build_train_augmentation(smartphone_style: bool = False) -> tf.keras.Sequential:
    """
    Membuat pipeline augmentasi sebagai layer Keras.

    Param smartphone_style:
      False -> augmentasi standar (untuk CNN sederhana dan tahap 1 MobileNet)
      True  -> augmentasi yang meniru foto smartphone (untuk tahap 2 MobileNet)

    Kenapa augmentasi?
      Dataset kita hanya 10.015 gambar dengan 7 kelas yang tidak seimbang.
      Tanpa augmentasi, model bisa saja menghafal gambar training (overfit)
      dan gagal pada gambar baru. Augmentasi membuat model melihat variasi
      dari gambar yang sama, sehingga dia wajib belajar "ciri umum" bukan
      "piksel spesifik".

    Kenapa augmentasi harus DI DALAM model (bukan di luar)?
      Kalau augmentasi dilakukan di luar (di pipeline data terpisah), maka
      saat inferensi di aplikasi kita harus ingat menyalin pipeline itu
      secara manual. Kalau augmentasi jadi bagian model, ia otomatis mati
      saat model dalam mode inference. Kerisauan ini hilang total.
    """
    layers = [
        # Membalik kiri-kanan: lesi bisa difoto dari sudut kiri atau kanan.
        # Wolf-Hirschhorn/pola asimetrik akan berubah arah, tapi untuk
        # malignant melanoma asimetri justru ciri penting. Untuk dataset
        # ini horizontal flip AMAN dan sudah lazim dipakai di paper.
        tf.keras.layers.RandomFlip("horizontal"),

        # Rotasi kecil: foto lesi bisa sedikit miring saat difoto.
        # Batas 10 derajat cukup untuk meniru kemiringan realistis tanpa
        # membuat lesi keluar dari frame.
        tf.keras.layers.RandomRotation(0.10),

        # Zoom acak: jarak kamera tidak selalu sama.
        tf.keras.layers.RandomZoom(height_factor=(-0.10, 0.10),
                                    width_factor=(-0.10, 0.10)),
    ]

    if smartphone_style:
        # Augmentasi khusus foto smartphone, dipakai pada tahap fine-tuning
        # MobileNetV2 dan untuk simulasi domain shift (uji PAD-UFES-20).
        #
        # TUJUANNYA: dataset HAM10000 adalah foto DERMOSKOPI (kamera khusus
        # mikroskop, cahaya melingkar, resolusi tinggi). Aplikasi kita
        # prediksi foto SMARTPHONE. Kalau model cuma pernah melihat
        # dermoskopi, ia akan gagal total pada foto smartphone.
        # Augmentasi ini membuat model belajar "ciri lesi" dan bukan
        # "ciri alat foto dermoskop".
        layers += [
            # Variasi brightness: cahaya kamar berbeda-beda.
            tf.keras.layers.RandomBrightness(0.20),

            # Variasi contrast:ISP smartphone dan pencahayaan bervariasi.
            tf.keras.layers.RandomContrast(0.20),

            # Variasi warna sedikit:ISP smartphone punya white balance
            # berbeda. Range kecil (0.1) supaya warna kulit tetap realistis.
            tf.keras.layers.RandomHue(factor=0.10),

            # Kompresi JPEG: foto smartphone hampir selalu terkompresi JPEG.
            # Ini menghasilkan artefak blok dan blur yang TIDAK ada di foto
            # dermoskopi. Diimplementasikan sebagai layer kustom di bawah.
            JPEGCompressionLayer(quality=75),
        ]

    return tf.keras.Sequential(layers, name="augmentation")


class JPEGCompressionLayer(tf.keras.layers.Layer):
    """
    Layer kustom yang meniru kompresi JPEG pada gambar.

    Kenapa perlu?
      Hampir semua foto smartphone disimpan dalam format JPEG dengan
      kualitas 60-95. Kompresi ini menghapus frekuensi tinggi dan
      membentuk blok-blok kecil. Model yang hanya dilatih pada foto
      dermoskopi akan kaget saat melihat gambar smartphone.

    Cara kerja:
      1. Ubah tensor float (0..255) menjadi uint8, sama seperti file JPEG
      2. Encode ke JPEG (kualitas tetap)
      3. Decode kembali ke float 0..255

    Kenapa implementasi ini (dan bukan `tf.py_function`)?
      Versi pertama memanggil PIL lewat `tf.py_function`. Hasilnya
      **DEADLOCK**: seluruh thread `tf.data` tertidur di `futex_do_wait`
      dan training berhenti total (terlihat di profiler: 172 thread
      sleeping, GPU 0%). `tf.py_function` di dalam `tf.data.map` dengan
      `num_parallel_calls=AUTOTUNE` ternyata tidak aman.

      Versi ini memakai `tf.io.encode_jpeg` / `tf.io.decode_jpeg` yang
      **murni TensorFlow**, jadi aman di graph mode dan bisa berjalan
      paralel.

    Catatan tentang kualitas yang tetap:
      `tf.io.encode_jpeg` menuntut `quality` berupa bilangan bulat statis,
      bukan tensor. Karena itu kualitas TIDAK bisa diacak per gambar di
      dalam graph. Yang diacak di layer lain adalah cahaya dan kontras,
      yang sudah cukup meniru variasi smartphone.

    Layer ini hanya berjalan saat training (mode inference passing through).
    """

    def __init__(self, quality: int = 75, **kwargs):
        super().__init__(**kwargs)
        self.quality = quality

    def call(self, inputs):
        """
        Melakukan encode-decode JPEG pada setiap gambar.

        PENTING: layer ini dipanggil dari dua tempat dengan bentuk berbeda:
          - Di dalam `tf.data.map`, setiap elemen adalah SATU gambar
            (H, W, 3) tanpa dimensi batch.
          - Saat dipanggil langsung (uji), bentuknya (B, H, W, 3).

        `tf.map_fn` memotong sumbu pertama. Kalau inputnya (H, W, 3),
        elemen yang dipetakan menjadi (W, 3) dan `encode_jpeg` gagal
        dengan "Shape must be rank 3 but is rank 2".

        Karena itu kita deteksi kasus tanpa batch, tambah dimensi batch,
        petakan, lalu kembalikan bentuk aslinya.
        """
        tanpa_batch = len(inputs.shape) == 3
        if tanpa_batch:
            inputs = tf.expand_dims(inputs, axis=0)

        img_u8 = tf.cast(tf.clip_by_value(inputs, 0.0, 255.0), tf.uint8)

        def _roundtrip(img):
            encoded = tf.io.encode_jpeg(img, quality=self.quality)
            decoded = tf.io.decode_jpeg(encoded, channels=3)
            return tf.cast(decoded, tf.float32)

        out = tf.map_fn(_roundtrip, img_u8, fn_output_signature=tf.float32)

        if tanpa_batch:
            out = tf.squeeze(out, axis=0)
        return out

    def compute_output_shape(self, input_shape):
        """Bentuk output sama dengan input (kompresi JPEG tidak mengubah dimensi)."""
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({"quality": self.quality})
        return config


# ---------------------------------------------------------------------------
# 4. PEMBUAT PIPELINE(tf.data)
# ---------------------------------------------------------------------------
def make_tf_dataset(
    dataframe,
    img_size: int,
    batch_size: int = C.BATCH_SIZE,
    training: bool = False,
    smartphone_style: bool = False,
    seed: int = C.SEED,
):
    """
    Membuat objek tf.data.Dataset dari DataFrame.

    Parameter:
      dataframe         : DataFrame berisi kolom image_path dan dx_idx
      img_size          : ukuran sisi gambar (128 atau 224)
      training          : True untuk data latih (shuffle + augmentasi)
      smartphone_style  : True untuk augmentasi gaya smartphone
      seed              : untuk hasil yang reproducible

    APA YANG TIDAK DIKERJAKAN DI SINI (dan kenapa)
    ----------------------------------------------
    Normalisasi TIDAK dilakukan di pipeline ini. Normalisasi adalah bagian
    dari MODEL (layer Rescaling untuk CNN, layer preprocess_input untuk
    MobileNetV2). Alasannya:

      - Model .keras yang disimpan di folder models/ akan berisi normalisasi
        di dalamnya, sehingga aplikasi web TIDAK PERLU ingat-ingat aturan
        normalisasi. Aturan tidak bisa "lupa" karena dia ikut tersimpan.
      - Isi model hanya memakai layer Keras bawaan (tidak ada layer kustom),
        jadi file .keras bisa dimuat di server mana pun tanpa mendaftarkan
        custom_objects. Ini penting untuk deployment ke Streamlit Cloud.

    APA YANG DIKERJAKAN DI SINI
    ---------------------------
      - Decode file JPEG menjadi tensor
      - Resize ke (img_size, img_size)
      - Ubah dtype uint8 -> float32 (nilai 0..255)
      - Augmentasi (HANYA kalau training=True)
      - Shuffle (HANYA kalau training=True)

    Kenapa pakai tf.data dan bukan load semua gambar ke RAM?
      10.015 gambar pada 224x224 RGB = 1.5 GB kalau dimuat semua ke memori
      (setelah resize). Kalau ditambah augmentasi, RAM akan habis.
      tf.data memuat gambar per batch saat dibutuhkan, sehingga RAM tetap
      ringan dan bisa dipakai di laptop dengan RAM 8 GB.

    Kenapa shuffle hanya untuk training?
      Data validasi dan test harus dalam urutan yang stabil supaya
      perbandingan antar model adil. Kalau di-shuffle, urutan berubah
      tiap run dan hasil bisa sedikit berbeda antar epoch.
    """
    paths = dataframe["image_path"].astype(str).values
    labels = dataframe["dx_idx"].astype("int32").values

    def _load(path, label):
        # tf.io.read_file membaca file sebagai byte mentah (cepat, tidak
        # butuh library PIL di dalam graph TensorFlow).
        raw = tf.io.read_file(path)
        img = tf.io.decode_image(raw, channels=3, expand_animations=False)
        img = tf.cast(img, tf.float32)
        img = resize_image(img, img_size)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        ds = ds.shuffle(
            buffer_size=min(4096, len(paths)),
            seed=seed,
            reshuffle_each_iteration=True,
        )
        aug = build_train_augmentation(smartphone_style=smartphone_style)
        ds = ds.map(
            lambda img, label: (aug(img, training=True), label),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        ds = ds.repeat()  # agar epoch tidak berhenti lebih dulu dari 1 epoch
    else:
        # CATATAN PENTING: untuk evaluasi, dataset TIDAK boleh di-repeat().
        # Kalau di-repeat, jumlah prediksi jadi lebih banyak dari jumlah
        # label, sehingga perbandingan y_true vs y_pred tidak bisa dilakukan
        # (dan metriknya jadi tidak valid).
        # Alasan repeat() dipakai di training: jumlah step per epoch
        # (ceil(n/batch)) bisa lebih besar dari jumlah data, jadi epoch
        # akan berhenti di tengah jalan kalau tidak di-repeat.
        # Pada evaluasi kita memanggil predict() dengan steps=ceil(n/batch),
        # jadi repeat() tidak diperlukan sama sekali.
        pass

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def count_steps(dataframe, batch_size: int = C.BATCH_SIZE) -> int:
    """Menghitung jumlah step per epoch (jumlah gambar / batch size)."""
    return int(np.ceil(len(dataframe) / batch_size))
