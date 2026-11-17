"""
modeling.py
===========
Pembuatan dan training model deep learning.

Isi modul:
  1. build_simple_cnn()    -> CNN dari nol (arsitektur sederhana)
  2. build_mobilenetv2_stage1() -> Transfer learning, backbone dibekukan
  3. build_mobilenetv2_stage2() -> Fine-tuning sebagian lapisan atas
  4. train_model()          -> Training loop + EarlyStopping
  5. save_model()           -> Menyimpan model .keras + label_map + config

Prinsip penting yang dipegang modul ini
----------------------------------------
A. Normalisasi ADALAH BAGIAN DARI MODEL, bukan bagian dari pipeline data.
   - CNN  : layer Rescaling(1/255) sebagai layer pertama
   - MobileNetV2 : layer preprocess_input sebagai layer pertama
   Kenapa? Supaya file .keras yang dipakai aplikasi web SUDAH membawa aturan
   normalisasinya. Aplikasi tidak perlu mengingat apa pun, dan tidak mungkin
   salah. Yang lebih penting: tidak ada custom layer di dalam model, jadi
   file .keras bisa dimuat di server mana pun tanpa mendaftarkan
   custom_objects. Ini syarat agar deploy ke Streamlit Cloud tidak gagal.

B. Augmentasi ada DI DALAM MODEL, dengan training=True/False.
  (tf.keras.layers.RandomFlip dst. otomatis menjadi identitas saat mode
   inferensi). Kita TETAP memakai pipeline augmentasi terpisah di
   `preprocessing.py` untuk tahap fine-tuning, karena di sana kita butuh
   augmentasi gaya smartphone yang lebih berat.

C. BatchNormalization di backbone MobileNetV2 harus tetap dalam mode
   inference saat fine-tuning. Alasannya dijelaskan di fungsi stage 2.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from . import config as C


# ---------------------------------------------------------------------------
# 1. CNN SEDERHANA
# ---------------------------------------------------------------------------
def build_simple_cnn(
    num_classes: int = len(C.CLASSES),
    img_size: int = C.IMG_SIZE_CNN,
    use_global_average_pooling: bool = True,
    name: str = "cnn_sederhana",
    width_multiplier: int = 1,
) -> tf.keras.Model:
    """
    Membangun CNN sederhana dari nol.

    Arsitektur (sesuai spesifikasi project):
        Input (img_size x img_size x 3)
          -> Rescaling(1/255)
          -> Conv2D(32, 3x3, relu, same)  -> MaxPooling2D(2)
          -> Conv2D(64, 3x3, relu, same)  -> MaxPooling2D(2)
          -> [GlobalAveragePooling2D ATAU Flatten]
          -> Dense(num_classes, softmax)

    Kenapa GlobalAveragePooling2D (GAP) vs Flatten?
      - Flatten: mengubah peta fitur 2D (misal 32x32x64 = 65.536 angka)
        menjadi 1 vektor panjang, lalu Dense. Parameternya SANGAT banyak
        (65.536 x 64 = 4,2 juta parameter pada Dense pertama saja).
        Risiko overfit tinggi pada dataset 10.015 gambar.
      - GAP: untuk setiap kanal, ambil rata-ratanya, sehingga 32x32x64
        menjadi 64 angka. Dense pertama jadi 64 x 64 = 4.096 parameter.
        Jauh lebih sedikit, dan otomatis "translation invariant":
        lesi yang bergeser sedikit posisinya tetap dikenali.

      Kita akan MEMBANDINGKAN keduanya di Fase 4 (eksperimen wajib),
      bukan asal pilih salah satu.

    Kenapa max pooling (bukan strided conv)?
      Max pooling memilih informasi paling kuat di setiap area (misal
      "ada tepi tajam di sini"), sementara rata-rata bisa mengaburkan
      informasi. Untuk lesi kulit, tepi dan pola lokal itu penting.

    ### Kenapa ada BatchNormalization dan `width_multiplier`?
    #
    # Arsitektur di atas adalah spesifikasi minimum dari project. Kami
    # melakukan eksperimen (lihat docs/CATATAN_TEMUAN_DIAGNOSTIK.md bagian 7)
    # dan menemukan dua hal:
    #
    # 1. **Tanpa BatchNormalization**, model sulit dilatih pada dataset
    #    sekecil ini. Distribusi aktivasi tiap layer bergeser terus selama
    #    training, sehingga optimizer selalu mengejar target yang bergerak.
    #    BatchNormalization menormalkan aktivasi tiap layer, sehingga
    #    optimasi jauh lebih stabil.
    #
    # 2. **Menambah lebar (jumlah filter) menaikkan performa nyata.**
    #    Varian 32-64 hanya memprediksi 3 dari 7 kelas (F1 macro 0,140),
    #    sedangkan varian 32-64-128 dengan BatchNormalization memprediksi
    #    SEMUA 7 kelas (F1 macro 0,254).
    #
    # `width_multiplier` dipakai supaya perbandingan ini tetap bisa
    # dilakukan dengan satu fungsi: 1 = spesifikasi project, 2 = varian
    # lebih lebar (dipakai di notebook sebagai pembanding).
    """
    base = 32 * width_multiplier

    model = tf.keras.Sequential([
        # Layer pertama: ubah nilai 0..255 menjadi 0..1.
        tf.keras.layers.Input(shape=(img_size, img_size, 3)),
        tf.keras.layers.Rescaling(1.0 / 255.0),

        # Blok konvolusi 1
        tf.keras.layers.Conv2D(base, (3, 3), activation="relu",
                               padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Blok konvolusi 2
        tf.keras.layers.Conv2D(base * 2, (3, 3), activation="relu",
                               padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
    ], name=name)

    if width_multiplier > 1:
        # Blok konvolusi 3 (hanya di varian lebar)
        model.add(tf.keras.layers.Conv2D(base * 4, (3, 3), activation="relu",
                                         padding="same"))
        model.add(tf.keras.layers.BatchNormalization())

    if use_global_average_pooling:
        model.add(tf.keras.layers.GlobalAveragePooling2D())
    else:
        model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))
    return model


# ---------------------------------------------------------------------------
# 2. MOBILENETV2 - TAHAP 1 (backbone dibekukan)
# ---------------------------------------------------------------------------
def build_mobilenetv2_stage1(
    num_classes: int = len(C.CLASSES),
    img_size: int = C.IMG_SIZE_MOBILENET,
    dropout_rate: float = 0.3,
) -> tf.keras.Model:
    """
    Tahap 1 transfer learning: backbone MobileNetV2 pretrained ImageNet
    DIBEKUAKAN, hanya classifier (kepala) yang dilatih.

    Kenapa transfer learning?
      Melatih CNN dari nol butuh JUTA gambar (ratusan ribu) dan berjam-jam
      GPU. Kita cuma punya 10.015 gambar. Tapi MobileNetV2 sudah dilatih
      pada 1,4 juta gambar ImageNet, jadi ia sudah "tahu" cara mengenali
      bentuk, tepi, dan tekstur umum. Kita hanya perlu menyesuaikan
      lapisan kepala (classifier) di atasnya.

    Tahap 1 ini persis sepertipz: kita hanya.layers.Dense kecil di atas
    backbone, jadi sangat cepat (beberapa epoch sudah cukup).

    Urutan layer PENTING:
        Input -> preprocess_input -> MobileNetV2 -> GAP -> Dropout -> Dense
    `preprocess_input` harus SEBELUM backbone, karena weight ImageNet
    dilatih dengan input dalam rentang -1..1.
    """
    inputs = tf.keras.layers.Input(shape=(img_size, img_size, 3))

    # Normalisasi ImageNet: nilai 0..255 menjadi -1..1.
    #
    # PENTING: kenapa memakai Rescaling dan bukan Lambda(preprocess_input)?
    #
    #   Rumus resmi MobileNetV2: (x - 127.5) / 127.5 = x/127.5 - 1.0
    #   Rescaling(scale, offset) menghitung: x * scale + offset
    #   Maka: Rescaling(1.0/127.5, -1.0) Memberikan hasil yang SAMA PERSIS.
    #
    #   Alasan memilih Rescaling: Layer Lambda yang membungkus fungsi
    #   `preprocess_input` TIDAK bisa dimuat kembali dari file .keras
    #   tanpa mendaftarkan custom_objects. Itu berarti file .keras harus
    #   dimuat dengan kode tambahan, dan bisa gagal di server deploy.
    #   Dengan Rescaling (layer bawaan Keras), file .keras bisa dimuat
    #   dengan `tf.keras.models.load_model(path)` polos di mana pun.
    #
    #   Menghapus Rescaling(1/255) di sini PENTING. Kalau kedua normalisasi
    #   dipasang, input berada di rentang yang tidak pernah dilihat model
    #   saat pretraining, dan performanya anjlok.
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0,
                                  name="preprocess_input")(inputs)

    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,          # buang classifier ImageNet
        weights="imagenet",         # bobot yang sudah dilatih
        alpha=1.0,                  # lebar backbone: 1.0 (penuh) sampai 1.4 (ringan)
    )
    # Bekukan semua backbone supaya tidak berubah saat tahap 1.
    backbone.trainable = False

    x = backbone(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    #. Dropout: secara random mematikan sebagian neuron saat training supaya
    # model tidak terlalu bergantung pada neuron tertentu (regularisasi).
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="mobilenetv2_stage1")
    return model


# ---------------------------------------------------------------------------
# 3. MOBILENETV2 - TAHAP 2 (fine-tuning sebagian lapisan atas)
# ---------------------------------------------------------------------------
def unfreeze_top_layers(
    model: tf.keras.Model,
    n_layers_to_unfreeze: int = 30,
) -> tf.keras.Model:
    """
    Membuka (membekukan -> trainable) lapisan atas backbone.

    Kenapa hanya lapisan ATAS?
      Lapisan bawah (yang pertama) telah belajar tugas dasar: mendeteksi garis,
      tepi, tekstur. Fitur itu berguna untuk SEMUA gambar di dunia, jadi
      tetap dipakai apa adanya.

      Lapisan atas (yang terakhir) belajar tugas spesifik ImageNet
      (klasifikasi 1000 kelas objek). Fitur itu kita perlakukan sebagai
      "bahan setengah jadi" yang perlu disesuaikan untuk lesi kulit.

    Kenapa learning rate kecil (1e-5)?
      Bobot yang sudah dilatih itu hasil kerja keras (puluhan jam GPU).
      Kita hanya butuh PENYESUAIAN kecil, bukan pelajaran dari nol.
      Learning rate besar akan MENGHANCURKAN fitur yang sudah bagus
      (katastrophic forgetting).

    CATATAN PENTANG TENTANG BatchNormalization
      Lapisan BatchNormalization punya 2 mode:
        - training=True  : memakai statistik batch saat ini, dan MENGUBAH
                           moving_mean/moving_variance setiap epoch
        - training=False : memakai statistik yang sudah disimpan (dari
                           ImageNet), dan TIDAK mengubah apa pun
      Kita HARUS mengunci BatchNormalization dalam mode inference
      (training=False) selama fine-tuning. Alasannya: batch size kita kecil
      (32) dan distribusinya berbeda dari ImageNet, jadi statistik batch
      yang noisy akan membuat prediksi BERAYAK (bolak-balik) antar epoch.
      Mengunci BN mempertahankan statistik yang stabil dan andal.
    """
    # Cari layer BatchNormalization di dalam backbone
    bn_layers = [
        layer for layer in model.layers
        if isinstance(layer, tf.keras.layers.BatchNormalization)
    ]

    # Aktifkan seluruh backbone dulu
    for layer in model.layers:
        layer.trainable = True

    # Kunci lagi semua BatchNormalization (mode inference)
    for layer in bn_layers:
        layer.trainable = False
        layer.trainable = False

    # Bekukan semua layer KECUALI n_layers_to_unfreeze terakhir
    total = len(model.layers)
    for i, layer in enumerate(model.layers):
        if i < total - n_layers_to_unfreeze:
            layer.trainable = False

    return model


# ---------------------------------------------------------------------------
# 4. TRAINING LOOP
# ---------------------------------------------------------------------------
def train_model(
    model: tf.keras.Model,
    train_ds,
    val_ds,
    train_steps: int,
    val_steps: int,
    epochs: int = C.MAX_EPOCHS_CNN,
    learning_rate: float = 1e-3,
    class_weight: dict | None = None,
    patience: int = C.EARLY_STOP_PATIENCE,
    run_name: str = "model",
    verbose: int = 1,
):
    """
    Melakukan training dengan kompilasi, class weight, dan EarlyStopping.

    Parameter class_weight:
      Dictionary {kelas: bobot}. Dihitung dari data TRAINING saja.
      Tujuannya: kelas dengan jumlah contoh sedikit (df: 115, vasc: 142)
      diberi bobot lebih besar, sehingga model tidak cuma mengabaikan
      kelas minoritas dan memprediksi "nv" untuk semuanya.

      Kenapa bobot ini penting? Tanpa class weight, loss fungsi akan didominasi
      oleh kelas nv (6705 dari 10.015 = 67% data). Model yang "malas" bisa
      mendapat accuracy 67% dengan selalu menebak nv. Itu tidak berguna
      untuk skrining kanker.

    Kenapa `restore_best_weights=True`?
      EarlyStopping berhenti saat validasi memburuk, TETAPI model pada saat
      itu belum tentu yang terbaik. Opsi ini mengembalikan bobot dari epoch
      dengan validasi terbaik. Tanpa ini, kita menyimpan model yang sudah
      mulai overfit.

    Mengembalikan: (model, history)
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        # Simpan yang terbaik ke disk (dalam format .keras)
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(C.MODELS_DIR / f"{run_name}_best.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        # Catat learning rate per epoch (berguna saat analisis fine-tuning)
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: logs.setdefault(
                "lr", float(tf.keras.backend.get_value(
                    model.optimizer.learning_rate))
            ) if logs is not None else None
        ),
    ]

    print(f"\n{'=' * 70}")
    print(f"TRAINING: {run_name}")
    print(f"  epochs (maks)     : {epochs}")
    print(f"  train steps/epoch : {train_steps}")
    print(f"  val steps/epoch   : {val_steps}")
    print(f"  learning rate     : {learning_rate}")
    print(f"  class weight      : {class_weight}")
    print(f"{'=' * 70}\n")

    t0 = time.time()
    history = model.fit(
        train_ds,
        steps_per_epoch=train_steps,
        validation_data=val_ds,
        validation_steps=val_steps,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=verbose,
    )
    elapsed = time.time() - t0

    print(f"\nTraining selesai dalam {elapsed / 60:.1f} menit "
          f"({elapsed:.0f} detik).")
    best_epoch = int(np.argmin(history.history["val_loss"]))
    print(f"Epoch dengan val_loss terbaik: epoch {best_epoch + 1} "
          f"(val_loss={min(history.history['val_loss']):.4f})")

    save_history_cache(run_name, history, elapsed, best_epoch + 1)

    return model, history


# ---------------------------------------------------------------------------
# 4b. CACHE HISTORY TRAINING
# ---------------------------------------------------------------------------
def save_history_cache(
    run_name: str,
    history,
    elapsed: float,
    best_epoch: int,
) -> Path:
    """
    Simpan history training (kurva loss/accuracy) ke docs/logs/.

    Kenapa perlu?
      Tanpa ini, kurva training yang muncul di notebook hanya ada di
      output `.ipynb`. Ketika notebook dibangun ulang (atau dibuka di
      komputer lain), kurvanya hilang padahal modelnya masih ada.
      Dengan menyimpan history ke JSON terpisah:
        - kurva bisa diplot ulang tanpa training ulang,
        - notebook bisa "memuat" model yang sudah dilatih (cache path),
        - angka di laporan bisa diverifikasi ke sumber aslinya.

    Isi file: {"run_name", "history", "train_time_s", "best_epoch"}.
    """
    path = C.DOCS_LOGS / f"{run_name}_history.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    # Ubah nilai numpy -> float biasa supaya JSON bisa dibaca rapi
    hist_bersih = {
        k: [float(v) for v in vals]
        for k, vals in history.history.items()
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_name": run_name,
                "history": hist_bersih,
                "train_time_s": float(elapsed),
                "best_epoch": int(best_epoch),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"History training disimpan: {path.name}")
    return path


def load_history_cache(run_name: str) -> dict | None:
    """
    Baca cache history training. Return None kalau belum ada / rusak.

    Dipakai notebook untuk memakai model yang sudah dilatih tanpa
    mengulang training yang mahal.
    """
    path = C.DOCS_LOGS / f"{run_name}_history.json"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if "history" not in data:
        return None
    return data


# ---------------------------------------------------------------------------
# 5. MENYIMPAN MODEL
# ---------------------------------------------------------------------------
def save_model(
    model: tf.keras.Model,
    run_name: str,
    extra_config: dict | None = None,
) -> Path:
    """
    Menyimpan model .keras beserta label_map.json dan preprocess_config.json.

    Kenapa file .keras (bukan .h5 atau SavedModel)?
      Format .keras adalah format resmi Keras 3: satu file berisi arsitektur,
      bobot, DAN konfigurasi training. Studiosi bisa memuatnya dengan
      `keras.models.load_model()` tanpa perlu knowing definisi class custom.

    Kenapa label_map.json dan preprocess_config.json disimpan terpisah?
      - label_map.json :/application web perlu tahu indeks ke-3 itu kelas apa.
        Tanpa ini, aplikasi tidak bisa menampilkan "Melanoma".
      - preprocess_config.json : aplikasi harus tahu ukuran gambar dan aturan
        normalisasi. Disimpan di file, bukan ditulis manual di kode app,
        supaya tidak bisa "lupa" dan tidak bisa melenceng.
    """
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = C.MODELS_DIR / f"{run_name}.keras"
    model.save(model_path)
    print(f"Model disimpan: {model_path} "
          f"({model_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # label_map.json
    label_map = {
        "classes": C.CLASSES,
        "class_to_idx": C.CLASS_TO_IDX,
        "dx_full_name": C.DX_FULL_NAME,
        "dx_indo": C.DX_INDO,
        "malignant_classes": C.MALIGNANT_CLASSES,
        "benign_classes": C.BENIGN_CLASSES,
    }
    with open(C.MODELS_DIR / C.LABEL_MAP_NAME, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)
    print(f"Label map disimpan: {C.MODELS_DIR / C.LABEL_MAP_NAME}")

    # preprocess_config.json
    preprocess_config = {
        "img_size": int(model.input_shape[1]),
        "normalization": (
            "rescaling_1_255"
            if "cnn" in run_name.lower()
            else "mobilenet_v2_preprocess_input"
        ),
        "color_mode": "RGB",
        "seed": C.SEED,
    }
    if extra_config:
        preprocess_config.update(extra_config)

    with open(C.MODELS_DIR / C.PREPROCESS_CONFIG_NAME, "w", encoding="utf-8") as f:
        json.dump(preprocess_config, f, indent=2, ensure_ascii=False)
    print(f"Config preprocessing disimpan: "
          f"{C.MODELS_DIR / C.PREPROCESS_CONFIG_NAME}")

    return model_path


def load_model(model_name: str):
    """Memuat model .keras beserta label map dan config preprocessing."""
    model_path = C.MODELS_DIR / f"{model_name}.keras"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model '{model_name}' tidak ditemukan di {model_path}. "
            "Jalankan notebook training terlebih dahulu."
        )
    model = tf.keras.models.load_model(model_path)

    with open(C.MODELS_DIR / C.LABEL_MAP_NAME, encoding="utf-8") as f:
        label_map = json.load(f)
    with open(C.MODELS_DIR / C.PREPROCESS_CONFIG_NAME, encoding="utf-8") as f:
        preprocess_config = json.load(f)

    return model, label_map, preprocess_config
