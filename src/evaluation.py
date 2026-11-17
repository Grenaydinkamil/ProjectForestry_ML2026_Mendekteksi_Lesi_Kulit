"""
evaluation.py
=============
Perhitungan metrik evaluasi dan visualisasi hasil model.

Metrik yang dipakai (dan kenapa)
--------------------------------
Accuracy TIDAK boleh jadi dasar pemilihan model untuk dataset ini.
Alasannya: 67% dari data adalah kelas "nv" (mola, jinak). Model yang
sembilan arah "hafal mola" dan selalu menjawab nv akan mendapat accuracy
67% tanpa benar-benar belajar apa pun.

Untuk alat skrining kanker, kesalahan yang TIDAK BOLEH terjadi adalah
"melanoma lolos sebagai jinak" (false negative). Jadi metrik utama kita:

  - F1 macro       : rata-rata harmonic mean precision & recall untuk semua
                     kelas. Menghukum kelas yang diabaikan.
  - Recall kelas ganas : dari semua lesi ganas (mel/bcc/akiec), berapa persen
                     yang BERHASIL dikenali? Ini metrik paling kritis untuk
                     aplikasi skrining.
  - Confusion matrix: menunjukkan KLASIFIKASI SALAH mana yang paling sering
                     terjadi. Sangat penting untuk melihat melanoma lolos.
  - Precision/Recall per kelas.

Modul ini juga menyediakan:
  - plot_confusion_matrix()
  - plot_training_curves()
  - evaluate_model() yang merangkum semua metrik ke satu dictionary
  - tune_binary_threshold() untuk mencari titik Operating terbaik dari
    data VALIDASI (bukan test!)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from . import config as C


# ---------------------------------------------------------------------------
# 1. EVALUASI KLASIFIKASI MULTI-KELAS
# ---------------------------------------------------------------------------
def predict_with_proba(model, dataset, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Menjalankan prediksi dan mengembalikan (y_pred, y_proba).

    ### Kenapa tidak langsung pakai `model.predict_proba`?
    # Model Sequential (CNN dari nol) **tidak punya** metode
    # `predict_proba` di Keras 3. Hanya model klasifikasi biner/kelompok
    # (SVC, RandomForest) yang memilikinya.
    #
    # Karena itu di sini kita pakai satu jalur yang berlaku untuk SEMUA
    # model:
    #   - kalau `predict_proba` ada, pakai itu
    #   - kalau tidak, pakai `predict` yang untuk model dengan aktivasi
    #     softmax sudah menghasilkan PROBABILITAS (bukan label keras)
    #
    # Model CNN dan MobileNetV2 kita mengakhiri dengan softmax, jadi
    # `predict` mengembalikan probabilitas. Ini diverifikasi dengan
    # pemeriksaan jumlah baris (lihat di bawah).
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(dataset, steps=steps, verbose=0)
    else:
        proba = model.predict(dataset, steps=steps, verbose=0)
        # Pastikan ini memang probabilitas: jumlah tiap baris harus 1
        row_sums = proba.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-3):
            raise ValueError(
                "Output model bukan probabilitas ( jumlah baris != 1 ). "
                "Pastikan layer terakhir memakai aktivasi softmax."
            )
    y_pred = np.argmax(proba, axis=1)
    return y_pred, proba


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray | None = None,
    classes: list | None = None,
) -> dict:
    """
    Menghitung seluruh metrik evaluasi untuk model multi-kelas.

    Parameter:
      y_true       : label asli (integer 0..6)
      y_pred       : label prediksi (integer 0..6)
      y_pred_proba : probabilitas prediksi (n x 7), opsional (untuk AUC)
      classes      : daftar nama kelas sesuai indeks

    Mengembalikan dictionary metrik.
    """
    classes = classes or C.CLASSES
    n_classes = len(classes)

    results = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }

    # --- Metrik per kelas ---
    per_class = {}
    for i, cls in enumerate(classes):
        tp = int(np.sum((y_true == i) & (y_pred == i)))
        fp = int(np.sum((y_true != i) & (y_pred == i)))
        fn = int(np.sum((y_true == i) & (y_pred != i)))
        support = int(np.sum(y_true == i))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        per_class[cls] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": support,
            "tp": tp, "fp": fp, "fn": fn,
        }
    results["per_class"] = per_class

    # --- AUC (multi-kelas, one-vs-rest) ---
    if y_pred_proba is not None:
        try:
            results["auc_ovr_macro"] = float(
                roc_auc_score(y_true, y_pred_proba, multi_class="ovr",
                              average="macro", labels=list(range(n_classes)))
            )
        except ValueError:
            # Happens kalau ada kelas yang tidak muncul di y_true
            results["auc_ovr_macro"] = None

    # --- Metrik BINER: ganas vs tidak ganas ---
    malignant_idx = [C.CLASS_TO_IDX[c] for c in C.MALIGNANT_CLASSES]
    y_true_mal = np.isin(y_true, malignant_idx).astype(int)
    y_pred_mal = np.isin(y_pred, malignant_idx).astype(int)

    results["binary"] = {
        "accuracy": float(accuracy_score(y_true_mal, y_pred_mal)),
        "precision": float(
            precision_score(y_true_mal, y_pred_mal, zero_division=0)
        ),
        "recall": float(recall_score(y_true_mal, y_pred_mal, zero_division=0)),
        "f1": float(f1_score(y_true_mal, y_pred_mal, zero_division=0)),
        # Recall ganas = berapa melanoma/bcc/akiec yangsuccess terdeteksi
        "malignant_recall": float(recall_score(y_true_mal, y_pred_mal,
                                               zero_division=0)),
        # Specificity = dari yang tidak ganas, berapa yang benar ditolak
        "specificity": float(
            recall_score(1 - y_true_mal, 1 - y_pred_mal, zero_division=0)
        ),
        # Confusion biner
        "tn": int(np.sum((y_true_mal == 0) & (y_pred_mal == 0))),
        "fp": int(np.sum((y_true_mal == 0) & (y_pred_mal == 1))),
        "fn": int(np.sum((y_true_mal == 1) & (y_pred_mal == 0))),
        "tp": int(np.sum((y_true_mal == 1) & (y_pred_mal == 1))),
    }

    # --- AUC biner (jika probabilitas tersedia) ---
    if y_pred_proba is not None:
        malignant_idx = [C.CLASS_TO_IDX[c] for c in C.MALIGNANT_CLASSES]
        prob_mal = y_pred_proba[:, malignant_idx].sum(axis=1)
        try:
            results["binary"]["auc"] = float(
                roc_auc_score(y_true_mal, prob_mal)
            )
        except ValueError:
            results["binary"]["auc"] = None

    return results


def print_report(results: dict, title: str = "HASIL EVALUASI") -> None:
    """Menampilkan hasil evaluasi dalam format yang mudah dibaca."""
    print("=" * 72)
    print(title)
    print("=" * 72)

    print("\n-- Metrik aggregate --")
    for key in ["accuracy", "precision_macro", "recall_macro", "f1_macro",
                "precision_weighted", "recall_weighted", "f1_weighted"]:
        print(f"  {key:22s}: {results[key]:.4f}")
    if results.get("auc_ovr_macro") is not None:
        print(f"  {'auc_ovr_macro':22s}: {results['auc_ovr_macro']:.4f}")

    print("\n-- Metrik per kelas --")
    header = f"  {'kelas':<8}{'precision':>11}{'recall':>10}{'f1':>9}{'support':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for cls, m in results["per_class"].items():
        print(f"  {cls:<8}{m['precision']:>11.4f}{m['recall']:>10.4f}"
              f"{m['f1']:>9.4f}{m['support']:>10d}")

    b = results["binary"]
    print("\n-- Biner (ganas vs tidak ganas) --")
    print(f"  malignant_recall : {b['malignant_recall']:.4f}   "
          "(bagian lesi ganas yang terdeteksi)")
    print(f"  specificity      : {b['specificity']:.4f}   "
          "(bagian tidak ganas yang benar ditolak)")
    print(f"  precision        : {b['precision']:.4f}")
    print(f"  f1               : {b['f1']:.4f}")
    if b.get("auc") is not None:
        print(f"  auc              : {b['auc']:.4f}")
    print(f"  TN={b['tn']}  FP={b['fp']}  FN={b['fn']}  TP={b['tp']}")
    print(f"\n  Catatan: FN={b['fn']} berarti ada lesi ganas yang TIDAK "
          f"terdeteksi.\n  Pada aplikasi skrining, FN adalah kesalahan "
          f"paling berbahaya.")
    print("=" * 72)


# ---------------------------------------------------------------------------
# 2. VISUALISASI
# ---------------------------------------------------------------------------
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    save_path: str | Path | None = None,
    normalize: bool = False,
):
    """
    Menggambar confusion matrix dengan heatmap seaborn.

    Sumbu Y = label asli, Sumbu X = label prediksi.
    Diagonal = prediksi benar. Yang OFF-diagonal = kesalahan.

    Fokus analisis: baris "mel" (melanoma) adalah yang paling penting. Kita ingin
    melihat berapa melanoma yang terpredict jadi kelas lain (malignant
    melanoma lolos sebagai jinak = false negative).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(C.CLASSES))))
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
        # Ganti 0 dengan "-" supaya tabel tidak ramai, dan tandai diagonal
        annot = np.where(cm > 0, cm, np.nan)
    else:
        fmt = "d"
        annot = cm

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        annot,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=C.CLASSES,
        yticklabels=C.CLASSES,
        ax=ax,
        cbar_kws={"label": "Proporsi" if normalize else "Jumlah"},
    )
    ax.set_xlabel("Prediksi Model")
    ax.set_ylabel("Label Asli")
    ax.set_title(title)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix disimpan: {save_path}")
    plt.show()
    plt.close(fig)


def plot_training_curves(history, save_path: str | Path | None = None,
                         title: str = "Kurva Training"):
    """
    Menggambar kurva loss dan accuracy untuk data train dan validasi.

    CARA MEMBACA KURVA INI (ini bagian yang akan ditanya saat viva):
      - Jarak kurva train dan validation yang MELAR = overfitting
      - Dua kurva yang sama-sama masih menurun dan BERDEKAT = masih bisa
        dilatih lebih lama (belum converge)
      - Kurva validation yang NAIK lagi sementara train turun = overfitting
        mulai terjadi; titik terbaik ada di epoch sebelum itu
      - Train loss yang SANGAT kecil (mendekati 0) sementara validation
        tinggi = overfitting parah (model menghafal)
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Loss ---
    ax = axes[0]
    ax.plot(history.history["loss"], label="Training Loss", color="#1f77b4")
    ax.plot(history.history["val_loss"], label="Validation Loss",
            color="#d62728")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss (sudah termasuk overfitting check)")
    ax.legend()
    ax.grid(alpha=0.3)

    # --- Accuracy ---
    ax = axes[1]
    ax.plot(history.history["accuracy"], label="Training Accuracy",
            color="#1f77b4")
    ax.plot(history.history["val_accuracy"], label="Validation Accuracy",
            color="#d62728")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy train vs validasi")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Kurva training disimpan: {save_path}")
    plt.show()
    plt.close(fig)


def plot_error_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    image_paths: list,
    class_names: list | None = None,
    n_examples: int = 8,
    focus_class: str = "mel",
    save_path: str | Path | None = None,
):
    """
    Menampilkan contoh prediksi SALAH, terutama dari kelas `focus_class`.

    Kenapa ini penting untuk laporan?
      Metrik angka memberi tahu "berapa banyak yang salah", tapi contoh
      gambarnya memberi tahu "KENAPA salah". Untuk presentasi, biasanya
     MENARIKANYA.
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    class_names = class_names or C.CLASSES
    focus_idx = C.CLASS_TO_IDX[focus_class]

    wrong_mask = (y_true == focus_idx) & (y_pred != focus_idx)
    wrong_indices = np.where(wrong_mask)[0]

    if len(wrong_indices) == 0:
        print(f"Tidak ada prediksi salah untuk kelas '{focus_class}'.")
        return

    print(f"Contoh prediksi salah untuk kelas '{focus_class}': "
          f"{len(wrong_indices)} dari {int(np.sum(y_true == focus_idx))}")

    n_show = min(n_examples, len(wrong_indices))
    fig, axes = plt.subplots(2, (n_show + 1) // 2, figsize=(16, 8))
    axes = axes.ravel()

    for ax_i, idx in enumerate(wrong_indices[:n_show]):
        ax = axes[ax_i]
        try:
            img = Image.open(image_paths[idx]).convert("RGB")
            ax.imshow(img)
        except Exception:  # noqa: BLE001
            ax.text(0.5, 0.5, "gambar tidak terbaca", ha="center")
        ax.set_title(
            f"Actual: {class_names[y_true[idx]]}\n"
            f"Prediksi: {class_names[y_pred[idx]]}",
            fontsize=9,
        )
        ax.axis("off")

    for ax_i in range(n_show, len(axes)):
        axes[ax_i].axis("off")

    fig.suptitle(
        f"Prediksi salah pada kelas '{focus_class}' "
        f"(lesi ganas yang lolos sebagai kelas lain)",
        fontsize=13,
    )
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Contoh error disimpan: {save_path}")
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. TUNING THRESHOLD BINER (dari data VALIDASI)
# ---------------------------------------------------------------------------
def tune_binary_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    thresholds: np.ndarray | None = None,
    target_recall: float = 0.90,
) -> dict:
    """
    Mencari threshold keputusan "ganas" dari data VALIDASI.

    MASALAH: model kita menghasilkan 7 probabilitas kelas. Aplikasi butuh
    satu angka: "risiko ganas" = jumlah probabilitas kelas ganas
    (mel + bcc + akiec). Kita perlu deciding KAPAN Official Categorical
    considerar itu "cukup tinggi untuk diperiksakan dokter".

    Kenapa TIDAK langsung pakai 0.5?
      Karena 0.5 terlalu arbitrary. Yang kita mau: threshold yang
      memenuhi syarat recall ganas tinggi (supaya sedikit melanoma lolos)
      dengan tetap menjaga specificity yang wajar (supaya tidak terlalu
      banyak false alarm).

    PARAMETER PENTING: fungsi ini HANYA boleh dipanggil dengan data
    VALIDASI. Kalau dipanggil dengan data test, threshold-nya jadi
    "tertuning" terhadap test set, dan hasil test jadi tidak lagi
    pengukuran yang jujur. Ini pelanggaran aturan metodologi.

    Parameter:
      y_true       : label validasi
      y_pred_proba : probabilitas prediksi validasi
      target_recall: target minimum recall kelas ganas
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.01)

    malignant_idx = [C.CLASS_TO_IDX[c] for c in C.MALIGNANT_CLASSES]
    prob_mal = y_pred_proba[:, malignant_idx].sum(axis=1)
    y_true_mal = np.isin(y_true, malignant_idx).astype(int)

    rows = []
    for t in thresholds:
        pred = (prob_mal >= t).astype(int)
        tp = int(np.sum((y_true_mal == 1) & (pred == 1)))
        fp = int(np.sum((y_true_mal == 0) & (pred == 1)))
        fn = int(np.sum((y_true_mal == 1) & (pred == 0)))
        tn = int(np.sum((y_true_mal == 0) & (pred == 0)))

        rec = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

        rows.append({
            "threshold": float(t),
            "recall": rec,
            "specificity": spec,
            "precision": prec,
            "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })

    # Cari threshold dengan recall >= target, ambil yang specificity terbaik
    # (karena threshold rendah = lebih banyak false alarm)
    candidates = [r for r in rows if r["recall"] >= target_recall]
    if candidates:
        best = max(candidates, key=lambda r: (r["specificity"], r["f1"]))
        best["strategy"] = (
            f"Threshold terkecil yang memenuhi recall ganas >= "
            f"{target_recall:.0%}, dengan specificity tertinggi"
        )
    else:
        # Tidakpun threshold yang mencapai target recall
        best = max(rows, key=lambda r: r["recall"])
        best["strategy"] = (
            f"Tidak ada threshold yang mencapai recall {target_recall:.0%}. "
            f"Dipakai threshold dengan recall tertinggi."
        )

    best["target_recall"] = target_recall
    best["curve"] = rows
    return best


# ---------------------------------------------------------------------------
# 4. PREDIKSI DARI MODEL
# ---------------------------------------------------------------------------
def predict(model, dataset, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Menjalankan model pada dataset dan mengembalikan (y_pred, y_pred_proba).

    Parameter n_samples penting: dataset kita memakai .repeat() supaya
    epoch tidak terputus, jadi kita harus membatasi jumlah batch yang
    diproses agar tidak mengambil data berulang.
    """
    y_pred_proba = model.predict(dataset, steps=n_samples, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    return y_pred, y_pred_proba


def recall_score_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Menghitung recall tanpa error saat salah satu kelas tidak ada.

    Perbandingan manual yang biasa (TP / (TP + FN)) akan menghasilkan
    ZeroDivisionError kalau tidak ada satu pun instance positif. Fungsi ini
    mengembalikan 0.0 dalam kasus itu (artinya model tidak menemukan
    apa pun dari kelas tersebut, yang memang layak dinilai 0).

    Dipakai di notebook uji eksternal untuk perhitungan sensitivitas dengan
    definisi kelas yang berbeda.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    positives = y_true == 1
    n_positive = int(positives.sum())
    if n_positive == 0:
        return 0.0
    return float((pred_positives := (y_pred[positives] == 1)).sum()) / n_positive


def save_metrics(results: dict, filename: str) -> Path:
    """Menyimpan metrik ke file JSON di docs/logs/."""
    out_dir = C.REPO_ROOT / "docs" / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename

    # Buang key yang bukan JSON-serializable (misal numpy types)
    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=_default)
    print(f"Metrik disimpan: {path}")
    return path
