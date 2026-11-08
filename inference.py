"""Inference utilities aligned with the three uploaded EfficientNet-B3 notebooks.

Important: TensorFlow EfficientNet-B3 embeds rescaling in the model. It expects
RGB float32 pixels in the [0, 255] range, NOT separately divided by 255.
Image size is read from the saved model, because notebook code used 299x299
while the testing and Grad-CAM notebooks loaded a 300x300 saved model.
"""
from __future__ import annotations

import base64
import os
import threading
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
CLASS_NAMES = ["Normal", "Doubtful", "Mild", "Moderate", "Severe"]
CLASS_META = [
    {"grade": 0, "name": "Normal", "label": "Grade 0 · Normal", "severity": "Tidak tampak tanda radiografis OA", "description": "Tidak terdapat tanda osteoarthritis yang jelas pada citra radiografi.", "tone": "normal"},
    {"grade": 1, "name": "Doubtful", "label": "Grade 1 · Doubtful", "severity": "Perubahan meragukan", "description": "Dapat tampak penyempitan celah sendi atau osteofit yang masih meragukan.", "tone": "doubtful"},
    {"grade": 2, "name": "Mild", "label": "Grade 2 · Mild", "severity": "Osteoarthritis ringan", "description": "Osteofit mulai terlihat jelas; kemungkinan terdapat penyempitan celah sendi.", "tone": "mild"},
    {"grade": 3, "name": "Moderate", "label": "Grade 3 · Moderate", "severity": "Osteoarthritis sedang", "description": "Penyempitan celah sendi tampak jelas, dengan osteofit multipel dan kemungkinan sklerosis tulang.", "tone": "moderate"},
    {"grade": 4, "name": "Severe", "label": "Grade 4 · Severe", "severity": "Osteoarthritis berat", "description": "Penyempitan celah sendi berat, osteofit besar, serta perubahan tulang yang nyata.", "tone": "severe"},
]
MODEL_NAMES = (
    "EfficientNetB3_Knee_Finall.h5",       # filename used by Testing and Grad-CAM notebooks
    "EfficientNetB3_Knee_Finall3.keras",   # filename in Training notebook
    "EfficientNetB3_KneeOA_best.keras",    # stage-2 checkpoint in Training notebook
    "model.keras",
    "model.h5",
)
MODEL_LOCK = threading.RLock()


class ModelUnavailable(RuntimeError):
    """Model weights are missing or TensorFlow cannot be loaded."""


class PredictionError(RuntimeError):
    """Model inference failed; no fabricated response is returned."""


def get_model_path() -> Path:
    configured = os.getenv("MODEL_PATH")
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_absolute() else ROOT / path
    for filename in MODEL_NAMES:
        path = ROOT / "model" / filename
        if path.is_file():
            return path
    return ROOT / "model" / MODEL_NAMES[0]


@lru_cache(maxsize=2)
def _load_model(path_str: str):
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ModelUnavailable(
            "TensorFlow belum terpasang. Jalankan: pip install -r requirements.txt"
        ) from exc
    try:
        # compile=False avoids requiring training-only optimizer/loss configuration.
        return tf.keras.models.load_model(path_str, compile=False)
    except Exception as exc:
        raise ModelUnavailable(
            "Model tidak bisa dibuka. Pastikan file .h5/.keras valid dan versi TensorFlow kompatibel. "
            f"Detail: {type(exc).__name__}: {str(exc)[:180]}"
        ) from exc


def _model_hw(model) -> tuple[int, int]:
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    if not shape or len(shape) != 4 or shape[-1] != 3:
        raise PredictionError("Model harus memiliki input RGB 4D (batch, tinggi, lebar, 3).")
    h = int(shape[1]) if shape[1] else 300
    w = int(shape[2]) if shape[2] else 300
    return h, w


def _encode_png(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _heatmap_to_rgb(heatmap: np.ndarray) -> Image.Image:
    """Small, deterministic blue→cyan→yellow→red colormap, no matplotlib needed."""
    h = np.clip(heatmap.astype(np.float32), 0.0, 1.0)
    points = np.array([0.0, 0.25, 0.50, 0.75, 1.0], dtype=np.float32)
    colors = np.array([
        [25, 38, 110], [19, 130, 224], [76, 210, 196], [252, 207, 75], [238, 68, 66]
    ], dtype=np.float32)
    stacked = np.stack([np.interp(h, points, colors[:, c]) for c in range(3)], axis=-1)
    return Image.fromarray(np.uint8(np.clip(stacked, 0, 255)), mode="RGB")


def _find_gradcam_output(model, tf, x):
    """Return differentiable conv activation and prediction tensors.

    For a flat Functional model, the top_conv activation is already in the
    outer graph. For an EfficientNet nested in the classifier, replay the
    sequential head on the backbone's output while recording gradients.
    """
    try:
        conv_layer = model.get_layer("top_conv")
        grad_model = tf.keras.Model(model.inputs, [conv_layer.output, model.output])
        return grad_model(x, training=False)
    except (ValueError, KeyError):
        pass

    for i, layer in enumerate(model.layers):
        if not isinstance(layer, tf.keras.Model):
            continue
        try:
            conv_layer = layer.get_layer("top_conv")
        except ValueError:
            continue
        feature_model = tf.keras.Model(layer.inputs, [conv_layer.output, layer.output])
        conv_maps, z = feature_model(x, training=False)
        for head in model.layers[i + 1:]:
            if isinstance(head, (tf.keras.layers.Dropout, tf.keras.layers.BatchNormalization)):
                z = head(z, training=False)
            else:
                z = head(z)
        return conv_maps, z

    raise PredictionError("Layer top_conv tidak ditemukan pada model untuk Grad-CAM.")


def _gradcam(model, image_batch: np.ndarray, class_idx: int) -> np.ndarray:
    import tensorflow as tf
    x = tf.convert_to_tensor(image_batch, dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_maps, outputs = _find_gradcam_output(model, tf, x)
        class_score = outputs[:, class_idx]
    gradients = tape.gradient(class_score, conv_maps)
    if gradients is None:
        raise PredictionError("Gradien Grad-CAM tidak tersedia pada model ini.")
    weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
    heatmap = tf.reduce_sum(conv_maps * weights, axis=-1)[0]
    heatmap = tf.nn.relu(heatmap)
    maximum = tf.reduce_max(heatmap)
    heatmap = tf.where(maximum > 1e-12, heatmap / maximum, tf.zeros_like(heatmap))
    h = heatmap.numpy()
    if not np.all(np.isfinite(h)):
        raise PredictionError("Nilai Grad-CAM tidak valid.")
    return h


def predict_image(image: Image.Image) -> dict:
    model_path = get_model_path()
    if not model_path.is_file():
        raise ModelUnavailable(
            "File bobot model belum tersedia. Salin EfficientNetB3_Knee_Finall.h5 "
            "atau EfficientNetB3_Knee_Finall3.keras ke folder model/."
        )

    with MODEL_LOCK:
        model = _load_model(str(model_path.resolve()))
        height, width = _model_hw(model)
        resized = image.resize((width, height), Image.Resampling.BILINEAR).convert("RGB")
        batch = np.expand_dims(np.asarray(resized, dtype=np.float32), axis=0)

        try:
            raw = np.asarray(model(batch, training=False).numpy(), dtype=np.float64)
        except Exception as exc:
            raise PredictionError(f"Prediksi gagal: {type(exc).__name__}: {str(exc)[:140]}") from exc

        if raw.shape != (1, len(CLASS_NAMES)):
            raise PredictionError(f"Output model tidak sesuai: {raw.shape}; diperlukan 5 kelas.")
        probabilities = raw[0]
        if not np.isfinite(probabilities).all():
            raise PredictionError("Model menghasilkan NaN/Infinity; periksa bobot dan training.")
        if np.any(probabilities < 0) or abs(probabilities.sum() - 1) > 0.01:
            scaled = probabilities - probabilities.max()
            exp = np.exp(scaled)
            probabilities = exp / exp.sum()
        else:
            probabilities = np.clip(probabilities, 0, 1)
            probabilities /= probabilities.sum()

        class_idx = int(np.argmax(probabilities))
        meta = CLASS_META[class_idx].copy()
        response = {
            "prediction": meta,
            "confidence": round(float(probabilities[class_idx] * 100), 2),
            "probabilities": [
                {"grade": i, "name": name, "probability": round(float(p * 100), 2)}
                for i, (name, p) in enumerate(zip(CLASS_NAMES, probabilities))
            ],
            "original_image": _encode_png(resized),
            "heatmap_image": None,
            "overlay_image": None,
            "gradcam_available": False,
            "gradcam_error": None,
            "model_name": "EfficientNet-B3",
            "model_file": model_path.name,
            "image_size": [width, height],
        }

        try:
            heatmap = _gradcam(model, batch, class_idx)
            heatmap_rgb = _heatmap_to_rgb(heatmap)
            heatmap_rgb = heatmap_rgb.resize((width, height), Image.Resampling.BILINEAR)
            overlay = Image.blend(resized, heatmap_rgb, alpha=0.40)
            response.update(
                heatmap_image=_encode_png(heatmap_rgb),
                overlay_image=_encode_png(overlay),
                gradcam_available=True,
            )
        except Exception as exc:
            # Grad-CAM is optional; a valid classification must still be shown.
            response["gradcam_error"] = (
                f"Visualisasi Grad-CAM tidak tersedia pada arsitektur model ini ({type(exc).__name__})."
            )
        return response
