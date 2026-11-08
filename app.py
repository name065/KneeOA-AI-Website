"""Web application for Knee Osteoarthritis grading with a trained EfficientNetB3 model.

Run: python app.py
Set MODEL_PATH to the .h5/.keras file exported from the user's notebook.
"""
from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps, UnidentifiedImageError

from inference import ModelUnavailable, PredictionError, get_model_path, predict_image

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)
app.config.update(
    MAX_CONTENT_LENGTH=10 * 1024 * 1024,  # 10 MB
    JSON_SORT_KEYS=False,
)
Image.MAX_IMAGE_PIXELS = 25_000_000


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/api/status")
def status():
    model_path = get_model_path()
    return jsonify({
        "model_available": model_path.is_file(),
        "model_filename": model_path.name if model_path.is_file() else None,
        "maximum_upload_mb": 10,
        "classes": ["Normal", "Doubtful", "Mild", "Moderate", "Severe"],
    })


@app.post("/api/predict")
def predict():
    upload = request.files.get("image")
    if upload is None or not upload.filename:
        return jsonify(error="Pilih gambar X-ray lutut terlebih dahulu."), 400

    if Path(upload.filename).suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        return jsonify(error="Format yang didukung: JPG, JPEG, PNG, atau WebP."), 400

    try:
        raw = upload.read()
        if not raw:
            return jsonify(error="File gambar kosong."), 400
        with Image.open(BytesIO(raw)) as raw_image:
            if raw_image.format not in {"JPEG", "PNG", "WEBP"}:
                return jsonify(error="Berkas bukan gambar JPG, PNG, atau WebP yang valid."), 400
            raw_image.load()
            image = ImageOps.exif_transpose(raw_image).convert("RGB")
        if min(image.size) < 64:
            return jsonify(error="Resolusi gambar terlalu kecil (minimal 64 × 64 piksel)."), 400
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        return jsonify(error="Gambar tidak dapat dibaca atau rusak."), 400

    try:
        result = predict_image(image)
        return jsonify(result)
    except ModelUnavailable as exc:
        return jsonify(error=str(exc), model_missing=not get_model_path().is_file()), 503
    except PredictionError as exc:
        app.logger.exception("Model prediction failed")
        return jsonify(error=str(exc)), 500


@app.errorhandler(413)
def too_large(_error):
    return jsonify(error="Ukuran gambar melebihi 10 MB."), 413


@app.errorhandler(404)
def not_found(_error):
    return jsonify(error="Halaman atau API tidak ditemukan."), 404


if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5000")), debug=debug)
