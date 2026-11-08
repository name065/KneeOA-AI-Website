
import json
import shutil
from pathlib import Path

import h5py

source = Path("model/EfficientNetB3_Knee_Finall.h5")
target = Path("model/EfficientNetB3_Knee_Finall_fixed.h5")

def clean_config(obj):
    removed = 0

    if isinstance(obj, dict):
        if obj.get("quantization_config", "missing") is None:
            del obj["quantization_config"]
            removed += 1

        for value in obj.values():
            removed += clean_config(value)

    elif isinstance(obj, list):
        for value in obj:
            removed += clean_config(value)

    return removed

if not source.exists():
    raise FileNotFoundError(f"Model tidak ditemukan: {source}")

# Salin model agar file original tidak berubah
shutil.copy2(source, target)

with h5py.File(target, "r+") as file:
    raw = file.attrs["model_config"]

    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")

    config = json.loads(raw)
    count = clean_config(config)

    file.attrs["model_config"] = json.dumps(config)

print(f"Konfigurasi yang dibersihkan: {count}")
print(f"Model hasil perbaikan: {target}")
print("Bobot model tidak diubah.")
