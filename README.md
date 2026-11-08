# KneeOA AI — Website Klasifikasi Osteoarthritis Lutut

Website berbasis **Flask + TensorFlow/Keras** untuk mengklasifikasikan citra X-ray lutut ke lima grade Kellgren–Lawrence menggunakan **EfficientNet-B3**, ditambah visualisasi **Grad-CAM**. Antarmuka bahasa Indonesia, responsif, dan mencakup unggah gambar, hasil prediksi, probabilitas per kelas, dan cetak hasil (melalui fitur Print → Save as PDF di browser).

## 1. Sumber implementasi

| Notebook yang Anda unggah | Bagian yang digunakan |
|---|---|
| `Training_ImageClassification_EfficientNet_B3 (8).ipynb` | EfficientNet-B3, input RGB, 5 kelas, preprocessing piksel 0–255, training dan penyimpanan model |
| `Testing-ImageClassification-EfficientNet-B3_b (4).ipynb` | Pengujian model dan prediksi 5 kelas; evaluasi 75% akurasi pada 140 citra uji (sesuai output notebook) |
| `gradcamm (2).ipynb` | Target layer `top_conv`, gradient-weighted class activation, heatmap dan overlay |

**Penting:** Tiga file `.ipynb` tidak berisi bobot model terlatih. Karena itu, website **tidak mengarang hasil prediksi**. Anda **harus mengunduh file `.h5` atau `.keras` hasil training** dari Google Drive agar analisis berfungsi.

Notebook training menunjukkan ukuran **299×299**, sedangkan notebook testing/Grad-CAM memuat file `.h5` dengan input **300×300**. Implementasi ini membaca bentuk input dari **model yang benar-benar Anda pasang** sehingga aman untuk kedua versi. EfficientNet Keras menyertakan layer rescaling internal, jadi input gambar diberi rentang RGB **0–255**, tanpa pembagian `255` tambahan.

## 2. Persyaratan

- Python **3.10 atau 3.11** direkomendasikan agar mudah memasang TensorFlow.
- RAM disarankan 8 GB atau lebih; aplikasi dapat berjalan dengan CPU.
- File bobot model asli hasil training: `.h5` atau `.keras`.

## 3. Instalasi dan menjalankan website

Buka terminal/Command Prompt pada folder `KneeOA-AI-Website`:

**Windows (PowerShell)**

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

**Mac/Linux/WSL**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Alternatif Windows: klik dua kali **`run_windows.bat`** (memerlukan Python 3.11). Pada Linux/Mac, jalankan **`./run_linux_mac.sh`** setelah Python 3.11 tersedia.

Kemudian buka **http://127.0.0.1:5000/** di browser.

## 4. Memasukkan bobot model yang benar

Dari notebook `Testing` atau `gradcamm`, file model disimpan di Google Drive:

```text
MyDrive/230411100065_Yudha Susilo/Dataset No3/EfficientNetB3/model/EfficientNetB3_Knee_Finall.h5
```

Unduh lalu letakkan di:

```text
KneeOA-AI-Website/
  model/
    EfficientNetB3_Knee_Finall.h5
```

Notebook training juga menyimpan file `EfficientNetB3_Knee_Finall3.keras`. Anda dapat menggunakan salah satunya, **pastikan bobotnya telah terlatih dengan benar**. Nama lain dapat ditentukan melalui variabel lingkungan:

**PowerShell**

```powershell
$env:MODEL_PATH = "C:\lokasi\model_saya.keras"
python app.py
```

**Linux/Mac**

```bash
MODEL_PATH="/lokasi/model_saya.keras" python app.py
```

Aplikasi secara otomatis memilih `EfficientNetB3_Knee_Finall.h5`, lalu `EfficientNetB3_Knee_Finall3.keras`, `EfficientNetB3_KneeOA_best.keras`, `model.keras`, atau `model.h5` di folder `model/`.

Jika model belum tersedia, indikator kanan atas akan menampilkan **"Bobot model belum ada"** dan API akan menjelaskan bahwa analisis tidak dapat dilakukan. Tidak ada mode prediksi palsu.

## 5. Alur penggunaan

1. Klik **Pilih Gambar** atau seret file X-ray lutut (JPG, PNG, WebP; maksimum 10 MB).
2. Klik **Analisis Gambar**.
3. Lihat hasil klasifikasi grade 0–4 dan probabilitas setiap grade.
4. Lihat Grad-CAM: gambar input, heatmap, dan overlay. Jika tidak kompatibel dengan struktur saved model, prediksi tetap ditampilkan dan pesan Grad-CAM yang relevan diberikan.
5. Klik **Cetak / Simpan PDF** dan pilih **Save as PDF** untuk menyimpan laporan dari browser.

### Grade

- 0 — Normal
- 1 — Doubtful
- 2 — Mild
- 3 — Moderate
- 4 — Severe

## 6. Struktur proyek

```text
KneeOA-AI-Website/
├── app.py                # Server Flask dan API upload/status
├── inference.py          # Muat model, preprocessing, prediksi, Grad-CAM
├── requirements.txt
├── run_windows.bat
├── run_linux_mac.sh
├── templates/
│   └── index.html        # Antarmuka website
├── static/
│   ├── styles.css        # Desain responsif
│   └── app.js            # Upload, API, prediksi, probabilitas
├── model/
│   └── README.txt        # Tempat meletakkan bobot model
└── tests/
    └── test_app.py       # Pengujian endpoint dasar
```

## 7. Endpoint API

- `GET /api/status`: informasi apakah bobot model tersedia.
- `POST /api/predict` dengan field multipart bernama `image`: menghasilkan grade, confidence, probabilitas, dan gambar Grad-CAM berformat data URI. Mengembalikan 503 jika model belum tersedia.

## 8. Privasi, keterbatasan, dan penggunaan penelitian

- File gambar diproses di memori server tanpa disimpan permanen oleh implementasi ini. Untuk produksi, tetap perlu HTTPS, autentikasi, kontrol akses, kebijakan privasi, audit, dan prosedur perlindungan data.
- Model dan tingkat akurasi **belum divalidasi ulang di lingkungan aplikasi ini**. Angka 75% berasal dari **hasil output notebook pengujian** yang diunggah, bukan dari benchmark website.
- Beberapa output training notebook berisi `loss: nan`; periksa proses training dan gunakan bobot yang benar. Kemungkinan prediksi menjadi terlalu yakin (*overconfident*) pada data tertentu; jangan menafsirkan confidence sebagai probabilitas klinis terkalibrasi.
- Model harus diuji terhadap data eksternal sebelum penggunaan klinis. Hasil tidak boleh dijadikan pengganti penilaian dokter/radiolog.
- **Jangan menggunakan server pengembangan Flask secara langsung untuk deployment publik**. Gunakan server produksi yang tepat, HTTPS, pembatasan akses, dan konsultasi keamanan.

## 9. Uji endpoint tanpa model

```bash
pip install pytest
pytest -q
```

Pengujian akan mengecek halaman utama, status model, unggah tidak valid, dan respons saat bobot model belum tersedia. Pengujian inference end-to-end memerlukan bobot model asli.
