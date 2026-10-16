# Klasifikasi Citra CIFAR-10 — Aplikasi Streamlit

Tugas Kelompok 2 (Week 8) · Artificial Intelligence

Aplikasi web sederhana yang menerima gambar (unggah file atau kamera) dan menampilkan prediksi
model CNN yang dilatih pada CIFAR-10, lengkap dengan tingkat keyakinan dan 5 prediksi teratas.

## Struktur folder

```
app/
├── app.py                  # aplikasi Streamlit
├── requirements.txt        # GANTI dengan file dari model_export.zip
├── .streamlit/config.toml  # tema dan batas ukuran unggahan
└── model/
    ├── best_model.keras    # dari notebook (model_export.zip)
    └── model_info.json     # dari notebook (model_export.zip)
```