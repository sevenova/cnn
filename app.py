"""
Klasifikasi Citra CIFAR-10 — Tugas Kelompok 2, Artificial Intelligence.

Aplikasi menerima gambar (unggah file atau kamera), melakukan preprocessing yang sama
dengan notebook (center-crop -> 32x32 -> [0,1]), lalu menampilkan prediksi model CNN.
"""
import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

MODEL_DIR = Path(__file__).parent / "model"
MODEL_PATH = MODEL_DIR / "best_model.keras"
INFO_PATH = MODEL_DIR / "model_info.json"

DEFAULT_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]
DEFAULT_CLASSES_ID = ["pesawat", "mobil", "burung", "kucing", "rusa",
                      "anjing", "katak", "kuda", "kapal", "truk"]
EMOJI = ["✈️", "🚗", "🐦", "🐱", "🦌", "🐶", "🐸", "🐴", "🚢", "🚚"]

st.set_page_config(page_title="Klasifikasi CIFAR-10", page_icon="🖼️", layout="centered")


# ------------------------------------------------------------------ model
@st.cache_resource(show_spinner="Memuat model…")
def load_model():
    import tensorflow as tf  # impor di sini agar halaman tampil lebih cepat
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


@st.cache_data
def load_info():
    if INFO_PATH.exists():
        return json.loads(INFO_PATH.read_text())
    return {}


def preprocess(img: Image.Image, crop: bool = True) -> tuple[np.ndarray, Image.Image]:
    """Center-crop ke persegi (opsional), resize 32x32, normalisasi ke [0,1]. Sama dengan notebook."""
    img = img.convert("RGB")
    if crop:
        w, h = img.size
        s = min(w, h)
        img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    small = img.resize((32, 32), Image.Resampling.BOX)
    arr = np.asarray(small, dtype="float32") / 255.0
    return arr, small


# ------------------------------------------------------------------ sidebar
info = load_info()
classes = info.get("class_names", DEFAULT_CLASSES)
classes_id = info.get("class_names_id", DEFAULT_CLASSES_ID)

with st.sidebar:
    st.header("Tentang Model")
    if info:
        st.write(f"**{info.get('nama_model', 'CNN')}**")
        c1, c2 = st.columns(2)
        c1.metric("Akurasi uji", f"{info.get('akurasi_uji', 0):.1%}")
        c2.metric("F1 macro", f"{info.get('f1_macro', 0):.3f}")
        st.caption(f"{info.get('parameter', 0):,} parameter · TensorFlow {info.get('tensorflow_version', '-')}")
        if info.get("perbandingan"):
            st.write("**Perbandingan akurasi uji**")
            df = pd.DataFrame({"Model": list(info["perbandingan"].keys()),
                               "Akurasi": list(info["perbandingan"].values())})
            st.dataframe(df.style.format({"Akurasi": "{:.1%}"}), hide_index=True,
                         width="stretch")
    else:
        st.info("model_info.json belum ada.")

    st.divider()
    st.write("**10 kelas yang dikenali**")
    st.write("  ".join(f"{e} {c}" for e, c in zip(EMOJI, classes_id)))
    st.divider()
    crop = st.toggle("Potong tengah jadi persegi", value=True,
                     help="CIFAR-10 berbentuk persegi. Matikan jika objek tidak di tengah gambar.")
    top_k = st.slider("Jumlah prediksi teratas", 3, 10, 5)
    st.caption("Tugas Kelompok 2 · Artificial Intelligence · BINUS University Online")

# ------------------------------------------------------------------ main
st.title("🖼️ Klasifikasi Citra CIFAR-10")
st.write("Unggah foto berisi salah satu dari 10 objek di samping — pesawat, mobil, burung, kucing, "
         "rusa, anjing, katak, kuda, kapal, atau truk — dan model CNN akan menebak isinya.")

if not MODEL_PATH.exists():
    st.error("File `model/best_model.keras` tidak ditemukan. Jalankan notebook Week 8 lalu salin "
             "folder `model/` hasil ekspor ke repositori aplikasi ini.")
    st.stop()

tab_upload, tab_kamera = st.tabs(["📁 Unggah gambar", "📷 Kamera"])
with tab_upload:
    files = st.file_uploader("Pilih satu atau beberapa gambar (JPG/PNG/WEBP)",
                             type=["jpg", "jpeg", "png", "webp", "bmp"],
                             accept_multiple_files=True)
with tab_kamera:
    foto = st.camera_input("Ambil foto")

sumber = [(f.name, f.getvalue()) for f in (files or [])]
if foto is not None:
    sumber.append(("foto_kamera.jpg", foto.getvalue()))

if not sumber:
    st.info("Belum ada gambar. Tips: objek sebaiknya memenuhi sebagian besar bingkai, "
            "karena model dilatih dengan citra 32×32 piksel yang berfokus pada satu objek.")
    st.stop()

model = load_model()

for nama, data in sumber:
    try:
        img = Image.open(io.BytesIO(data))
    except Exception:
        st.warning(f"`{nama}` tidak bisa dibaca sebagai gambar.")
        continue

    arr, small = preprocess(img, crop=crop)
    t0 = time.perf_counter()
    prob = model.predict(arr[None, ...], verbose=0)[0]
    ms = (time.perf_counter() - t0) * 1000
    urut = np.argsort(prob)[::-1]
    top = int(urut[0])

    st.divider()
    st.subheader(nama)
    kiri, kanan = st.columns([1, 1.3])
    with kiri:
        st.image(img.convert("RGB"), caption="Gambar asli", width="stretch")
        st.image(small.resize((128, 128), Image.Resampling.NEAREST),
                 caption="Input model (32×32 piksel)", width=128)
    with kanan:
        yakin = float(prob[top])
        st.markdown(f"### {EMOJI[top]} {classes_id[top].capitalize()}")
        st.caption(f"Kelas: `{classes[top]}` · inferensi {ms:.0f} ms")
        st.progress(yakin, text=f"Keyakinan {yakin:.1%}")
        if yakin < 0.5:
            st.warning("Keyakinan rendah — gambar mungkin ambigu atau objeknya tidak termasuk 10 kelas.")

        df = pd.DataFrame({
            "Kelas": [f"{EMOJI[i]} {classes_id[i]}" for i in urut[:top_k]],
            "Probabilitas": prob[urut[:top_k]] * 100,
        })
        st.dataframe(
            df, hide_index=True, width="stretch",
            column_config={"Probabilitas": st.column_config.ProgressColumn(
                "Probabilitas", format="%.1f%%", min_value=0.0, max_value=100.0)},
        )
