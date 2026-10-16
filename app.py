import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras

st.set_page_config(page_title="Clasificador CNN CIFAR-10", page_icon="🖼️", layout="centered")

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "modelo" / "cnn_cifar10.keras"
CLASSES_PATH = ROOT / "modelo" / "class_names.json"

st.title("Clasificador de imágenes con CNN")
st.write(
    "Carga una imagen para que la red neuronal intente clasificarla en una de las "
    "10 categorías de CIFAR-10."
)
st.info(
    "El modelo fue entrenado con imágenes pequeñas de 32 × 32 píxeles. "
    "Solo reconoce las categorías de CIFAR-10, por lo que los resultados con "
    "fotos muy diferentes al dataset pueden ser incorrectos."
)

@st.cache_resource
def load_model_and_classes():
    model = keras.models.load_model(MODEL_PATH)
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)
    return model, class_names

if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
    st.error("No se encontró el modelo.")
    st.stop()

uploaded_file = st.file_uploader("Selecciona una imagen", type=["png", "jpg", "jpeg", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen cargada", use_container_width=True)

    if st.button("Realizar predicción", type="primary"):
        with st.spinner("Analizando imagen..."):
            model, class_names = load_model_and_classes()
            resized = image.resize((32, 32))
            array = np.asarray(resized, dtype=np.float32) / 255.0
            probabilities = model.predict(np.expand_dims(array, axis=0), verbose=0)[0]
            best_index = int(np.argmax(probabilities))
            best_class = class_names[best_index]
            confidence = float(probabilities[best_index]) * 100

        st.subheader("Resultado")
        st.success(f"Clase predicha: {best_class}")
        st.metric("Confianza del modelo", f"{confidence:.2f}%")
        st.caption("La confianza es la probabilidad estimada por el modelo, no una garantía de que la predicción sea correcta.")

        st.subheader("Probabilidades por clase")
        ranking = np.argsort(probabilities)[::-1]
        for idx in ranking:
            st.write(f"{class_names[int(idx)]}: {probabilities[idx] * 100:.2f}%")
            st.progress(float(probabilities[idx]))
