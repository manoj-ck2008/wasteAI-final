import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model("waste_model_transfer.keras")

IMG_SIZE = 128

classes = [
    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Miscellaneous Trash",
    "Paper",
    "Plastic",
    "Textile Trash",
    "Vegetation"
]

# =========================
# UI
# =========================
st.set_page_config(page_title="Smart Waste Classifier", layout="centered")

st.title("♻️ Smart Waste Classification System")
st.write("Upload an image to classify waste and get disposal guidance.")

uploaded_file = st.file_uploader("Upload waste image", type=["jpg", "png", "jpeg"])

# =========================
# PREDICTION
# =========================
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0
    img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))

    prediction = model.predict(img)
    result = classes[np.argmax(prediction)]

    st.subheader(f"🧠 Detected: {result}")

    if result in ["Plastic", "Paper", "Metal", "Glass", "Cardboard"]:
        st.success("♻️ Category: Recyclable (Dry Waste)")
    elif result in ["Food Organics", "Vegetation"]:
        st.warning("🌱 Category: Wet Waste")
    else:
        st.error("🗑️ Category: General Trash")