
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import glob

st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Brain Tumor Detection")
st.write("Upload an MRI scan to get an AI model prediction.")

@st.cache_resource
def load_brain_model():
    model_files = glob.glob("*.keras") + glob.glob("*.h5")

    if not model_files:
        raise FileNotFoundError("Model file not found.")

    return tf.keras.models.load_model(model_files[0])

try:
    model = load_brain_model()

    with open("class_names.json", "r") as f:
        class_names = json.load(f)

    if isinstance(class_names, dict):
        class_names = list(class_names.values())

    st.success("AI Model Loaded Successfully!")

    uploaded_file = st.file_uploader(
        "Upload Brain MRI Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(image, caption="Uploaded MRI Image",
                 use_container_width=True)

        if st.button("Predict Tumor"):

            resized_image = image.resize((224, 224))

            img_array = np.array(resized_image) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)

            predicted_index = np.argmax(prediction[0])
            confidence = float(prediction[0][predicted_index]) * 100

            predicted_class = class_names[predicted_index]

            st.subheader("Prediction Result")

            st.success(
                f"Predicted Class: {predicted_class.upper()}"
            )

            st.write(f"Model confidence: {confidence:.2f}%")

            st.info(
                "This is an educational AI project. "
                "It is not a medical diagnosis. "
                "Consult a qualified medical professional."
            )

except Exception as e:
    st.error(f"Error: {e}")
