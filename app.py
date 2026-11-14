

import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import os

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Brain Tumor Detection")
st.write("Upload a brain MRI image to get an AI model prediction.")

MODEL_PATH = "brain_tumor_model.keras"
CLASS_PATH = "class_names.json"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

try:
    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found in GitHub repository.")
        st.stop()

    if not os.path.exists(CLASS_PATH):
        st.error("class_names.json not found.")
        st.stop()

    model = load_model()

    with open(CLASS_PATH, "r") as f:
        class_names = json.load(f)

    st.success("AI Model Loaded Successfully!")

    uploaded_file = st.file_uploader(
        "Upload Brain MRI Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded MRI Image",
            use_container_width=True
        )

        if st.button("Predict Tumor"):

            resized_image = image.resize((224, 224))

            img_array = np.array(
                resized_image,
                dtype=np.float32
            )

            img_array = np.expand_dims(img_array, axis=0)

            # Same preprocessing used during Colab training
            img_array = preprocess_input(img_array)

            prediction = model.predict(
                img_array,
                verbose=0
            )

            predicted_index = int(np.argmax(prediction[0]))

            predicted_class = class_names[predicted_index]

            confidence = float(
                prediction[0][predicted_index]
            ) * 100

            st.subheader("Prediction Result")

            st.success(
                f"Predicted Class: {predicted_class.upper()}"
            )

            st.write(
                f"Model confidence: {confidence:.2f}%"
            )

            st.info(
                "This is an educational AI project only. "
                "It is not a medical diagnosis. "
                "Consult a qualified medical professional."
            )

except Exception as e:
    st.error(f"Error: {e}")
