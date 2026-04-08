import streamlit as st
from PIL import Image
import os
from model import predict_image, init_model, CLASS_NAMES

import gdown

# -------------------------------
# Download and access dataset folder
# -------------------------------
DATA_DRIVE_URL = "https://drive.google.com/uc?id=12We6beOBig11JAZcl1gswVuow-TRKE-P"
DATA_FOLDER = "marine.animals"

if not os.path.exists(DATA_FOLDER):
    gdown.download_folder(DATA_DRIVE_URL, quiet=False, output=DATA_FOLDER)

# Build class list
CLASS_NAMES.clear()
CLASS_NAMES.extend([d for d in os.listdir(DATA_FOLDER) if os.path.isdir(os.path.join(DATA_FOLDER,d))])

# Initialize model now that classes are known
init_model(len(CLASS_NAMES))

# -------------------------------
# Streamlit App
# -------------------------------
st.title("Marine Animal Classifier 🐠")

uploaded_file = st.file_uploader("Upload an image of a marine animal", type=["png","jpg","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Predicting...")

    label = predict_image(image)
    st.success(f"Predicted: {label}")

    # Wikipedia info
    try:
        import wikipedia
        summary = wikipedia.summary(label, sentences=2)
        st.info(summary)
    except:
        st.info("No Wikipedia info found.")
