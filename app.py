import streamlit as st
import os
import gdown
import zipfile
from model import predict_image, CLASS_NAMES

# --- Download and unzip dataset ---
ZIP_URL = "https://drive.google.com/uc?id=12We6beOBig11JAZcl1gswVuow-TRKE-P"  # your dataset zip ID
ZIP_PATH = "marine.animals.zip"
DATA_FOLDER = "marine.animals"

if not os.path.exists(ZIP_PATH):
    st.info("Downloading marine dataset...")
    gdown.download(ZIP_URL, ZIP_PATH, quiet=False)

if not os.path.exists(DATA_FOLDER):
    st.info("Extracting dataset...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(".")

# --- Streamlit UI ---
st.title("🐠 Marine Animal Classifier")

uploaded_file = st.file_uploader("Upload an image of a marine animal", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Save uploaded file temporarily
    temp_path = os.path.join("temp.jpg")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Prediction
    st.image(temp_path, caption="Uploaded Image", use_column_width=True)
    st.write("Predicting...")
    prediction = predict_image(temp_path)
    st.success(f"This looks like: **{prediction}**")
