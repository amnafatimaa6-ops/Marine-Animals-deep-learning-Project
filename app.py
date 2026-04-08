import streamlit as st
import os
import zipfile
import gdown
from model import init_model, predict_image

# -----------------------------
# Step 1: Download & extract dataset zip
# -----------------------------
ZIP_URL = "https://drive.google.com/uc?id=1gu5_SlVbtl0YKoVH0y9rSGl_I1URieid"  # your new zipped dataset
ZIP_PATH = "marine_animals.zip"
DATA_FOLDER = "marine.animals"

st.title("🐠 Marine Animal Classifier")

# Download dataset if not exists
if not os.path.exists(ZIP_PATH):
    with st.spinner("Downloading Marine Animals dataset..."):
        gdown.download(ZIP_URL, ZIP_PATH, quiet=False)

# Extract if not exists
if not os.path.exists(DATA_FOLDER):
    with st.spinner("Extracting dataset..."):
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(".")

# Get class names from folder
CLASS_NAMES = sorted([d for d in os.listdir(DATA_FOLDER) if os.path.isdir(os.path.join(DATA_FOLDER, d))])

st.sidebar.write("Detected classes:")
st.sidebar.write(CLASS_NAMES)

# -----------------------------
# Step 2: Setup model
# -----------------------------
# Download the model file from Drive if not already
MODEL_DRIVE_URL = "https://drive.google.com/uc?id=1oCKoxaMzdi2ffrVeGnKKTrOoIpt0W4vE"
MODEL_FILE = "marine_final_224_aug.pth"

if not os.path.exists(MODEL_FILE):
    with st.spinner("Downloading model..."):
        gdown.download(MODEL_DRIVE_URL, MODEL_FILE, quiet=False)

# Initialize model with class list
init_model(CLASS_NAMES)

# -----------------------------
# Step 3: Image upload & prediction
# -----------------------------
uploaded = st.file_uploader("Upload a marine animal image", type=["jpg","png","jpeg"])

if uploaded:
    # Save temporary
    temp_path = os.path.join("temp.jpg")
    with open(temp_path, "wb") as f:
        f.write(uploaded.getbuffer())

    st.image(temp_path, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Predicting..."):
        result = predict_image(temp_path)

    st.success(f"**Prediction:** {result}")
