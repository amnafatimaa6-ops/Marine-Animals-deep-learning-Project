import streamlit as st
from PIL import Image
import os
from model import predict_image, DATA_FOLDER, CLASS_NAMES

st.set_page_config(page_title="Marine Animal Classifier", layout="wide")
st.title("🐟 Marine Animal Classifier")

# -----------------------------
# IMAGE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("Upload a marine animal image", type=["jpg","jpeg","png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Predict
    st.write("Predicting...")
    class_name, confidence = predict_image(img)
    st.success(f"Predicted: **{class_name}** ({confidence*100:.2f}%)")

# -----------------------------
# SHOW AVAILABLE CLASSES
# -----------------------------
with st.expander("Available Classes"):
    st.write(CLASS_NAMES)

# -----------------------------
# OPTIONAL: SAMPLE IMAGE PICKER
# -----------------------------
st.markdown("---")
st.write("Or pick a sample image from dataset:")

import random
sample_class = st.selectbox("Choose Class", CLASS_NAMES)
sample_class_folder = os.path.join(DATA_FOLDER, sample_class)
sample_images = [os.path.join(sample_class_folder, f) for f in os.listdir(sample_class_folder) if f.lower().endswith((".jpg",".png",".jpeg"))]

if sample_images:
    sample_img_path = random.choice(sample_images)
    sample_img = Image.open(sample_img_path)
    st.image(sample_img, caption=f"Sample from {sample_class}", use_column_width=True)

    if st.button("Predict Sample Image"):
        class_name, confidence = predict_image(sample_img)
        st.success(f"Predicted: **{class_name}** ({confidence*100:.2f}%)")
