# app.py
import streamlit as st
from model import predict_image
import os
from PIL import Image

st.title("Marine Animals Classifier 🐠🦀🐙")

# ----------- LOAD DATA FOLDER -------------
DATA_FOLDER = "marine.animals"  # folder you linked from Drive

if not os.path.exists(DATA_FOLDER):
    st.error(f"Dataset folder '{DATA_FOLDER}' not found. Please upload it or mount it.")
    st.stop()

# List all images grouped by class folders
classes = sorted(os.listdir(DATA_FOLDER))
st.sidebar.header("Available Animal Classes")
for c in classes:
    st.sidebar.write(f"- {c}")

# Allow user to pick a class
selected_class = st.sidebar.selectbox("Pick an animal class to explore", classes)

class_folder = os.path.join(DATA_FOLDER, selected_class)
images_in_class = [f for f in os.listdir(class_folder) if f.lower().endswith(("png", "jpg", "jpeg"))]

st.write(f"**Images in {selected_class}:**")
selected_image = st.selectbox("Pick an image to classify", images_in_class)

img_path = os.path.join(class_folder, selected_image)
img = Image.open(img_path)
st.image(img, caption=f"Selected: {selected_image}", use_column_width=True)

# Predict button
if st.button("Predict"):
    st.write("Running prediction...")
    class_name, confidence = predict_image(img_path)
    st.success(f"Predicted Class: **{class_name}**  |  Confidence: {confidence:.2f}")
