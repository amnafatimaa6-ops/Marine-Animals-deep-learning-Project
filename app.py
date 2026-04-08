import streamlit as st
from model import predict_image
from PIL import Image
import gdown
import os

st.set_page_config(page_title="Marine Animal Classifier", layout="wide")

st.title("🐠 Marine Animal Classifier")
st.write("Upload an image of a marine animal, and the model will tell you what it is and some info from Wikipedia.")

# Sidebar: Upload image
uploaded_file = st.file_uploader("Choose a marine animal image", type=["jpg","png","jpeg"])

# Optional: Google Drive dataset browsing
st.sidebar.write("Or pick an image from the dataset")
dataset_url = "https://drive.google.com/uc?id=12We6beOBig11JAZcl1gswVuow-TRKE-P"  # dataset folder (zip?) -> needs manual download
st.sidebar.markdown(f"[Download dataset folder here]({dataset_url})")

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Predicting..."):
        label = predict_image(uploaded_file)
        st.success(f"Prediction: **{label}**")

        # Wikipedia info
        try:
            import wikipedia
            summary = wikipedia.summary(label, sentences=3)
            st.markdown(f"**About {label}:** {summary}")
        except Exception as e:
            st.write("Could not fetch Wikipedia info:", e)
