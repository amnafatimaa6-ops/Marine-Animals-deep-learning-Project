import streamlit as st
from model import predict_image, CLASS_NAMES
import wikipedia

st.set_page_config(page_title="Marine Animal Classifier", layout="centered")

st.title("🐬 Marine Animal Classifier + Wikipedia Info")
st.write("Upload an image of a marine animal, and the model will predict it and show info from Wikipedia.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

url_input = st.text_input("Or enter an image URL:")

img_path = uploaded_file if uploaded_file else url_input

if img_path:
    try:
        st.image(img_path, caption="Uploaded Image", use_column_width=True)
        pred, prob = predict_image(img_path)
        st.success(f"Prediction: **{pred}** ({prob*100:.2f}%)")
        
        # Fetch summary from Wikipedia
        try:
            summary = wikipedia.summary(pred, sentences=3)
            st.info(f"**About {pred}:**\n{summary}")
        except:
            st.warning(f"No Wikipedia info found for {pred}.")
            
    except Exception as e:
        st.error(f"Error: {e}")
