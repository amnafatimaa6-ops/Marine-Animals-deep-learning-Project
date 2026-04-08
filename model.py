import streamlit as st
from model import predict_image, CLASS_NAMES
import wikipedia
from PIL import Image
import requests
from io import BytesIO

st.set_page_config(page_title="Marine Animal Classifier")

st.title("🌊 Marine Animal Classifier")
st.write("Select an animal and see what the model predicts!")

# Function to fetch valid Wikipedia image
@st.cache_data
def fetch_wiki_image(animal_name):
    try:
        page = wikipedia.page(animal_name)
        images = [img for img in page.images if img.lower().endswith((".jpg",".jpeg",".png"))]
        for img_url in images:
            try:
                response = requests.get(img_url, timeout=5)
                img = Image.open(BytesIO(response.content)).convert("RGB")
                return img_url
            except:
                continue
    except:
        return None
    return None

# User selects an animal
selected_animal = st.selectbox("Pick a marine animal:", CLASS_NAMES)

if st.button("Show image & Predict"):
    img_url = fetch_wiki_image(selected_animal)
    if img_url:
        st.image(img_url, caption=f"Example {selected_animal}", use_column_width=True)
        pred, conf = predict_image(img_url)
        st.success(f"Predicted: {pred} ({conf*100:.1f}% confidence)")
    else:
        st.error("Could not fetch an image from Wikipedia.")
