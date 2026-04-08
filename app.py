import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import wikipedia
from model import predict_image, CLASS_NAMES

st.title("Marine Animal Classifier 🌊🐠")

# 1. User picks animal
animal = st.selectbox("Choose a marine animal:", CLASS_NAMES)

# 2. Fetch image from Wikipedia
st.write(f"Fetching image of {animal} from Wikipedia...")
try:
    search_results = wikipedia.search(animal)
    page = wikipedia.page(search_results[0])
    # find first image from page
    image_url = page.images[0]
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    st.image(img, caption=f"{animal} from Wikipedia", use_column_width=True)
except Exception as e:
    st.error(f"Could not fetch image: {e}")
    img = None

# 3. Predict with model
if img:
    prediction = predict_image(img)
    st.success(f"Model Prediction: **{prediction}**")

# 4. Show Wikipedia info
try:
    summary = wikipedia.summary(animal, sentences=3)
    st.markdown(f"### About {animal}")
    st.write(summary)
except Exception as e:
    st.warning(f"Could not fetch info: {e}")
