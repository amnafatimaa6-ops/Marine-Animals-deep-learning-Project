import streamlit as st
import wikipedia
from model import predict_image, CLASS_NAMES

st.set_page_config(page_title="Marine Animal Classifier", layout="centered")
st.title("🌊 Marine Animal Classifier + Wikipedia Info")
st.write("Pick an animal from the list below, the app fetches an image from Wikipedia and predicts it.")

# -------------------
# User picks an animal
# -------------------
animal_choice = st.selectbox("Choose a marine animal:", CLASS_NAMES)

# Use Wikipedia image
@st.cache_data
def fetch_wiki_image(animal_name):
    try:
        page = wikipedia.page(animal_name)
        images = [img for img in page.images if img.lower().endswith(("jpg","jpeg","png"))]
        if images:
            return images[0]  # First image
    except:
        return None
    return None

img_url = fetch_wiki_image(animal_choice)

if img_url:
    st.image(img_url, caption=f"Example {animal_choice}", use_column_width=True)
    pred, prob = predict_image(img_url)
    st.success(f"Prediction: **{pred}** ({prob*100:.2f}%)")

    # Wikipedia summary
    try:
        summary = wikipedia.summary(pred, sentences=3)
        st.info(f"**About {pred}:** {summary}")
    except:
        st.warning(f"No Wikipedia info found for {pred}.")
else:
    st.error("Could not fetch an image from Wikipedia for this animal.")
