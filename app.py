import streamlit as st
from model import predict_image, CLASS_NAMES

st.set_page_config(page_title="Marine Animal Classifier", layout="centered")
st.title("Marine Animal Classifier 🐬🦈🐙")

# Dropdown to select an animal
animal_choice = st.selectbox("Choose a marine animal:", CLASS_NAMES)

# Wikipedia image URLs
wiki_urls = {
    "Dolphin": "https://upload.wikimedia.org/wikipedia/commons/3/3e/Common_dolphin_delphinus_delphis_2.jpg",
    "Shark": "https://upload.wikimedia.org/wikipedia/commons/f/f6/White_shark.jpg",
    "Whale": "https://upload.wikimedia.org/wikipedia/commons/5/56/Humpback_Whale_underwater_shot.jpg",
    "Octopus": "https://upload.wikimedia.org/wikipedia/commons/5/57/Octopus2.jpg",
    "Sea Turtle": "https://upload.wikimedia.org/wikipedia/commons/9/9c/Green_turtle_swimming_over_coral_reefs.jpg"
}

img_url = wiki_urls[animal_choice]

st.image(img_url, caption=f"Example of {animal_choice}", use_column_width=True)

if st.button("Predict"):
    pred, conf = predict_image(img_url)
    st.success(f"Prediction: {pred} ({conf*100:.2f}% confidence)")

    # Simple facts
    facts = {
        "Dolphin": "Dolphins are intelligent marine mammals, playful and agile.",
        "Shark": "Sharks are predatory fish with cartilaginous skeletons.",
        "Whale": "Whales are large marine mammals breathing through a blowhole.",
        "Octopus": "Octopuses are eight-limbed mollusks, masters of camouflage.",
        "Sea Turtle": "Sea turtles are reptiles living mostly in oceans, laying eggs on land."
    }

    if pred in facts:
        st.info(facts[pred])
    else:
        st.info("No facts available.")
