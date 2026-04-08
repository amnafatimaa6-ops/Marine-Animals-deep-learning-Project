import streamlit as st
from model import predict_image, CLASS_NAMES
import requests
from PIL import Image

st.title("Marine Animal Image Classifier 🐬🦈🐙")

# Dropdown to select an animal
animal_choice = st.selectbox("Choose a marine animal to test:", CLASS_NAMES)

# Fetch an image from Wikipedia
wiki_urls = {
    "Dolphin": "https://upload.wikimedia.org/wikipedia/commons/3/3e/Common_dolphin_delphinus_delphis_2.jpg",
    "Shark": "https://upload.wikimedia.org/wikipedia/commons/f/f6/White_shark.jpg",
    "Whale": "https://upload.wikimedia.org/wikipedia/commons/5/56/Humpback_Whale_underwater_shot.jpg",
    "Octopus": "https://upload.wikimedia.org/wikipedia/commons/5/57/Octopus2.jpg",
    "Sea Turtle": "https://upload.wikimedia.org/wikipedia/commons/9/9c/Green_turtle_swimming_over_coral_reefs.jpg"
}

img_url = wiki_urls[animal_choice]

st.image(img_url, caption=f"Example of {animal_choice}", use_column_width=True)

# Predict button
if st.button("Predict"):
    pred, conf = predict_image(img_url)
    st.success(f"Prediction: {pred} ({conf*100:.2f}% confidence)")

    # Simple facts about the animal
    facts = {
        "Dolphin": "Dolphins are highly intelligent marine mammals known for their agility and playful behavior.",
        "Shark": "Sharks are a group of elasmobranch fish characterized by a cartilaginous skeleton.",
        "Whale": "Whales are large marine mammals that breathe air through a blowhole on top of their heads.",
        "Octopus": "Octopuses are soft-bodied, eight-limbed mollusks known for their intelligence and camouflage abilities.",
        "Sea Turtle": "Sea turtles are reptiles that spend most of their lives in the ocean and return to land to lay eggs."
    }
    st.info(facts[pred])
