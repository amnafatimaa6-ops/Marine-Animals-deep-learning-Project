import streamlit as st
from model import predict_image, CLASS_NAMES

st.title("Marine Animal Classifier 🐬🦈🐙")

# Dropdown to select an animal
animal_choice = st.selectbox("Choose a marine animal:", CLASS_NAMES)

# Wikipedia images
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

    facts = {
        "Dolphin": "Dolphins are intelligent marine mammals known for agility and playfulness.",
        "Shark": "Sharks are elasmobranch fish with cartilaginous skeletons.",
        "Whale": "Whales are large marine mammals breathing through a blowhole.",
        "Octopus": "Octopuses are soft-bodied, eight-limbed mollusks known for intelligence and camouflage.",
        "Sea Turtle": "Sea turtles are reptiles spending most of their lives in the ocean."
    }
    st.info(facts.get(pred, "No facts available."))
