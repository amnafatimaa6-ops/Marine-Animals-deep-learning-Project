import os
import random
from PIL import Image
import wikipedia

from model import predict_image

# ------------------------------
# Dataset Path (Drive)
# ------------------------------
base_path = "/content/drive/MyDrive/marine.animals"

# ------------------------------
# Get ALL images from dataset
# ------------------------------
all_images = []

for root, dirs, files in os.walk(base_path):
    for f in files:
        if f.lower().endswith((".jpg", ".png", ".jpeg")):
            all_images.append(os.path.join(root, f))

print(f"Total images available: {len(all_images)}")

# ------------------------------
# Pick RANDOM image
# ------------------------------
img_path = random.choice(all_images)

print(f"\n📂 Selected Image:\n{img_path}")

image = Image.open(img_path).convert("RGB")

# ------------------------------
# Predict using your model
# ------------------------------
prediction = predict_image(image)

print(f"\n🧠 Model Prediction: {prediction}")

# ------------------------------
# Wikipedia Info
# ------------------------------
def get_info(name):
    try:
        return wikipedia.summary(name, sentences=2)
    except:
        return "No info found on Wikipedia."

info = get_info(prediction)

print(f"\n📚 About {prediction}:\n{info}")
