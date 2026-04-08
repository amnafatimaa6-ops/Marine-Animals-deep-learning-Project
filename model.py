# model.py
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO

# --------- CONFIG ---------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Link to your final trained model weights
MODEL_URL = "https://drive.google.com/uc?id=1oCKoxaMzdi2ffrVeGnKKTrOoIpt0W4vE"

# File listing all marine classes
CLASSES_FILE = "marine_classes.txt"

IMG_SIZE = 224  # Image size used in final training

# -------------------------


# --------- LOAD CLASSES ---------
try:
    with open(CLASSES_FILE, "r") as f:
        CLASSES = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    raise FileNotFoundError(f"Please create {CLASSES_FILE} with all marine classes listed.")

NUM_CLASSES = len(CLASSES)


# --------- MODEL DEFINITION ---------
class CNN_Transformer(nn.Module):
    def __init__(self, cnn, num_classes):
        super().__init__()
        self.cnn = cnn
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=2)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        features = self.cnn(x)            # [batch, 512]
        features = features.unsqueeze(0)  # [seq_len=1, batch, 512] for transformer
        out = self.transformer(features)  # [seq_len=1, batch, 512]
        out = out.squeeze(0)              # [batch, 512]
        out = self.fc(out)
        return out


# Load CNN backbone
cnn = models.resnet18(pretrained=True)
cnn.fc = nn.Identity()  # Remove final classifier

# Initialize model
model = CNN_Transformer(cnn, NUM_CLASSES).to(DEVICE)
model.eval()


# --------- HELPER TO LOAD MODEL WEIGHTS FROM DRIVE ---------
import gdown
import os

MODEL_PATH = "marine_final_224_aug.pth"

if not os.path.exists(MODEL_PATH):
    print("Downloading model weights from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
print("Model loaded successfully.")


# --------- IMAGE TRANSFORM ---------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


# --------- PREDICTION FUNCTION ---------
def predict_image(img_path_or_url):
    """
    Predict marine animal class from image.
    Can use local path or image URL.
    Returns: class_name, confidence
    """
    # Load image
    if img_path_or_url.startswith("http"):
        response = requests.get(img_path_or_url)
        img = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        img = Image.open(img_path_or_url).convert("RGB")

    # Transform
    img_t = transform(img).unsqueeze(0).to(DEVICE)

    # Forward pass
    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)
        class_name = CLASSES[pred_idx.item()]
        confidence = conf.item()

    return class_name, confidence
