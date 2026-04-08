import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO
import requests

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Classes
CLASS_NAMES = ["Dolphin", "Shark", "Whale", "Octopus", "Sea Turtle"]

# Transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Simple CNN + Transformer stub model
class SimpleCNNTransformer(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32*32*32, 128)
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = self.flatten(x)
        x = self.fc1(x).unsqueeze(0)  # shape (seq_len=1, batch, features)
        x = self.transformer(x)
        x = x.squeeze(0)
        x = self.fc2(x)
        return x

# Instantiate model
model = SimpleCNNTransformer(len(CLASS_NAMES)).to(device)
model.eval()  # set to eval

# Dummy predict function
def predict_image(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img_t = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img_t)
            probs = torch.softmax(outputs, dim=1)
            conf, idx = torch.max(probs, 1)
            return CLASS_NAMES[idx.item()], conf.item()
    except Exception as e:
        print("Error:", e)
        return "Error loading image", 0.0
