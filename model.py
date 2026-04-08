import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ------------------------------
# Device
# ------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------
# Classes (MUST match training order)
# ------------------------------
classes = [
    "Clams","Corals","Crabs","Dolphin","Eel","Fish","Jelly Fish",
    "Lobster","Nudibranchs","Octopus","Otter","Penguin","Puffers",
    "Sea Rays","Sea Urchins","Seahorse","Seal","Sharks","Shrimp",
    "Squid","Starfish","Turtle_Tortoise","Whale"
]

# ------------------------------
# Model Architecture (same as training)
# ------------------------------
cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
cnn.fc = nn.Identity()

class CNN_Transformer(nn.Module):
    def __init__(self, cnn, num_classes):
        super().__init__()
        self.cnn = cnn
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=2)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x).unsqueeze(1)
        x = self.transformer(x)
        x = x[:, 0, :]
        return self.fc(x)

# ------------------------------
# Load BEST MODEL
# ------------------------------
model = CNN_Transformer(cnn, len(classes)).to(device)

model.load_state_dict(torch.load(
    "/content/drive/MyDrive/marine_final_224_aug.pth",  # ✅ YOUR BEST MODEL
    map_location=device
))

model.eval()

# ------------------------------
# Transform (224 SAME AS TRAINING)
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ------------------------------
# Prediction Function
# ------------------------------
def predict_image(image):
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        pred = torch.argmax(outputs, dim=1).item()

    return classes[pred]
