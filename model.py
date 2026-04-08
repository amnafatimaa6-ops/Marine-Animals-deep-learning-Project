import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gdown
import os

# ---------------------------
# Load model from Google Drive
# ---------------------------

MODEL_DRIVE_URL = "https://drive.google.com/uc?id=1oCKoxaMzdi2ffrVeGnKKTrOoIpt0W4vE"
MODEL_PATH = "marine_final_224_aug.pth"

# Download if not present
if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_DRIVE_URL, MODEL_PATH, quiet=False)

# Dummy: load class names from folder (will be updated in app.py)
CLASS_NAMES = []

# CNN + Transformer hybrid (same as training)
class CNN_Transformer(nn.Module):
    def __init__(self, cnn, num_classes):
        super().__init__()
        self.cnn = cnn
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=2)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        features = self.cnn(x)
        features = features.unsqueeze(0)
        out = self.transformer(features)
        out = out.squeeze(0)
        out = self.fc(out)
        return out

# Placeholder model, will load weights after knowing classes
device = "cuda" if torch.cuda.is_available() else "cpu"
cnn = models.resnet18(pretrained=True)
cnn.fc = nn.Identity()
model = None  # will initialize later after class count

# Transform for inference
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

def init_model(num_classes):
    global model
    model = CNN_Transformer(cnn, num_classes).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

# Prediction function
def predict_image(image: Image.Image):
    img = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        preds = torch.argmax(outputs, dim=1).item()
    return CLASS_NAMES[preds]
