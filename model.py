import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gdown
import os

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Download model from Google Drive if not exists
model_url = "https://drive.google.com/uc?id=1oCKoxaMzdi2ffrVeGnKKTrOoIpt0W4vE"
model_path = "marine_final_224_aug.pth"

if not os.path.exists(model_path):
    print("Downloading model...")
    gdown.download(model_url, model_path, quiet=False)
    print("Model downloaded.")

# CNN + Transformer hybrid (same as training)
class CNN_Transformer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        cnn = models.resnet18(pretrained=True)
        cnn.fc = nn.Identity()
        self.cnn = cnn
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=2)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        features = self.cnn(x)  # [batch, 512]
        features = features.unsqueeze(0)  # [seq_len=1, batch, 512]
        out = self.transformer(features)  # [seq_len, batch, dim]
        out = out.squeeze(0)  # [batch, dim]
        out = self.fc(out)
        return out

# You need a list of classes (from your marine.animals folders)
# We'll download from Google Drive if needed or manually define later
classes_file = "marine_classes.txt"
if not os.path.exists(classes_file):
    # Placeholder: user can manually create a text file with one class per line
    raise FileNotFoundError(f"Please create {classes_file} with all marine classes listed.")

with open(classes_file, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

num_classes = len(class_names)
model = CNN_Transformer(num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Transform for input images
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

# Prediction function
def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(x)
        pred = torch.argmax(outputs, dim=1).item()
    return class_names[pred]
