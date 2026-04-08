import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# --------------------------
# Image transform
# --------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

# --------------------------
# CNN+Transformer Hybrid
# --------------------------
class CNN_Transformer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        cnn = models.resnet18(pretrained=True)
        cnn.fc = nn.Identity()
        self.cnn = cnn
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=2)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        features = self.cnn(x)            # [batch,512]
        features = features.unsqueeze(1)  # [batch,1,512]
        out = self.transformer(features)  # [batch,1,512]
        out = out.squeeze(1)              # [batch,512]
        out = self.fc(out)                # [batch,num_classes]
        return out


# --------------------------
# Load pretrained model
# --------------------------
MODEL_PATH = "marine_final_224_aug.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"
model = None
CLASS_NAMES = []  # filled in app.py after dataset is downloaded

def init_model(class_list):
    global model
    CLASS_NAMES.clear()
    CLASS_NAMES.extend(class_list)

    net = CNN_Transformer(len(class_list)).to(device)
    net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    net.eval()
    model = net
    return model


# --------------------------
# Predict function
# --------------------------
def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        pred = torch.argmax(outputs, dim=1).item()

    return CLASS_NAMES[pred]
