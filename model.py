import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CNN + Transformer backbone
class CNNTransformer(nn.Module):
    def __init__(self, num_classes=5):  # adjust number of marine animals
        super(CNNTransformer, self).__init__()
        # CNN backbone
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Transformer encoder
        self.pos_embedding = nn.Parameter(torch.randn(1, 128*8*8, 128))
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv(x)  # [B, C, H, W]
        B, C, H, W = x.shape
        x = x.view(B, C*H*W, 1).permute(2, 0, 1)  # [seq_len, batch, feature]
        x = x + self.pos_embedding[:, :x.size(0), :]
        x = self.transformer(x)
        x = x.mean(dim=0)
        x = self.fc(x)
        return x

# Initialize model
CLASS_NAMES = ["Dolphin", "Shark", "Whale", "Octopus", "Sea Turtle"]
model = CNNTransformer(num_classes=len(CLASS_NAMES)).to(device)

# Load pretrained weights if any (here we just assume randomly initialized)
# model.load_state_dict(torch.load("marine_model.pth", map_location=device))

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

# Predict function
def predict_image(img_input):
    if isinstance(img_input, str):
        response = requests.get(img_input)
        img = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        img = img_input

    img = transform(img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(img)
        prob = F.softmax(output, dim=1)
        conf, pred = torch.max(prob, 1)
    return CLASS_NAMES[pred.item()], conf.item()
