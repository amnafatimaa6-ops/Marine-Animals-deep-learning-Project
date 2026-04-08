import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO

# -------------------
# CNN + Transformer
# -------------------
class CNNTransformer(nn.Module):
    def __init__(self, num_classes=10, img_size=224, patch_size=16, dim=256, depth=4, heads=8, mlp_dim=512):
        super().__init__()
        self.num_classes = num_classes
        # CNN backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.patch_size = patch_size
        self.to_patch_embedding = nn.Linear(256 * patch_size * patch_size, dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, x):
        x = self.cnn(x)
        B, C, H, W = x.shape
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(B, -1, C * self.patch_size * self.patch_size)
        x = self.to_patch_embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.mlp_head(x)

# -------------------
# Classes & Model
# -------------------
CLASS_NAMES = [
    "Dolphin", "Shark", "Octopus", "Jellyfish", "Whale",
    "Sea Turtle", "Crab", "Seahorse", "Starfish", "Clownfish"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNTransformer(num_classes=len(CLASS_NAMES)).to(device)
model.eval()  # No trained weights for demo; can add online pretraining later

transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])

# -------------------
# Predict function from URL
# -------------------
def predict_image(img_url):
    response = requests.get(img_url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = probs.argmax().item()
        return CLASS_NAMES[pred_idx], probs[0, pred_idx].item()
