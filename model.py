import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO

# -------------------
# Model Definition
# -------------------
class CNNTransformer(nn.Module):
    def __init__(self, num_classes=10, img_size=224, patch_size=16, dim=256, depth=4, heads=8, mlp_dim=512):
        super().__init__()
        self.num_classes = num_classes
        
        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        
        # Flatten into patches for transformer
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2
        self.to_patch_embedding = nn.Linear(256 * patch_size * patch_size, dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Classification head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
    
    def forward(self, x):
        # x shape: (B, 3, H, W)
        x = self.cnn(x)  # (B, 256, H/8, W/8)
        B, C, H, W = x.shape
        
        # Make patches
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(B, C, -1, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, num_patches, C, p, p)
        x = x.view(B, -1, C * self.patch_size * self.patch_size)
        
        # Patch embeddings
        x = self.to_patch_embedding(x)  # (B, num_patches, dim)
        
        # Transformer
        x = self.transformer(x)  # (B, num_patches, dim)
        x = x.mean(dim=1)  # Global average pooling
        x = self.mlp_head(x)
        return x

# -------------------
# Utility: Prediction
# -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example class names (replace with your animals)
CLASS_NAMES = [
    "Dolphin", "Shark", "Octopus", "Jellyfish", "Whale",
    "Sea Turtle", "Crab", "Seahorse", "Starfish", "Clownfish"
]

# Initialize model
model = CNNTransformer(num_classes=len(CLASS_NAMES)).to(device)

# Load trained weights (replace MODEL_PATH with your trained .pth)
MODEL_PATH = "marine_best_model.pth"
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
except:
    print("⚠️ No trained weights found, using random weights (for demo only)")

model.eval()

# -------------------
# Image Prediction
# -------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict_image(img_path_or_url):
    if img_path_or_url.startswith("http"):
        response = requests.get(img_path_or_url)
        img = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        img = Image.open(img_path_or_url).convert("RGB")
    
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img)
        probs = torch.softmax(logits, dim=1)
        pred_idx = probs.argmax().item()
        return CLASS_NAMES[pred_idx], probs[0, pred_idx].item()
