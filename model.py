import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import requests
from io import BytesIO

# Classes
CLASS_NAMES = ["Dolphin", "Shark", "Whale", "Octopus", "Sea Turtle"]

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dummy model: pretrained resnet + transformer head
class MarineClassifier(nn.Module):
    def __init__(self, num_classes=len(CLASS_NAMES)):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

# Initialize model
model = MarineClassifier().to(device)
model.eval()  # evaluation mode

# Transform for image
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_image(url):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img_t = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img_t)
            probs = torch.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, 1)
            pred_class = CLASS_NAMES[pred_idx.item()]
        return pred_class, conf.item()
    except Exception as e:
        return "Error loading image", 0.0
