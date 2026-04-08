import os
import torch
from torchvision import models, transforms
from PIL import Image
import gdown

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

MODEL_URL = "https://drive.google.com/uc?id=1oCKoxaMzdi2ffrVeGnKKTrOoIpt0W4vE"
MODEL_PATH = "marine_model.pth"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

def load_model(class_names):
    download_model()
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, len(class_names))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_image(model, class_names, image_path):
    image = Image.open(image_path).convert("RGB")
    img_t = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_t)
        _, predicted = torch.max(outputs, 1)
    return class_names[predicted.item()]
