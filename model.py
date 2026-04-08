import os
import torch
from torchvision import models, transforms
from PIL import Image
import gdown

# --- Download the pre-trained model from Drive ---
MODEL_URL = "https://drive.google.com/uc?id=1oCKoxaMzdi2ffrVeGnKKTrOoIpt0W4vE"  # replace with your file ID
MODEL_PATH = "marine_model.pth"

if not os.path.exists(MODEL_PATH):
    print("Downloading pre-trained model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# --- Load dataset classes ---
DATA_FOLDER = "marine.animals"
if not os.path.exists(DATA_FOLDER):
    raise FileNotFoundError(
        f"Dataset folder '{DATA_FOLDER}' not found. Please zip and download it in app.py."
    )

CLASS_NAMES = [d for d in os.listdir(DATA_FOLDER) if os.path.isdir(os.path.join(DATA_FOLDER, d))]
CLASS_NAMES.sort()  # optional: consistent order

# --- Define model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# --- Image transforms ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- Prediction function ---
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    img_t = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_t)
        _, predicted = torch.max(outputs, 1)
    return CLASS_NAMES[predicted.item()]
