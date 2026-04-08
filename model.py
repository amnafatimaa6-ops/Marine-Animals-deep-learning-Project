import torch
from torchvision import transforms, models
from PIL import Image
import requests
from io import BytesIO

# Load pretrained model from Drive
MODEL_PATH = "marine_model.pth"  # change if downloaded differently
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=False)
num_classes = 10  # set this to number of animals in your model
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Class names
CLASS_NAMES = [
    "Dolphin", "Shark", "Octopus", "Jellyfish", "Whale",
    "Sea Turtle", "Crab", "Seahorse", "Starfish", "Clownfish"
]

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

def predict_image(image: Image.Image):
    img_t = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_t)
        _, pred = torch.max(outputs, 1)
        return CLASS_NAMES[pred.item()]
