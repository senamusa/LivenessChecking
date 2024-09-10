from fastapi import FastAPI, UploadFile, File, HTTPException
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision import transforms
from PIL import Image
from io import BytesIO
import json

app = FastAPI()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load target names
with open('target_names.json', 'r') as f:
    target_names = json.load(f)

# Define the SpoofNet model class
class SpoofNet(nn.Module):
    def __init__(self):
        super(SpoofNet, self).__init__()
        self.mobilenet = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        self.features = self.mobilenet.features
        self.conv2d = nn.Conv2d(1280, 32, kernel_size=(3, 3), padding=1)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.features(x)
        x = self.conv2d(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

# Define the FaceRecogNet model class
class FaceRecogNet(nn.Module):
    def __init__(self, num_classes):
        super(FaceRecogNet, self).__init__()
        self.mobilenet = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        self.mobilenet.classifier = nn.Sequential(
            nn.Linear(self.mobilenet.last_channel, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.mobilenet(x)

# Instantiate the models
model_spoof = SpoofNet().to(device)
model_face = FaceRecogNet(num_classes=len(target_names)).to(device)

# Load the model state dictionaries
model_spoof.load_state_dict(torch.load('spoof_weight.pth', map_location=device))
model_face.load_state_dict(torch.load('face_recognition_weight.pth', map_location=device))

model_spoof.eval()
model_face.eval()

# Define image transformations
transform_spoof = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[105.32685981,  91.9504575 ,  91.54538125], std=[63.06708699, 58.47346108, 58.98859229])
])

transform_face = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.post("/spoof")
async def spoof(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image_tensor = transform_spoof(image).unsqueeze(0).to(device)

        threshold = 0.5
        with torch.no_grad():
            outputs = model_spoof(image_tensor)
            preds = (outputs.squeeze() > threshold).item()
        
        return {"Spoof": bool(preds)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image_tensor = transform_face(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model_face(image_tensor)
            _, predicted = torch.max(outputs, 1)

        class_index = predicted.item()
        name = target_names[class_index]
        
        return {"Name": name}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

# uvicorn main:app --reload