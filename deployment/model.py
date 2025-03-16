import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

# Define the model architecture
class CropDiseaseModel(nn.Module):
    def __init__(self, num_classes=38):  # Adjust num_classes based on dataset
        super(CropDiseaseModel, self).__init__()
        self.model = models.resnet18(pretrained=False)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Load the model
def load_model(model_path="crop_disease_detection.pth", num_classes=38):
    model = CropDiseaseModel(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cuda")))
    model.eval()
    return model

# Image preprocessing function
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Predict function
def predict(image, model, class_names):
    image_tensor = transform_image(image)
    with torch.no_grad():
        output = model(image_tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    predicted_class = torch.argmax(probabilities).item()
    return class_names[predicted_class], probabilities[predicted_class].item()
